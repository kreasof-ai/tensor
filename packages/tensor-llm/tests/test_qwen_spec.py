"""Independent accepted-prefix and recurrent rollback qualification."""
import os
import numpy as np
import pytest
from tensor_llm.speculative.acceptance import accepted_prefix
from tensor_llm.speculative.lookup import OutputLookup


def test_acceptance_commits_pending_input_and_emits_corrected_bonus():
    inputs=np.array([[10,11,12,13],[20,21,22,23],[30,31,32,33],[40,41,42,43],[0,0,0,0]],'int32')
    predictions=np.array([[11,12,13,14],[99,22,23,24],[31,98,33,34],[41,42,97,44],[-1,-1,-1,-1]],'int32')
    counts,outputs=accepted_prefix(inputs,predictions,np.array([4,4,4,3,0],'int32'))
    np.testing.assert_array_equal(counts,[4,1,2,3,0])
    assert outputs==[[11,12,13,14],[99],[31,98],[41,42,97],[]]


def test_output_lookup_uses_only_confirmed_past_transitions_and_is_bounded():
    lookup=OutputLookup(min_context=2,max_context=4,capacity=12)
    for token in [1,2,3,1,2]:lookup.append(token)
    assert lookup.propose(3) is None
    for token in [3,1,2,3,1,2]:lookup.append(token)
    assert lookup.propose(3)==[3,1,2]
    lookup.append(99)
    assert lookup.propose(1) is None
    for token in range(100,140):lookup.append(token)
    assert len(lookup.entries)<=12 and len(lookup.history)<=4


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native speculative rollback qualification')
@pytest.mark.parametrize('chunk',[2,4,8])
def test_recurrent_snapshots_restore_every_rejection_depth(tmp_path,chunk):
    import torch,tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    torch.set_num_threads(1)
    slots,heads,d=8,32,128;rows=slots*chunk
    rng=np.random.default_rng(817)
    initial=rng.normal(0,.03,(slots,heads,d,d)).astype('float32')
    q=rng.normal(size=(rows,heads,d)).astype('float32');k=rng.normal(size=q.shape).astype('float32')
    q/=np.sqrt((q*q).sum(-1,keepdims=True)+1e-6)*np.sqrt(d)
    k/=np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
    v=rng.normal(size=q.shape).astype('float32')
    g=-rng.uniform(.01,1,(rows,heads)).astype('float32');beta=rng.uniform(0,1,g.shape).astype('float32')
    lengths=np.array([chunk,chunk,chunk,chunk,chunk-1,max(chunk-2,0),1,0],'int32')
    accepted=np.array([1,min(2,chunk),max(chunk-1,1),chunk,1,max(chunk-2,0),1,0],'int32')
    expected=torch.from_numpy(initial.copy());tq,tk,tv,tg,tb=map(torch.from_numpy,(q,k,v,g,beta))
    for slot in range(slots):
        for step in range(accepted[slot]):
            row=slot*chunk+step;decayed=expected[slot]*tg[row].exp()[:,None,None]
            delta=tv[row]-(decayed@tk[row].unsqueeze(-1)).squeeze(-1)
            expected[slot]=decayed+torch.einsum('hv,hk->hvk',delta*tb[row,:,None],tk[row])
    def artifact(kind,p):
        entry=tmp_path/(kind+'.py');out=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.speculative','make_kernel',kind,p,dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,out,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9');return out
    scan=artifact('gdn_scan',dict(slots=slots,chunk=chunk))
    restore=artifact('restore',dict(slots=slots,chunk=chunk,shape=[heads,d,d]))
    with tensor.Device() as dev:
        inputs=[dev.from_numpy(a) for a in (q,k,v,g,beta,lengths,initial)]
        out=dev.empty(q.shape);saved=dev.empty((slots,chunk-1,heads,d,d))
        dev.load(scan).launch(*inputs,out,saved)
        dev.load(restore).launch(saved,inputs[-1],dev.from_numpy(accepted),inputs[-2])
        np.testing.assert_allclose(inputs[-1].to_numpy(),expected.numpy(),rtol=4e-5,atol=5e-6)
        np.testing.assert_array_equal(inputs[-1].to_numpy()[lengths==0],initial[lengths==0])


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native speculative history qualification')
def test_convolution_history_restore_keeps_only_accepted_inputs(tmp_path):
    import tensor,torch
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    slots,chunk,channels=8,4,8192;rng=np.random.default_rng(917)
    initial=rng.normal(size=(slots,channels,3)).astype('float32')
    x=rng.normal(size=(slots*chunk,channels)).astype('float32')
    rounded=torch.from_numpy(x).bfloat16().float().numpy().reshape(slots,chunk,channels)
    w=torch.from_numpy(rng.normal(size=(channels,1,4)).astype('float32')).bfloat16().float().numpy()
    lengths=np.array([4,4,4,4,3,2,1,0],'int32');accepted=np.array([1,2,3,4,1,2,1,0],'int32')
    expected=initial.copy()
    for slot,count in enumerate(accepted):
        for step in range(count):
            expected[slot,:,:2]=expected[slot,:,1:]
            expected[slot,:,2]=rounded[slot,step]
    def artifact(kind,p):
        entry=tmp_path/(kind+'.py');out=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.speculative','make_kernel',kind,p,dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,out,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9');return out
    conv=artifact('gdn_conv',dict(slots=slots,chunk=chunk))
    restore=artifact('restore',dict(slots=slots,chunk=chunk,shape=[channels,3]))
    with tensor.Device() as dev:
        xx=dev.from_numpy(x);ww=dev.from_numpy(w,dtype='bfloat16');state=dev.from_numpy(initial)
        saved=dev.empty((slots,chunk-1,channels,3));ll=dev.from_numpy(lengths)
        dev.load(conv).launch(xx,ww,ll,state,dev.empty(x.shape),saved)
        dev.load(restore).launch(saved,state,dev.from_numpy(accepted),ll)
        np.testing.assert_array_equal(state.to_numpy(),expected)


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native speculative attention qualification')
@pytest.mark.parametrize('chunk',[2,4,8])
def test_split_attention_preserves_causality_at_partition_boundaries(tmp_path,chunk):
    import tensor,torch
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    from test_qwen_fp8_kv import quantize
    torch.set_num_threads(1)
    s,c,cap,d=8,chunk,128,256;rows=s*c;splits=16;rng=np.random.default_rng(5119)
    q=torch.from_numpy(rng.normal(size=(rows,16,d)).astype('float32')).bfloat16()
    k=torch.from_numpy(rng.normal(size=(s,2,cap,d)).astype('float32')).bfloat16()
    v=torch.from_numpy(rng.normal(size=k.shape).astype('float32')).bfloat16()
    kb,ks,k=quantize(k);vb,vs,v=quantize(v)
    proj=rng.normal(size=(rows,8192)).astype('float32')
    pos=np.array([120,63,64,0,1,15,25,100],'int32');lengths=np.minimum(c,np.array([8,8,3,8,0,2,1,8],'int32'))
    active=(np.arange(c)[None,:]<lengths[:,None]).astype('int32').reshape(-1)
    want=np.zeros((rows,4096),'float32')
    for slot in range(s):
        for t in range(lengths[slot]):
            row=slot*c+t;count=int(pos[slot])+t+1
            kk=k[slot,:,:count].float().repeat_interleave(8,0);vv=v[slot,:,:count].float().repeat_interleave(8,0)
            prob=(torch.einsum('hd,htd->ht',q[row].float(),kk)/16).softmax(-1).bfloat16().float()
            value=torch.einsum('ht,htd->hd',prob,vv).bfloat16().float()
            gate=torch.from_numpy(proj[row]).bfloat16().float().reshape(16,512)[:,256:]
            want[row]=(value*gate.sigmoid()).bfloat16().float().reshape(4096).numpy()
    artifacts=[]
    for kind in ('partial','merge'):
        entry=tmp_path/(kind+'.py');out=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.speculative_attention',kind,
            dict(slots=s,chunk=c,capacity=cap,splits=splits,packed_loads=True,key_rows=32),dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,out,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9');artifacts.append(out)
    with tensor.Device() as dev:
        qq=dev.from_numpy(q.float().numpy(),dtype='bfloat16')
        caches=[dev.from_numpy(x.numpy()) for x in (kb,vb,ks,vs)]
        parts=dev.empty((s,2,splits,c*8,d));stats=dev.empty((s,2,splits,c*8,2))
        dev.load(artifacts[0]).launch(qq,*caches,dev.from_numpy(pos),dev.from_numpy(lengths),parts,stats)
        out=dev.empty(want.shape,'bfloat16')
        dev.load(artifacts[1]).launch(parts,stats,dev.from_numpy(proj),dev.from_numpy(active),out)
        got=out.to_numpy();assert np.isfinite(got).all()
        assert np.linalg.norm(got-want)/np.linalg.norm(want)<.005
        np.testing.assert_allclose(got,want,rtol=.025,atol=.025)


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native speculative expert qualification')
@pytest.mark.parametrize('routed',[False,True])
@pytest.mark.parametrize('block_m',[16,32])
def test_split_experts_preserve_hot_routes_and_scales(tmp_path,routed,block_m):
    import tensor,torch
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    torch.set_num_threads(1)
    rows,k,o,parts,top=32,256,128,2,8;rng=np.random.default_rng(1167)
    shape=(rows,top,k) if routed else (rows,k)
    x=torch.from_numpy(rng.normal(size=shape).astype('float32')).to(torch.float8_e4m3fn)
    a=torch.from_numpy(rng.uniform(.02,.1,(*shape[:-1],k//128)).astype('float32'))
    w=torch.from_numpy(rng.normal(size=(256,o,k)).astype('float32')).to(torch.float8_e4m3fn)
    scales=torch.from_numpy(rng.uniform(.02,.1,(256,1,k//128)).astype('float32')).bfloat16()
    counts=np.zeros(256,'int32');counts[:top]=[1,7,15,16,17,31,32,3]
    routes=np.full((256,rows),-1,'int32')
    want=np.empty((rows,top,o),'float32')
    for rank in range(top):
        selected=rng.permutation(rows)[:counts[rank]]
        routes[rank,:counts[rank]]=selected*top+rank
        xx=x[:,rank] if routed else x;aa=a[:,rank] if routed else a
        xd=xx.float()*aa.repeat_interleave(128,-1)
        wd=w[rank].float()*scales[rank].float().repeat_interleave(128,0).repeat_interleave(128,1)
        want[:,rank]=(xd@wd.T).numpy()
    entry=tmp_path/'expert.py';artifact=entry.with_suffix('.tbin')
    entry.write_text(export_source('tensor_llm.qwen35.kernels.speculative_linear','expert_kernel',
        dict(rows=rows,k=k,o=o,routed_input=routed,columns=128,block_m=block_m,
             threads=256 if block_m>=32 else 128,partitions=parts,stages=2),dependencies=('tensor.compiler.entry',)))
    build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
    with tensor.Device() as dev:
        inputs=[dev.from_numpy(v,dtype=dt) for v,dt in ((x.view(torch.uint8).numpy(),'uint8'),(a.numpy(),'float32'),
            (w.view(torch.uint8).numpy(),'uint8'),(scales.float().numpy(),'bfloat16'),(counts,'int32'),(routes,'int32'))]
        out=dev.empty((rows,top,parts,o))
        dev.load(artifact).launch(*inputs,out)
        got=out.to_numpy().sum(axis=2)
        valid=np.zeros((rows,top),bool)
        for rank in range(top):valid[routes[rank,:counts[rank]]//top,rank]=True
        assert np.isfinite(got[valid]).all()
        assert np.linalg.norm(got[valid]-want[valid])/np.linalg.norm(want[valid])<.001


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native all-row head qualification')
@pytest.mark.parametrize('rows',[32,64])
def test_all_row_head_matches_independent_bf16_projection(tmp_path,rows):
    import torch,tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    torch.set_num_threads(1);rng=np.random.default_rng(8991)
    x=torch.from_numpy(rng.normal(size=(rows,2048)).astype('float32')).bfloat16()
    w=torch.from_numpy(rng.normal(size=(256,2048)).astype('float32')).bfloat16()
    want=(x.float()@w.float().T).numpy()
    entry=tmp_path/'head.py';artifact=entry.with_suffix('.tbin')
    entry.write_text(export_source('tensor_llm.qwen35.kernels.speculative','head_kernel',
        dict(r=rows,k=2048,o=256,block_m=rows,columns=128,depth=128),dependencies=('tensor.compiler.entry',)))
    build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
    with tensor.Device() as dev:
        out=dev.empty(want.shape)
        dev.load(artifact).launch(dev.from_numpy(x.float().numpy(),dtype='bfloat16'),
            dev.from_numpy(w.float().numpy(),dtype='bfloat16'),out)
        got=out.to_numpy();assert np.isfinite(got).all()
        # CPU SGEMM and BF16 MMA accumulate the 2048 products in different
        # FP32 orders. This bound is separate from the unchanged model gate.
        assert np.linalg.norm(got-want)/np.linalg.norm(want)<1e-5
