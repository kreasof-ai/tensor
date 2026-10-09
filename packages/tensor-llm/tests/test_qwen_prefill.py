"""Chunk boundaries, sparse expert packing and prefill state qualification."""
import os
import numpy as np
import pytest
GPU=pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native CUDA prefill qualification')


def build(tmp_path,kind,p):
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    entry=tmp_path/(kind+'.py');artifact=entry.with_suffix('.tbin')
    entry.write_text(export_source('tensor_llm.qwen35.kernels.prefill','make_kernel',kind,p,
        dependencies=('tensor.compiler.entry',)))
    build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
    return artifact


@GPU
def test_gdn_chunk_state_matches_sequential_updates(tmp_path):
    import torch,tensor
    torch.set_num_threads(1)
    slots,chunk,heads,d=8,4,32,128;rows=slots*chunk
    rng=np.random.default_rng(503)
    state=rng.normal(0,.03,(slots,heads,d,d)).astype('float32')
    q=rng.normal(size=(rows,heads,d)).astype('float32');k=rng.normal(size=q.shape).astype('float32')
    q/=np.sqrt((q*q).sum(-1,keepdims=True)+1e-6)*np.sqrt(d)
    k/=np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
    v=rng.normal(size=q.shape).astype('float32')
    g=-rng.uniform(.01,1,(rows,heads)).astype('float32');beta=rng.uniform(0,1,g.shape).astype('float32')
    lengths=np.array([4,3,0,2,1,4,0,4],'int32')
    expected=torch.from_numpy(state.copy());output=torch.zeros(rows,heads,d)
    tq,tk,tv,tg,tb=map(torch.from_numpy,(q,k,v,g,beta))
    for slot in range(slots):
        for step in range(lengths[slot]):
            row=slot*chunk+step
            decayed=expected[slot]*tg[row].exp()[:,None,None]
            delta=tv[row]-(decayed@tk[row].unsqueeze(-1)).squeeze(-1)
            expected[slot]=decayed+torch.einsum('hv,hk->hvk',delta*tb[row,:,None],tk[row])
            output[row]=(expected[slot]@tq[row].unsqueeze(-1)).squeeze(-1)
    artifact=build(tmp_path,'gdn_scan',dict(slots=slots,chunk=chunk))
    with tensor.Device() as dev:
        inputs=[dev.from_numpy(a) for a in (q,k,v,g,beta,lengths,state)]
        out=dev.empty(q.shape)
        dev.load(artifact).launch(*inputs,out)
        np.testing.assert_allclose(inputs[-1].to_numpy(),expected.numpy(),rtol=4e-5,atol=5e-6)
        np.testing.assert_allclose(out.to_numpy(),output.numpy(),rtol=4e-5,atol=5e-6)
        np.testing.assert_array_equal(inputs[-1].to_numpy()[lengths==0],state[lengths==0])


@GPU
def test_expert_packing_has_no_capacity_drops_or_duplicate_rows(tmp_path):
    import tensor
    slots,chunk,top,experts=8,32,8,256;rows=slots*chunk
    rng=np.random.default_rng(215)
    ids=np.stack([rng.choice(experts,top,replace=False) for _ in range(rows)]).astype('int32')
    ids[:64]=np.arange(top) # a hot expert receives more than the average count
    active=rng.integers(0,2,rows,dtype='int32');active[:64]=1
    artifact=build(tmp_path,'expert_routes',dict(slots=slots,chunk=chunk))
    with tensor.Device() as dev:
        ii=dev.from_numpy(ids);aa=dev.from_numpy(active)
        counts=dev.empty((experts,),'int32');routes=dev.empty((experts,rows),'int32')
        dev.load(artifact).launch(ii,aa,counts,routes)
        got_counts,got_routes=counts.to_numpy(),routes.to_numpy()
        assert got_counts.sum()==int(active.sum())*top
        for expert in range(experts):
            expected=sorted(row*top+rank for row in range(rows) if active[row]
                            for rank in range(top) if ids[row,rank]==expert)
            assert sorted(got_routes[expert,:got_counts[expert]].tolist())==expected


@GPU
@pytest.mark.parametrize('kv_dtype',['bfloat16','fp8','fp8-packed'])
def test_causal_chunk_attention_matches_independent_dense_attention(tmp_path,kv_dtype):
    import torch,tensor
    torch.set_num_threads(1)
    slots,chunk,cap,d=8,16,64,256;rows=slots*chunk
    rng=np.random.default_rng(702)
    q=torch.from_numpy(rng.normal(size=(rows,16,d)).astype('float32')).bfloat16()
    k=torch.from_numpy(rng.normal(size=(slots,2,cap,d)).astype('float32')).bfloat16()
    v=torch.from_numpy(rng.normal(size=k.shape).astype('float32')).bfloat16()
    if kv_dtype.startswith('fp8'):
        from test_qwen_fp8_kv import quantize
        kb,ks,k=quantize(k);vb,vs,v=quantize(v)
    projection=torch.from_numpy(rng.normal(size=(rows,8192)).astype('float32'))
    positions=np.array([0,7,13,0,4,29,1,21],'int32')
    lengths=np.array([16,13,0,7,1,16,9,16],'int32')
    expected=np.zeros((rows,4096),'float32')
    valid=[]
    for slot in range(slots):
        for step in range(lengths[slot]):
            row=slot*chunk+step;count=int(positions[slot])+step+1
            kk=k[slot,:,:count].float().repeat_interleave(8,0)
            vv=v[slot,:,:count].float().repeat_interleave(8,0)
            prob=(torch.einsum('hd,htd->ht',q[row].float(),kk)/16).softmax(-1).bfloat16().float()
            value=torch.einsum('ht,htd->hd',prob,vv).bfloat16().float()
            gate=projection[row].bfloat16().float().reshape(16,512)[:,256:]
            expected[row]=(value*gate.sigmoid()).bfloat16().float().reshape(4096).numpy();valid.append(row)
    p=dict(slots=slots,chunk=chunk,capacity=cap)
    if kv_dtype.startswith('fp8'):p['kv_dtype']='fp8'
    if kv_dtype=='fp8-packed':p.update(packed_loads=True,query_rows=128,key_rows=16)
    artifact=build(tmp_path,'attention',p)
    with tensor.Device() as dev:
        inputs=[dev.from_numpy(a,dtype=dt) for a,dt in ((q.float().numpy(),'bfloat16'),
            (k.float().numpy(),'bfloat16'),(v.float().numpy(),'bfloat16'),(projection.numpy(),'float32'),
            (positions,'int32'),(lengths,'int32'))]
        if kv_dtype.startswith('fp8'):
            for buffer in inputs[1:3]:buffer.release()
            inputs[1:3]=[dev.from_numpy(x.numpy()) for x in (kb,vb,ks,vs)]
        out=dev.from_numpy(np.full((rows,4096),np.nan,'float32'),dtype='bfloat16')
        dev.load(artifact).launch(*inputs,out)
        got=out.to_numpy()[valid];want=expected[valid]
        # Online softmax rounds unnormalized probabilities to BF16 in each
        # tile, whereas the dense oracle rounds after normalization.
        assert np.linalg.norm(got-want)/np.linalg.norm(want)<.005
        np.testing.assert_allclose(got,want,rtol=.02,atol=.02)


@GPU
def test_decode_attention_merge_rounds_before_gate(tmp_path):
    import tensor,torch
    from tensor_llm.qwen35.kernels.decode import source
    from tensor.compiler.build import build_artifact
    slots,splits=8,16
    rng=np.random.default_rng(427)
    partial=np.zeros((slots,2,splits,8,256),'float32')
    partial[:,:,0]=rng.normal(size=(slots,2,8,256)).astype('float32')
    stats=np.zeros((slots,2,splits,8,2),'float32');stats[...,0]=-np.inf
    stats[:,:,0,:,0]=0;stats[:,:,0,:,1]=1
    projection=rng.normal(size=(slots,8192)).astype('float32');active=np.ones(slots,'int32')
    value=torch.from_numpy(partial[:,:,0].reshape(slots,16,256)).bfloat16().float()
    gate=torch.from_numpy(projection).bfloat16().float().reshape(slots,16,512)[:,:,256:]
    expected=(value*gate.sigmoid()).bfloat16().float().reshape(slots,4096).numpy()
    entry=tmp_path/'merge.py';artifact=entry.with_suffix('.tbin')
    entry.write_text(source('attention_merge',dict(r=slots,splits=splits)))
    build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
    with tensor.Device() as dev:
        inputs=[dev.from_numpy(a) for a in (partial,stats,projection,active)]
        out=dev.empty(expected.shape,'bfloat16');dev.load(artifact).launch(*inputs,out)
        np.testing.assert_allclose(out.to_numpy(),expected,rtol=0,atol=.015625)
        assert np.linalg.norm(out.to_numpy()-expected)/np.linalg.norm(expected)<.001


@GPU
@pytest.mark.parametrize('routed',[False,True])
def test_large_expert_tiles_preserve_hot_experts_and_tail_rows(tmp_path,routed):
    import torch,tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    torch.set_num_threads(1)
    rows,k,o,experts,top=65,256,128,256,8;rng=np.random.default_rng(1124)
    shape=(rows,top,k) if routed else (rows,k)
    x=torch.from_numpy(rng.normal(size=shape).astype('float32')).to(torch.float8_e4m3fn)
    a=torch.from_numpy(rng.uniform(.02,.1,(*shape[:-1],k//128)).astype('float32'))
    w=torch.from_numpy(rng.normal(size=(experts,o,k)).astype('float32')).to(torch.float8_e4m3fn)
    scales=torch.from_numpy(rng.uniform(.02,.1,(experts,1,k//128)).astype('float32')).bfloat16()
    counts=np.zeros(experts,'int32');counts[:top]=rows
    routes=np.full((experts,rows),-1,'int32')
    for rank in range(top):routes[rank]=np.arange(rows)*top+rank
    expected=np.empty((rows,top,o),'float32')
    for rank in range(top):
        xx=x[:,rank] if routed else x;aa=a[:,rank] if routed else a
        xd=xx.float()*aa.repeat_interleave(128,-1)
        wd=w[rank].float()*scales[rank].float().repeat_interleave(128,0).repeat_interleave(128,1)
        expected[:,rank]=(xd@wd.T).numpy()
    entry=tmp_path/'expert-m64.py';artifact=entry.with_suffix('.tbin')
    entry.write_text(export_source('tensor_llm.qwen35.kernels.prefill','expert_kernel',
        dict(rows=rows,k=k,o=o,routed_input=routed,block_m=64,threads=256),dependencies=('tensor.compiler.entry',)))
    build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
    with tensor.Device() as dev:
        inputs=[dev.from_numpy(v,dtype=dt) for v,dt in ((x.view(torch.uint8).numpy(),'uint8'),(a.numpy(),'float32'),
            (w.view(torch.uint8).numpy(),'uint8'),(scales.float().numpy(),'bfloat16'),(counts,'int32'),(routes,'int32'))]
        out=dev.from_numpy(np.full((rows,top,o),np.nan,'float32'))
        dev.load(artifact).launch(*inputs,out);actual=out.to_numpy()
        assert np.isfinite(actual).all()
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<.001
