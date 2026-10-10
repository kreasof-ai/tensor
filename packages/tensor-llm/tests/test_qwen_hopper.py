"""Independent dense attention oracle for bounded Hopper verification tiles."""
import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='Hopper padded attention qualification')
@pytest.mark.parametrize('query_rows,threads,joint,decoded,value_splits',[(128,256,False,False,1),(128,256,True,False,1),(64,128,True,False,1),(128,128,True,False,1),(256,256,True,False,1),(128,256,False,True,1),(128,256,False,True,2)])
def test_hopper_prefill_attention_preserves_control_bits(tmp_path,query_rows,threads,joint,decoded,value_splits):
    import torch,tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    from test_qwen_fp8_kv import quantize
    torch.set_num_threads(1)
    s,c,cap,d=2,32,128,256;rows=s*c;rng=np.random.default_rng(109)
    q=rng.normal(size=(rows,16,d)).astype('float32')
    k=torch.from_numpy(rng.normal(size=(s,2,cap,d)).astype('float32')).bfloat16()
    v=torch.from_numpy(rng.normal(size=k.shape).astype('float32')).bfloat16()
    kb,ks,_=quantize(k);vb,vs,_=quantize(v)
    projection=rng.normal(size=(rows,8192)).astype('float32')
    p=dict(slots=s,chunk=c,capacity=cap,query_rows=128,key_rows=16,packed_loads=True)
    artifacts=[]
    for module,target in [('fp8_kv','sm_90'),('hopper_prefill_attention','sm_90a')]:
        schedule=p if module=='fp8_kv' else dict(p,query_rows=query_rows,threads=threads,joint_kv=joint,decoded_kv=decoded,value_splits=value_splits)
        entry=tmp_path/(module+'.py');artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.'+module,'attention',schedule,
                                       dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,artifact,target=target,compiler='nvrtc',
                       nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
        artifacts.append(artifact)
    if decoded:
        entry=tmp_path/'decode.py';artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.attention_workspace','decode',
                        dict(slots=s,capacity=cap),dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,artifact,target='sm_90a',compiler='nvrtc',
                       nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
    with tensor.Device() as device:
        kernels=[device.load(path) for path in artifacts]
        if decoded:
            decoder=device.load(artifact)
            workspace=[device.empty((s,2,cap,d),'bfloat16') for _ in range(2)]
        shared=[device.from_numpy(q,dtype='bfloat16'),
                *[device.from_numpy(x.numpy()) for x in (kb,vb,ks,vs)],device.from_numpy(projection)]
        for pos,length in [([0,0],[32,17]),([96,64],[31,3]),([96,64],[0,32])]:
            # Device KV beyond the valid prefix is not initialized by the
            # engine. Poison it so a masked future read cannot hide in zeros.
            cache_values=[x.numpy().copy() for x in (kb,vb,ks,vs)]
            for slot in range(s):
                for value in cache_values:
                    value[slot,:,pos[slot]+length[slot]:]=127 if value.dtype==np.uint8 else np.nan
            for index,value in enumerate(cache_values,1):
                shared[index].release();shared[index]=device.from_numpy(value)
            args=[*shared,device.from_numpy(np.array(pos,'int32')),device.from_numpy(np.array(length,'int32'))]
            results=[]
            for index,kernel in enumerate(kernels):
                selected=args
                if decoded and index==1:
                    decoder.launch(*shared[1:5],*args[-2:],*workspace)
                    selected=[shared[0],*workspace,shared[5],*args[-2:]]
                output=device.zeros((rows,4096),'bfloat16')
                kernel.launch(*selected,output);results.append(output.to_numpy());output.release()
            assert np.isfinite(results[1]).all()
            np.testing.assert_array_equal(results[1],results[0])


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='captured profiling event qualification')
def test_captured_profile_has_real_event_timestamps(tmp_path):
    import json
    from pathlib import Path
    from types import SimpleNamespace
    import tensor
    from tensor.providers.cuda_graph import CudaGraph
    from tensor.runtime.abi import BoundCall
    from benchmarks.qwen35.modal_profile import profile_captured_plan
    artifact=tmp_path/'elementwise.tbin'
    entry=tmp_path/'elementwise.py'
    entry.write_text("""import tilelang.language as T
@T.prim_func
def kernel(a:T.Tensor((4096,),'float32'),b:T.Tensor((4096,),'float32'),out:T.Tensor((4096,),'float32')):
    with T.Kernel(16,threads=256) as block:
        for i in T.Parallel(256):out[block*256+i]=2*a[block*256+i]+b[block*256+i]
def tensor_export():return {'kernel':kernel,'outputs':['out']}
""")
    tensor.build(entry,artifact,compiler='nvrtc')
    (tmp_path/'prefill.json').write_text(json.dumps({'kernels':{'elementwise':{'kind':'elementwise'}}}))
    with tensor.Device() as device:
        kernel=device.load(artifact);a=device.arange(4096);b=device.ones((4096,));out=device.zeros((4096,))
        storage,symbols,launch=kernel._bind((a,b,out),{},include_outputs=True)
        bound=BoundCall(device,kernel.manifest,storage,symbols,launch,validated=True)
        with CudaGraph(device,lambda:device._launch(kernel,bound),resources=(kernel,a,b,out)) as graph:
            executor=SimpleNamespace(device=device,kernels={'elementwise':kernel},
                                     plan=[(kernel,bound)],graph=graph)
            result=profile_captured_plan(executor,tmp_path,tmp_path/'profile.json')
            assert 0<result['groups']['elementwise']<=result['captured_plan_milliseconds']
            np.testing.assert_array_equal(out.to_numpy(),2*np.arange(4096)+1)


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native Hopper projection qualification')
@pytest.mark.parametrize('paired',[False,True,'async','pipeline','warp','persistent','packed'])
@pytest.mark.parametrize('columns',[64,128])
@pytest.mark.parametrize('grouped',[False,True])
@pytest.mark.parametrize('parts',[1,2])
def test_wide_hopper_projections_preserve_control_bits(tmp_path,grouped,parts,paired,columns):
    if paired in ('warp','persistent') and not grouped:pytest.skip('expert-only candidate')
    import torch,tensor
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    rng=np.random.default_rng(721);torch.set_num_threads(1)
    rows,k,o=257,2048 if parts in (1,8) else 256,128
    shape=(rows,8,k) if grouped else (rows,k)
    x=torch.from_numpy(rng.normal(size=shape).astype('float32')).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    wshape=(256,o,k) if grouped else (o,k)
    w=torch.from_numpy(rng.normal(size=wshape).astype('float32')).to(torch.float8_e4m3fn).view(torch.uint8).numpy()
    if paired:
        # Exercise every finite FP8 exponent, signs and cancellation. Small
        # normal inputs alone can conceal a changed reduction association.
        x=rng.integers(0,254,size=shape,dtype='uint8');x+=x>=127
        w=rng.integers(0,254,size=wshape,dtype='uint8');w+=w>=127
    scale_shape=(256,1,k//128) if grouped else (1,k//128)
    a=rng.uniform(.001,.01,(*shape[:-1],k//128)).astype('float32')
    scales=rng.uniform(.001,.01,scale_shape).astype('float32')
    p=dict(r=rows,k=k,o=o,block_m=64,threads=128 if columns==64 else 256,columns=columns,partitions=parts)
    if paired=='warp':p.update(block_m=16,threads=128)
    if paired:p.update(mma_reduction=32,mma_reorder=True,async_mma=paired in ('async','pipeline'))
    if paired=='packed':p.update(packed_widen=True)
    if paired=='pipeline':p.update(stages=2,packed_gather=True)
    if grouped:
        p.update(rows=rows,routed_input=True,compact=True,packed_gather=True,bf16_mma=paired!='warp')
        control_module,control_factory,control_args=('speculative_linear','expert_kernel',
            [dict(p,block_m=32,stages=2)])
        if parts==1:
            control_module,control_factory,control_args='prefill','expert_kernel',[dict(p,compact=False,columns=64)]
        selected_module,selected_factory,selected_args='hopper_experts','expert_kernel',[p]
        if paired=='persistent':
            control_module,control_factory,control_args='hopper_experts','expert_kernel',[dict(p)]
            p.update(persistent_tiles=4)
    else:
        control_module,control_factory,control_args=('matmul','make_kernel',
            ['fp8_linear_mma_prequantized',dict(p,columns=64,packed_copy=True,stages=2)])
        selected_module,selected_factory,selected_args='hopper_dense','make_kernel',[p]
    artifacts=[]
    for i,(module,factory,args,target) in enumerate([
        (control_module,control_factory,control_args,'sm_90a' if paired=='persistent' else 'sm_90'),
        (selected_module,selected_factory,selected_args,'sm_90' if paired=='warp' and grouped else 'sm_90a')]):
        entry=tmp_path/(str(i)+'.py');artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.'+module,factory,*args,
            dependencies=('tensor.compiler.entry','tensor_llm.qwen35.kernels.fp8_operand')))
        build_artifact(entry,artifact,target=target,compiler='nvrtc',
                       nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
        artifacts.append(artifact)
    with tensor.Device() as device:
        xx=device.from_numpy(x);ww=device.from_numpy(w)
        ss=device.from_numpy(scales,dtype='bfloat16');aa=device.from_numpy(a)
        output_shape=(rows,8,parts,o) if grouped else (rows,parts,o)
        if parts==1:output_shape=(rows,8,o) if grouped else (rows,o)
        out=device.empty(output_shape);args=[xx,ww,ss,aa]
        selected_tail=[]
        if grouped:
            ids=np.stack([rng.choice(256,8,replace=False) for _ in range(rows)])
            ids[:128]=np.arange(8)
            routes=np.full((256,rows),-1,'int32');counts=np.zeros(256,'int32')
            m=p['block_m']
            experts=np.full((rows*8+m-1)//m+255,-1,'int32');offsets=np.zeros_like(experts);cursor=0
            for expert in range(256):
                row,rank=np.nonzero(ids==expert);values=row*8+rank;rng.shuffle(values)
                routes[expert,:len(values)]=values;counts[expert]=len(values)
                for tile in range((len(values)+m-1)//m):
                    experts[cursor]=expert;offsets[cursor]=tile;cursor+=1
            args=[xx,aa,ww,ss,device.from_numpy(counts),device.from_numpy(routes)]
            selected_tail=[device.from_numpy(experts),device.from_numpy(offsets)]
        device.load(artifacts[0]).launch(*args,out,*(selected_tail if paired=='persistent' else []));expected=out.to_numpy()
        device.load(artifacts[1]).launch(*args,out,*selected_tail);actual=out.to_numpy()
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native Hopper dense Split-K qualification')
@pytest.mark.parametrize('paired',[True,'async','packed'])
@pytest.mark.parametrize('columns',[64,128])
def test_hopper_dense_preserves_eight_part_control_bits(tmp_path,paired,columns):
    test_wide_hopper_projections_preserve_control_bits(tmp_path,False,8,paired,columns)

@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native speculative attention qualification')
@pytest.mark.parametrize('chunk,query_tokens',[(8,8),(16,8),(32,8),(16,16),(32,16)])
@pytest.mark.parametrize('decoded',[False,True])
def test_hopper_query_tiles_preserve_long_window_causality(tmp_path,chunk,query_tokens,decoded):
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
    pos=np.array([96,63,64,0,1,15,25,96],'int32');lengths=np.minimum(c,np.array([c,c,3,c-1,0,2,1,c],'int32'))
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
        entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_attention' if kind=='partial' else 'tensor_llm.qwen35.kernels.speculative_attention',kind,
            dict(slots=s,chunk=c,capacity=cap,splits=splits,packed_loads=True,key_rows=32,query_tokens=query_tokens,
                 decoded_kv=decoded if kind=='partial' else False),dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,out,target='sm_90a' if kind=='partial' else 'sm_90',compiler='nvrtc',nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'));artifacts.append(out)
    if decoded:
        for name,module,factory,schedule in (
            ('decode','attention_workspace','decode',dict(slots=s,capacity=cap)),
            ('control','hopper_attention','partial',dict(slots=s,chunk=c,capacity=cap,
                splits=splits,packed_loads=True,key_rows=32,query_tokens=query_tokens))):
            entry=tmp_path/(name+'.py')
            entry.write_text(export_source('tensor_llm.qwen35.kernels.'+module,factory,schedule,
                dependencies=('tensor.compiler.entry',)))
            build_artifact(entry,entry.with_suffix('.tbin'),target='sm_90a',compiler='nvrtc',
                nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
    with tensor.Device() as dev:
        qq=dev.from_numpy(q.float().numpy(),dtype='bfloat16')
        caches=[dev.from_numpy(x.numpy()) for x in (kb,vb,ks,vs)]
        parts=dev.empty((s,2,splits,c*8,d));stats=dev.empty((s,2,splits,c*8,2))
        positions=dev.from_numpy(pos);length_buffer=dev.from_numpy(lengths)
        if decoded:
            dev.load(tmp_path/'control.tbin').launch(qq,*caches,positions,length_buffer,parts,stats)
            expected_parts=parts.to_numpy();expected_stats=stats.to_numpy()
            scratch=[dev.empty((s,2,cap,d),'bfloat16') for _ in range(2)]
            dev.load(tmp_path/'decode.tbin').launch(*caches,positions,length_buffer,*scratch)
            dev.load(artifacts[0]).launch(qq,*scratch,positions,length_buffer,parts,stats)
            np.testing.assert_array_equal(parts.to_numpy(),expected_parts)
            np.testing.assert_array_equal(stats.to_numpy(),expected_stats)
        else:
            dev.load(artifacts[0]).launch(qq,*caches,positions,length_buffer,parts,stats)
        out=dev.empty(want.shape,'bfloat16')
        dev.load(artifacts[1]).launch(parts,stats,dev.from_numpy(proj),dev.from_numpy(active),out)
        got=out.to_numpy();assert np.isfinite(got).all()
        assert np.linalg.norm(got-want)/np.linalg.norm(want)<.005
        np.testing.assert_allclose(got,want,rtol=.025,atol=.025)
