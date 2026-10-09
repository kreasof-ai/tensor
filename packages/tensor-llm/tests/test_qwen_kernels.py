"""Numerical qualification of native FP8 bits and independent recurrent slots."""
import os

import numpy as np
import pytest

GPU=pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',
    reason='set TENSOR_QWEN_CUDA=1 for native FP8/gated delta qualification')


def build(tmp_path,kind,p):
    from tensor.compiler.build import build_artifact
    from tensor_llm.qwen35.kernels.decode import source
    entry=tmp_path/(kind+'.py');entry.write_text(source(kind,p))
    artifact=tmp_path/(kind+'.tbin')
    build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',
        nvrtc_home='build/nvrtc-12.9')
    return artifact


def build_mma(tmp_path,factory,*args):
    from tensor.compiler.build import build_artifact
    from tensor.compiler.entry import export_source
    entry=tmp_path/(factory+str(len(list(tmp_path.glob('*.py'))))+'.py')
    entry.write_text(export_source('tensor_llm.qwen35.kernels.matmul',factory,*args,
        dependencies=('tensor.compiler.entry','tensor.compiler.cuda_lowering')))
    artifact=entry.with_suffix('.tbin')
    build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
    return artifact


@GPU
@pytest.mark.parametrize('top',[None,4])
def test_prequantization_bits_and_scales(tmp_path,top):
    import tensor
    import torch
    torch.set_num_threads(1)
    r,k=8,512
    shape=(r,top,k) if top else (r,k)
    rng=np.random.default_rng(836)
    x=torch.from_numpy(rng.normal(size=shape).astype('float32')).bfloat16().float()
    x[...,0:128]=0
    blocks=x.reshape(*shape[:-1],k//128,128)
    scales=blocks.abs().amax(-1,keepdim=True).clamp_min(1e-12)/448
    bits=(blocks/scales).to(torch.float8_e4m3fn).view(torch.uint8).reshape(shape).numpy()
    p=dict(r=r,k=k)
    if top:p['top']=top
    artifact=build_mma(tmp_path,'quantize_kernel',p)
    with tensor.Device() as d:
        xx=d.from_numpy(x.numpy(),dtype='bfloat16');out=d.empty(shape,'uint8')
        ss=d.empty(tuple(scales.shape[:-1]),'float32')
        d.load(artifact).launch(xx,out,ss)
        np.testing.assert_array_equal(out.to_numpy(),bits)
        np.testing.assert_allclose(ss.to_numpy(),scales.squeeze(-1).numpy(),rtol=1e-7,atol=0)


@GPU
@pytest.mark.parametrize('packed_copy',[False,True])
@pytest.mark.parametrize('block_m',[16,64])
def test_prequantized_split_mma_matches_unsplit_mma(tmp_path,packed_copy,block_m):
    import tensor
    import torch
    torch.set_num_threads(1)
    r,k,o=(8 if block_m==16 else 65),512,256
    rng=np.random.default_rng(621)
    x=torch.from_numpy(rng.normal(size=(r,k)).astype('float32')).bfloat16().float()
    w=torch.from_numpy(rng.normal(size=(o,k)).astype('float32')).to(torch.float8_e4m3fn)
    scales=torch.from_numpy(rng.uniform(.02,.1,size=(o//128,k//128)).astype('float32')).bfloat16().float()
    qp=build_mma(tmp_path,'quantize_kernel',dict(r=r,k=k))
    artifacts=[build_mma(tmp_path,'make_kernel','fp8_linear_mma_prequantized',
        dict(r=r,k=k,o=o,columns=64,partitions=part,packed_copy=packed_copy if part==4 else False,
             block_m=16 if part==1 else block_m,threads=128 if part==1 else (256 if block_m==64 else 128))) for part in (1,4)]
    merge=build_mma(tmp_path,'merge_kernel',dict(r=r,o=o,partitions=4))
    with tensor.Device() as d:
        xx=d.from_numpy(x.numpy(),dtype='bfloat16');bits=d.empty((r,k),'uint8')
        ascales=d.empty((r,k//128),'float32')
        ww=d.from_numpy(w.view(torch.uint8).numpy());ss=d.from_numpy(scales.numpy(),dtype='bfloat16')
        direct=d.empty((r,o));partial=d.empty((r,4,o));merged=d.empty((r,o))
        d.load(qp).launch(xx,bits,ascales)
        d.load(artifacts[0]).launch(bits,ww,ss,ascales,direct)
        d.load(artifacts[1]).launch(bits,ww,ss,ascales,partial)
        d.load(merge).launch(partial,merged)
        # Both paths use the same FP8 tensor-core dot products. Only their
        # FP32 sum across independently scaled K blocks is reassociated.
        np.testing.assert_allclose(merged.to_numpy(),direct.to_numpy(),rtol=3e-5,atol=3e-5)


@GPU
def test_split_merge_covers_all_request_expert_rows(tmp_path):
    import tensor
    r,top,parts,o=8,8,4,512
    values=np.random.default_rng(773).normal(size=(r,top,parts,o)).astype('float32')
    artifact=build_mma(tmp_path,'merge_kernel',dict(r=r,top=top,o=o,partitions=parts))
    expected=np.zeros((r,top,o),'float32')
    for part in range(parts):expected+=values[:,:,part]
    with tensor.Device() as d:
        partial=d.from_numpy(values)
        out=d.from_numpy(np.full(expected.shape,np.nan,'float32'))
        d.load(artifact).launch(partial,out)
        np.testing.assert_allclose(out.to_numpy(),expected,rtol=0,atol=0)


@GPU
def test_every_e4m3_storage_bit_against_torch(tmp_path):
    import tensor
    import torch
    bits=np.arange(256,dtype='uint8')
    expected=torch.from_numpy(bits).view(torch.float8_e4m3fn).float().numpy()
    artifact=build(tmp_path,'e4m3_decode',{})
    with tensor.Device() as d:
        x=d.from_numpy(bits);y=d.empty((256,),'float32')
        d.load(artifact).launch(x,y)
        np.testing.assert_array_equal(y.to_numpy(),expected)
        assert np.signbit(y.to_numpy()[128]) # preserve negative zero


@GPU
@pytest.mark.parametrize('r',[1,8])
def test_native_fp8_block_linear_against_independent_quantized_matmul(tmp_path,r):
    import tensor
    import torch
    from tensor.runtime.dtypes import decode_bfloat16,encode_bfloat16
    torch.set_num_threads(1)
    rng=np.random.default_rng(1841+r)
    k,o=256,128
    x=decode_bfloat16(encode_bfloat16(rng.normal(size=(r,k)).astype('float32')))
    # Include zero blocks and uneven block scales to expose row/column swapping.
    x[0,:128]=0
    w=torch.from_numpy(rng.normal(size=(o,k)).astype('float32')).to(torch.float8_e4m3fn)
    scales=decode_bfloat16(encode_bfloat16(np.array([[.03125,.3125]],'float32')))
    a=torch.from_numpy(x).reshape(r,k//128,128)
    a_scale=a.abs().amax(-1,keepdim=True).clamp_min(1e-12)/448
    a=(a/a_scale).to(torch.float8_e4m3fn).float()*a_scale
    expected=a.reshape(r,k)@(w.float()*torch.from_numpy(scales).repeat_interleave(128,0).repeat_interleave(128,1)).T
    artifact=build(tmp_path,'fp8_linear',dict(r=r,k=k,o=o))
    with tensor.Device() as d:
        xx=d.from_numpy(x,dtype='bfloat16');ww=d.from_numpy(w.view(torch.uint8).numpy())
        ss=d.from_numpy(scales,dtype='bfloat16');out=d.empty((r,o),'float32')
        d.load(artifact).launch(xx,ww,ss,out)
        np.testing.assert_allclose(out.to_numpy(),expected.numpy(),rtol=3e-5,atol=3e-5)


@GPU
def test_gated_delta_c8_state_isolation_and_continuation(tmp_path):
    import tensor
    import torch
    torch.set_num_threads(1)
    slots,heads,key,value=8,32,128,128
    p=dict(slots=slots,heads=heads,key=key,value=value)
    artifact=build(tmp_path,'gdn_recurrent',p)
    rng=np.random.default_rng(83)
    state=rng.normal(0,.02,(slots,heads,value,key)).astype('float32')
    expected=torch.from_numpy(state.copy())
    active=np.array([1,1,0,1,0,1,1,1],dtype='int32')
    with tensor.Device() as d:
        s=d.from_numpy(state);aa=d.from_numpy(active);out=d.empty((slots,heads,value))
        kernel=d.load(artifact)
        for step in range(3):
            q=rng.normal(size=(slots,heads,key)).astype('float32')
            k=rng.normal(size=(slots,heads,key)).astype('float32')
            q/=np.sqrt((q*q).sum(-1,keepdims=True)+1e-6)*np.sqrt(key)
            k/=np.sqrt((k*k).sum(-1,keepdims=True)+1e-6)
            v=rng.normal(size=(slots,heads,value)).astype('float32')
            g=-rng.uniform(.01,1,(slots,heads)).astype('float32')
            beta=rng.uniform(0,1,(slots,heads)).astype('float32')
            tq,tk,tv,tg,tb=map(torch.from_numpy,(q,k,v,g,beta))
            # Reference uses matrix products and outer products rather than
            # the kernel's reductions or value partitioning.
            decayed=expected*tg.exp()[...,None,None]
            delta=tv-(decayed@tk.unsqueeze(-1)).squeeze(-1)
            updated=decayed+torch.einsum('bhv,bhk->bhvk',delta*tb.unsqueeze(-1),tk)
            mask=torch.from_numpy(active.astype(bool))[:,None,None,None]
            expected=torch.where(mask,updated,expected)
            want=(expected@tq.unsqueeze(-1)).squeeze(-1)
            want[torch.from_numpy(active==0)]=0
            inputs=[d.from_numpy(a) for a in (q,k,v,g,beta)]
            try:
                kernel.launch(*inputs,aa,s,out)
                np.testing.assert_allclose(out.to_numpy(),want.numpy(),rtol=3e-5,atol=3e-6)
                np.testing.assert_allclose(s.to_numpy(),expected.numpy(),rtol=3e-5,atol=3e-6)
                np.testing.assert_array_equal(s.to_numpy()[active==0],state[active==0])
            finally:
                for b in inputs:b.release()


@GPU
@pytest.mark.parametrize('routed',[False,True])
def test_grouped_experts_reuse_weights_without_mixing_routes(tmp_path,routed):
    import tensor
    import torch
    torch.set_num_threads(1)
    r,top,e,k,o=8,4,16,256,128
    rng=np.random.default_rng(271)
    ids=np.stack([rng.choice(e,top,replace=False) for _ in range(r)]).astype('int32')
    ids[1]=ids[0] # exact overlap, alongside partially shared and distinct routes
    shape=(r,top,k) if routed else (r,k)
    x=torch.from_numpy(rng.normal(size=shape).astype('float32')).bfloat16()
    w=torch.from_numpy(rng.normal(size=(e,o,k)).astype('float32')).to(torch.float8_e4m3fn)
    scales=torch.from_numpy(rng.uniform(.02,.1,size=(e,1,2)).astype('float32')).bfloat16()
    expected=np.empty((r,top,o),dtype='float32')
    for b in range(r):
        for rank in range(top):
            expert=ids[b,rank]
            xx=x[b,rank] if routed else x[b]
            blocks=xx.float().reshape(2,128)
            scale=blocks.abs().amax(-1,keepdim=True).clamp_min(1e-12)/448
            xx=((blocks/scale).to(torch.float8_e4m3fn).float()*scale).reshape(k)
            ww=w[expert].float()*scales[expert].float().repeat_interleave(128,0).repeat_interleave(128,1)
            expected[b,rank]=(xx@ww.T).numpy()
    group=build(tmp_path,'moe_groups',dict(r=r,top=top))
    expert_kernel=build(tmp_path,'fp8_experts',dict(r=r,top=top,experts=e,k=k,o=o,routed_input=routed))
    with tensor.Device() as d:
        ii=d.from_numpy(ids);ee=d.empty((r*top,),'int32');rr=d.empty((r*top,r),'int32')
        d.load(group).launch(ii,ee,rr)
        actual_experts=ee.to_numpy();actual_routes=rr.to_numpy()
        assert set(actual_experts[actual_experts>=0])==set(ids.ravel())
        assert len(actual_experts[actual_experts>=0])==len(set(ids.ravel()))
        for idx,expert in enumerate(actual_experts):
            for b in range(r):
                rank=actual_routes[idx,b]
                if rank>=0:assert ids[b,rank]==expert
        xx=d.from_numpy(x.float().numpy(),dtype='bfloat16')
        ww=d.from_numpy(w.view(torch.uint8).numpy());ss=d.from_numpy(scales.float().numpy(),dtype='bfloat16')
        out=d.from_numpy(np.full((r,top,o),np.nan,'float32'))
        d.load(expert_kernel).launch(xx,ww,ss,ee,rr,out)
        np.testing.assert_allclose(out.to_numpy(),expected,rtol=4e-5,atol=5e-5)


@GPU
@pytest.mark.parametrize('o',[1,32,256])
def test_small_bf16_decode_parallel_reduction(tmp_path,o):
    import torch,tensor
    torch.set_num_threads(1)
    rng=np.random.default_rng(814)
    r,k=8,2048
    x=torch.from_numpy(rng.normal(size=(r,k)).astype('float32')).bfloat16().float()
    w=torch.from_numpy(rng.normal(size=(o,k)).astype('float32')).bfloat16().float()
    expected=(x@w.T).numpy()
    artifact=build_mma(tmp_path,'bf16_decode_kernel',dict(r=r,k=k,o=o))
    with tensor.Device() as dev:
        xx=dev.from_numpy(x.numpy(),dtype='bfloat16');ww=dev.from_numpy(w.numpy(),dtype='bfloat16')
        out=dev.empty((r,o));dev.load(artifact).launch(xx,ww,out)
        np.testing.assert_allclose(out.to_numpy(),expected,rtol=3e-5,atol=1e-4)


@GPU
def test_bf16_vocabulary_mma_matches_independent_cpu_projection(tmp_path):
    import torch,tensor
    torch.set_num_threads(1);rng=np.random.default_rng(973)
    r,k,o=8,2048,1024
    x=torch.from_numpy(rng.normal(size=(r,k)).astype('float32')).bfloat16().float()
    w=torch.from_numpy(rng.normal(size=(o,k)).astype('float32')).bfloat16().float()
    artifact=build_mma(tmp_path,'bf16_head_kernel',dict(r=r,k=k,o=o,columns=64,depth=128))
    with tensor.Device() as dev:
        xx=dev.from_numpy(x.numpy(),dtype='bfloat16');ww=dev.from_numpy(w.numpy(),dtype='bfloat16');out=dev.empty((r,o))
        dev.load(artifact).launch(xx,ww,out)
        expected=(x.double()@w.double().T).float().numpy();got=out.to_numpy()
        assert np.linalg.norm(got-expected)/np.linalg.norm(expected)<1e-5
        np.testing.assert_allclose(got,expected,rtol=4e-5,atol=1e-3)
