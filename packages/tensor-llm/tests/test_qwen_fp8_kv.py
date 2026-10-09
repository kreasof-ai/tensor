"""Independent FP8 KV quantization, persistence and attention quality gates."""
import os
import numpy as np
import pytest
GPU=pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='native FP8 KV qualification')


def build(tmp_path,kind,p):
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    entry=tmp_path/(kind+'-'+str(p.get('kv_dtype','bfloat16'))+'.py')
    entry.write_text(export_source('tensor_llm.qwen35.kernels.decode','make_kernel',kind,p,
        dependencies=('tensor.compiler.entry','tensor.compiler.cuda_lowering','tensor_llm.qwen35.kernels.fp8_kv')))
    artifact=entry.with_suffix('.tbin')
    build_artifact(entry,artifact,target=os.environ.get('TENSOR_QWEN_TARGET','sm_89'),compiler='nvrtc',nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
    return artifact


def quantize(x):
    import torch
    values=x.float().reshape(*x.shape[:-1],2,128)
    scales=values.abs().amax(-1).clamp_min(1e-12)/448
    bits=(values/scales[...,None]).to(torch.float8_e4m3fn).view(torch.uint8).reshape(x.shape)
    decoded=(bits.view(torch.float8_e4m3fn).float().reshape(values.shape)*scales[...,None]).reshape(x.shape).bfloat16()
    return bits,scales,decoded


@GPU
def test_cache_write_matches_independent_quantization_and_preserves_other_slots(tmp_path):
    import tensor,torch
    torch.set_num_threads(1)
    r,cap=8,64;rng=np.random.default_rng(823)
    projection=rng.normal(size=(r,512)).astype('float32');values=rng.normal(size=projection.shape).astype('float32')
    w=rng.normal(0,.02,256).astype('float32')
    positions=np.array([0,7,31,3,63,12,0,23],'int32');active=np.array([1,0,1,1,0,1,1,0],'int32')
    p=dict(r=r,eps=1e-6,capacity=cap,theta=1e7)
    baseline=build(tmp_path,'attention_kv',p);fp8=build(tmp_path,'attention_kv',dict(p,kv_dtype='fp8'))
    with tensor.Device() as dev:
        inputs=[dev.from_numpy(a,dtype=dt) for a,dt in ((projection,'float32'),(values,'float32'),(w,'bfloat16'),(positions,'int32'),(active,'int32'))]
        bf=[dev.from_numpy(np.zeros((r,2,cap,256),'float32'),dtype='bfloat16') for _ in range(2)]
        bits=[dev.from_numpy(np.full((r,2,cap,256),123,'uint8')) for _ in range(2)]
        scales=[dev.from_numpy(np.full((r,2,cap,2),-999,'float32')) for _ in range(2)]
        dev.load(baseline).launch(*inputs,*bf);dev.load(fp8).launch(*inputs,*bits,*scales)
        for ref,encoded,scale in zip(bf,bits,scales):
            x=torch.from_numpy(ref.to_numpy()).bfloat16()
            qb,qs,qd=quantize(x)
            actual,ascale=encoded.to_numpy(),scale.to_numpy()
            mask=np.zeros((r,cap),bool);mask[np.flatnonzero(active),positions[active!=0]]=True
            for slot,pos in zip(np.flatnonzero(active),positions[active!=0]):
                np.testing.assert_array_equal(actual[slot,:,pos],qb.numpy()[slot,:,pos])
                np.testing.assert_allclose(ascale[slot,:,pos],qs.numpy()[slot,:,pos],rtol=1e-6)
                e=(qd[slot,:,pos].float()-x[slot,:,pos].float()).square().mean().sqrt()/x[slot,:,pos].float().square().mean().sqrt()
                assert float(e)<.04
            for slot in range(r):
                np.testing.assert_array_equal(actual[slot,:,~mask[slot]],123)
                np.testing.assert_array_equal(ascale[slot,:,~mask[slot]],-999)


@GPU
@pytest.mark.parametrize('packed_loads',[False,True])
@pytest.mark.parametrize('key_rows',[64,32,16])
def test_fp8_decode_attention_matches_dequantized_cpu_oracle(tmp_path,packed_loads,key_rows):
    import tensor,torch
    torch.set_num_threads(1)
    r,cap,splits,d=8,128,4,256;rng=np.random.default_rng(991)
    q=torch.from_numpy(rng.normal(size=(r,16,d)).astype('float32')).bfloat16()
    k=torch.from_numpy(rng.normal(size=(r,2,cap,d)).astype('float32')).bfloat16()
    v=torch.from_numpy(rng.normal(size=k.shape).astype('float32')).bfloat16()
    kb,ks,kd=quantize(k);vb,vs,vd=quantize(v)
    positions=np.array([0,1,17,63,64,126,127,22],'int32');active=np.array([1,1,1,1,1,1,1,0],'int32')
    projection=rng.normal(size=(r,8192)).astype('float32')
    partial=build(tmp_path,'attention_partial',dict(r=r,capacity=cap,splits=splits,kv_dtype='fp8',packed_loads=packed_loads,key_rows=key_rows))
    merge=build(tmp_path,'attention_merge',dict(r=r,splits=splits))
    expected=np.zeros((r,4096),'float32');baseline=expected.copy()
    for slot in np.flatnonzero(active):
        count=int(positions[slot])+1
        for kk,vv,dest in ((kd,vd,expected),(k,v,baseline)):
            key=kk[slot,:,:count].float().repeat_interleave(8,0);value=vv[slot,:,:count].float().repeat_interleave(8,0)
            prob=(torch.einsum('hd,htd->ht',q[slot].float(),key)/16).softmax(-1).bfloat16().float()
            attended=torch.einsum('ht,htd->hd',prob,value).bfloat16().float()
            gate=torch.from_numpy(projection[slot]).bfloat16().float().reshape(16,512)[:,256:]
            dest[slot]=(attended*gate.sigmoid()).bfloat16().float().reshape(-1).numpy()
    with tensor.Device() as dev:
        args=[dev.from_numpy(x,dtype=dt) for x,dt in ((q.float().numpy(),'bfloat16'),(kb.numpy(),'uint8'),(vb.numpy(),'uint8'),(ks.numpy(),'float32'),(vs.numpy(),'float32'),(positions,'int32'),(active,'int32'))]
        parts=dev.empty((r,2,splits,8,d));stats=dev.empty((r,2,splits,8,2))
        out=dev.empty((r,4096),'bfloat16');proj=dev.from_numpy(projection)
        dev.load(partial).launch(*args,parts,stats);dev.load(merge).launch(parts,stats,proj,args[-1],out)
        got=out.to_numpy();valid=active!=0
        rms=np.sqrt(np.mean((got[valid]-expected[valid])**2))/np.sqrt(np.mean(expected[valid]**2))
        assert rms<.005
        quant_error=np.sqrt(np.mean((got[valid]-baseline[valid])**2))/np.sqrt(np.mean(baseline[valid]**2))
        assert quant_error<.06
        np.testing.assert_array_equal(got[~valid],0)
