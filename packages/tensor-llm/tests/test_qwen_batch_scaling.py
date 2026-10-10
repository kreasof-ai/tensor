"""Larger expert batches must preserve each independent 16-row result."""
import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='physical batch scaling qualification')
@pytest.mark.parametrize('rows',[17,32,63,64])
@pytest.mark.parametrize('routed',[False,True])
def test_expert_row_tiles_preserve_small_batch_bits(tmp_path,rows,routed):
    import tensor
    from test_qwen_kernels import build,build_mma
    rng=np.random.default_rng(70921);top,k,o,e,parts=8,512,128,8,4
    ids=np.tile(np.arange(top,dtype='int32'),(rows,1))
    for row in range(rows):ids[row]=rng.permutation(top)
    shape=(rows,top,k) if routed else (rows,k)
    # Every finite E4M3 encoding, including both signed zeros.
    bits=rng.integers(0,256,shape,dtype='uint8');bits[bits==127]=126;bits[bits==255]=254
    weight=rng.integers(0,256,(e,o,k),dtype='uint8');weight[weight==127]=126;weight[weight==255]=254
    scales=rng.uniform(.001,.1,(e,1,k//128)).astype('float32')
    activation=rng.uniform(.001,.1,(*shape[:-1],k//128)).astype('float32')
    def produce(count):
        directory=tmp_path/str(count);directory.mkdir()
        p=dict(r=count,top=top,experts=e,k=k,o=o,routed_input=routed,
               columns=128,threads=128,block_m=16,partitions=parts,stages=2,packed_copy=True)
        return (build(directory,'moe_groups',dict(r=count,top=top)),
                build_mma(directory,'make_kernel','fp8_experts_mma_prequantized',p))
    large=produce(rows);small=produce(16);tail=produce(rows%16) if rows%16 else small
    with tensor.Device() as dev:
        ww=dev.from_numpy(weight);ss=dev.from_numpy(scales,dtype='bfloat16')
        def evaluate(start,count,artifacts):
            ii=dev.from_numpy(ids[start:start+count]);ee=dev.empty((count*top,),'int32')
            rr=dev.empty((count*top,count),'int32');xx=dev.from_numpy(bits[start:start+count])
            aa=dev.from_numpy(activation[start:start+count]);out=dev.from_numpy(np.full((count,top,parts,o),np.nan,'float32'))
            group,kernel=(dev.load(p) for p in artifacts)
            group.launch(ii,ee,rr);kernel.launch(xx,ww,ss,aa,ee,rr,out)
            value=out.to_numpy()
            for b in (ii,ee,rr,xx,aa,out):b.release()
            group.release();kernel.release();return value
        actual=evaluate(0,rows,large)
        expected=np.concatenate([evaluate(start,min(16,rows-start),small if rows-start>=16 else tail)
                                 for start in range(0,rows,16)])
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='physical batch scaling qualification')
@pytest.mark.parametrize('rows',[17,32,64])
def test_large_logical_dense_projection_preserves_independent_rows(tmp_path,rows):
    import tensor
    from test_qwen_kernels import build_mma
    rng=np.random.default_rng(80391);k,o=256,128
    x=rng.normal(size=(rows,k)).astype('float32')
    weight=rng.integers(0,256,(o,k),dtype='uint8')
    weight[weight==127]=126;weight[weight==255]=254
    scales=rng.uniform(.001,.1,(1,k//128)).astype('float32')
    def produce(count):
        directory=tmp_path/str(count);directory.mkdir()
        return build_mma(directory,'make_kernel','fp8_linear_mma',
                         dict(r=count,k=k,o=o,block_m=16,columns=64,threads=128))
    large=produce(rows);small=produce(8)
    tail=produce(rows%8) if rows%8 else small
    with tensor.Device() as device:
        ww=device.from_numpy(weight);ss=device.from_numpy(scales,dtype='bfloat16')
        def evaluate(start,count,path):
            xx=device.from_numpy(x[start:start+count],dtype='bfloat16')
            out=device.empty((count,o));kernel=device.load(path)
            try:
                kernel.launch(xx,ww,ss,out)
                return out.to_numpy()
            finally:
                kernel.release();xx.release();out.release()
        actual=evaluate(0,rows,large)
        expected=np.concatenate([evaluate(start,min(8,rows-start),small if rows-start>=8 else tail)
                                 for start in range(0,rows,8)])
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual,expected)


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA')!='1',reason='physical pooled attention qualification')
def test_padded_short_attention_preserves_parent_accumulation(tmp_path):
    import tensor,torch
    from tensor.compiler.entry import export_source
    from tensor.compiler.build import build_artifact
    from test_qwen_fp8_kv import quantize
    torch.set_num_threads(1)
    slots,cap,splits=8,4096,16;rng=np.random.default_rng(94017)
    queries=rng.normal(size=(slots,128,16,256)).astype('float32')
    lengths=np.array([4,3,1,0,4,2,3,1],'int32')
    positions=np.array([4092,2047,2048,0,1,15,25,3071],'int32')
    caches=[]
    for _ in range(2):
        x=torch.from_numpy(rng.normal(size=(slots,2,cap,256)).astype('float32')).bfloat16()
        caches.append(quantize(x))
    def compile_kernel(chunk,decoded):
        entry=tmp_path/f'attention-{chunk}.py';artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_attention','partial',
            dict(slots=slots,chunk=chunk,capacity=cap,splits=splits,key_rows=32,
                 query_tokens=8,pad_queries=True,packed_loads=True,decoded_kv=decoded),
            dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,artifact,target='sm_90a',compiler='nvrtc',
                      nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
        return artifact
    parent=compile_kernel(128,True);short=compile_kernel(4,False)
    with tensor.Device() as dev:
        pp=dev.from_numpy(positions);ll=dev.from_numpy(lengths)
        def evaluate(chunk,decoded,artifact):
            qq=dev.from_numpy(queries[:,:chunk].reshape(slots*chunk,16,256),dtype='bfloat16')
            if decoded:
                cache_args=[dev.from_numpy(cache[2].float().numpy(),dtype='bfloat16') for cache in caches]
            else:
                cache_args=[dev.from_numpy(caches[0][0].numpy()),dev.from_numpy(caches[1][0].numpy()),
                            dev.from_numpy(caches[0][1].numpy()),dev.from_numpy(caches[1][1].numpy())]
            out=dev.empty((slots,2,splits,chunk*8,256));stats=dev.empty((slots,2,splits,chunk*8,2))
            kernel=dev.load(artifact)
            try:
                kernel.launch(qq,*cache_args,pp,ll,out,stats)
                # Retain every active query/head, including partition boundaries.
                values=[out.to_numpy(),stats.to_numpy()]
                return [np.concatenate([value[s,:,:,:int(lengths[s])*8]
                                        for s in range(slots)],axis=2) for value in values]
            finally:
                kernel.release()
                for buffer in (qq,*cache_args,out,stats):buffer.release()
        expected=evaluate(128,True,parent);actual=evaluate(4,False,short)
        for control,candidate in zip(expected,actual):
            np.testing.assert_array_equal(candidate,control)
