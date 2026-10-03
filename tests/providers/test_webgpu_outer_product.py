"""Resource legality and independent numeric checks for staged outer products."""
import os
import numpy as np
import pytest
import tensor
from tensor.compiler.webgpu_lowering import outer_product_matmul_schedule


@pytest.mark.parametrize('change', [
    {'rows':0}, {'micro_m':3}, {'threads':128}, {'unroll':3},
    {'lhs_pad':-1}, {'rhs_pad':True}, {'owner_axis':'invalid'},
    {'lhs_layout':'invalid'}, {'fma':1}, {'epilogue':'invalid'}, {'explicit_unroll':1},
    {'dot_width':True}, {'dot_width':3}, {'tile_k':16,'unroll':16,'dot_width':2},
    {'packed_pairs':1}, {'packed_pairs':True},
    {'packed_pairs':True,'dtype':'float32','dot_width':2},
    {'half_accum':1},{'half_accum':True},{'group_order':'invalid'},
    {'packed_pairs':True,'dtype':'float16','dot_width':2,'lhs_layout':'mk'},
    {'tile_m':128,'tile_n':128,'tile_k':32,'micro_m':8,'micro_n':8,'lhs_pad':1},
])
def test_illegal_outer_product_resources_rejected(change):
    with pytest.raises(ValueError):
        outer_product_matmul_schedule(**{'rows':33,'depth':37,'columns':65,**change})


@pytest.mark.skipif(os.environ.get('TENSOR_WEBGPU')!='1',reason='requires native WebGPU')
@pytest.mark.parametrize('dtype,mode', [
    ('float32','gemm'),('float32','linear'),('float16','gemm'),('float16','linear')])
@pytest.mark.parametrize('config', [
    dict(lhs_layout='km',lhs_pad=1,rhs_pad=1,explicit_unroll=True),
    dict(tile_m=32,tile_n=128,tile_k=32,micro_m=4,micro_n=8,threads=128,
         lhs_layout='mk',lhs_pad=1,rhs_pad=1,owner_axis='row',unroll=8,fma=False),
    dict(tile_m=32,tile_n=64,tile_k=32,micro_m=2,micro_n=4,threads=256,
         lhs_layout='km',unroll=4,dot_width=2,explicit_unroll=True),
    dict(tile_m=16,tile_n=32,tile_k=32,micro_m=2,micro_n=4,threads=64,
         lhs_layout='mk',unroll=4,dot_width=4,explicit_unroll=True),
    dict(tile_m=16,tile_n=32,tile_k=32,micro_m=2,micro_n=4,threads=64,
         unroll=4,dot_width=2,packed_pairs=True,explicit_unroll=True),
    dict(tile_m=32,tile_n=64,tile_k=32,micro_m=2,micro_n=4,threads=256,
         lhs_layout='km',unroll=4,group_order='row',explicit_unroll=True),
    dict(tile_m=16,tile_n=32,tile_k=32,micro_m=2,micro_n=4,threads=64,
         unroll=4,dot_width=2,packed_pairs=True,group_order='row',explicit_unroll=True),
])
def test_outer_product_tails_precision_and_fused_epilogue(tmp_path,dtype,mode,config):
    from benchmarks.inference.webgpu_outer_product_search import source
    if config.get('packed_pairs') and dtype!='float16':
        pytest.skip('packed half pairs require F16 operands')
    # Multiple workgroups, partial M/N/K tiles and a long reduction.
    m,n,k=35,131,1027
    path=tmp_path/'outer.py';artifact=path.with_suffix('.tbin')
    path.write_text(source(m,n,k,config,mode,dtype))
    tensor.build(path,artifact,provider='webgpu')
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact)
        output=device.from_numpy(np.full(m*n,np.nan,np.float32))
        for seed,scale in ((804,1),(819,.01),(835,1e-5)):
            rng=np.random.default_rng(seed)
            a=(rng.normal(size=(m,k))*scale).astype(dtype)
            b=rng.normal(size=(n,k)).astype(dtype)
            bias=(rng.normal(size=n)*scale).astype(dtype)
            aa,bb=a.astype(np.float64),b.astype(np.float64)
            expected=aa@bb.T
            bound=(np.abs(aa)@np.abs(bb).T)*3e-6+1e-10
            if mode=='linear':
                expected=np.maximum(expected+bias.astype(np.float64),0)
                bound+=np.abs(bias.astype(np.float64))*3e-6
            uploaded=[device.from_numpy(v.ravel()) for v in
                      ((a,b,bias) if mode=='linear' else (a,b))]
            try:
                device.write(output,np.full(m*n,np.nan,np.float32))
                kernel.launch(*uploaded,output)
                actual=output.to_numpy().reshape(m,n)
                assert np.isfinite(actual).all()
                assert np.all(np.abs(actual-expected)<=bound)
            finally:
                for value in uploaded:value.release()


def test_explicit_unroll_expands_only_marked_bounded_loops(tmp_path):
    from tensor.artifacts.portable import export_spec
    from tensor.compiler.webgpu_lowering import lower_explicit_unroll
    path=tmp_path/'loops.py'
    path.write_text('''import tilelang.language as T
@T.prim_func
def kernel(x:T.Tensor((64,),"float32"),out:T.Tensor((64,),"float32")):
    with T.Kernel(1,threads=64):
        tx=T.get_thread_binding()
        acc=T.alloc_var("float32")
        acc=0
        for tile in T.serial(31):
            for u in T.unroll(4):
                acc=acc+x[tx]*(tile+u)
        out[tx]=acc
def tensor_export():return {"kernel":kernel}
''')
    kernel=export_spec(path,None)['kernel']
    assert lower_explicit_unroll(kernel) is kernel
    transformed=lower_explicit_unroll(kernel.with_attr('tensor.webgpu.loop_unroll','explicit'))
    text=transformed.script()
    assert 'T.unroll' not in text
    assert 'range(31)' in text
    with pytest.raises(ValueError,match='unroll mode'):
        lower_explicit_unroll(kernel.with_attr('tensor.webgpu.loop_unroll','invalid'))
    path.write_text(path.read_text().replace('T.unroll(4)','T.unroll(32)'))
    with pytest.raises(ValueError,match='at most 16'):
        lower_explicit_unroll(export_spec(path,None)['kernel'].with_attr('tensor.webgpu.loop_unroll','explicit'))
