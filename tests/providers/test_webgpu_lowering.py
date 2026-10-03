"""Numerical contracts for reusable WebGPU reductions and register tiles."""
import os
from pathlib import Path
import numpy as np
import pytest
import tensor
from tensor.artifacts.format import read_artifact

NATIVE=pytest.mark.skipif(os.environ.get('TENSOR_WEBGPU')!='1',reason='requires a native WebGPU adapter')


@NATIVE
@pytest.mark.parametrize('owner_axis,tile_n',[('row',5),('column',8)])
@pytest.mark.parametrize('k_layout',['blocked','striped'])
@pytest.mark.parametrize('dot_width',[1,2,4])
def test_partitioned_matmul_tail_outputs_and_half_subnormals(tmp_path,owner_axis,tile_n,k_layout,dot_width):
    from tensor.compiler.webgpu_lowering import partitioned_matmul_schedule
    from tensor_llm.kernels import emit
    from tensor_llm.webgpu_kernels import round_half
    r,k,o=3,128,7
    lhs=round_half('x[({row}) * 128 + ({k})]')
    rhs='T.cast(w[({column}) * 128 + ({k})], "float32")'
    body=partitioned_matmul_schedule(r,k,o,lhs,rhs,tile_m=8,tile_n=tile_n,threads=64,
                                     partitions=8,unroll=4,dot_width=dot_width,owner_axis=owner_axis,k_layout=k_layout)
    path=tmp_path/'partitioned.py';artifact=path.with_suffix('.tbin')
    path.write_text(emit([('x',r*k,'float32'),('w',o*k,'float16'),('out',r*o,'float32')],body))
    tensor.build(path,artifact,provider='webgpu')
    weight=np.random.default_rng(403).normal(size=(o,k)).astype(np.float16)
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);weights=device.from_numpy(weight.ravel());output=device.full(r*o,np.nan)
        for seed,scale in ((29,.01),(202,1),(303,.00001)):
            inputs=(np.random.default_rng(seed).normal(size=(r,k))*scale).astype(np.float32)
            lhs=inputs.astype(np.float16).astype(np.float64);rhs=weight.astype(np.float64)
            expected=lhs@rhs.T;bound=np.abs(lhs)@np.abs(rhs).T*3e-6+1e-10
            device.write(output,np.full(r*o,np.nan,dtype=np.float32))
            input=device.from_numpy(inputs.ravel());kernel.launch(input,weights,output)
            actual=output.to_numpy().reshape(r,o)
            assert np.isfinite(actual).all()
            assert np.all(np.abs(actual-expected)<=bound)
            input.release()


@pytest.mark.parametrize('consumer', ['none', 'read', 'write', 'clear', 'reduce'])
def test_whole_loop_accumulators_require_exclusive_fragment_ownership(tmp_path, consumer):
    from tensor.artifacts.portable import export_spec
    from tensor.compiler.webgpu_lowering import lower_simt_gemm
    root=Path(__file__).resolve().parents[2]
    source=(root/'examples/webgpu_gemm.py').read_text()
    gemm='T.gemm(aa, bb, cc, transpose_B=TRANSPOSE_B)'
    edits={
        'none':gemm,
        'read':'out[0, 0] = cc[0, 0]\n            '+gemm,
        'write':'cc[0, 0] = 1\n            '+gemm,
        'clear':gemm.replace(')', ', clear_accum=True)'),
        'reduce':'T.reduce_sum(cc, partial, dim=1)\n            '+gemm,
    }
    if consumer=='reduce':
        source=source.replace('T.clear(cc)', 'partial = T.alloc_fragment((BLOCK,), "float32")\n        T.clear(cc)')
    path=tmp_path/'ownership.py';path.write_text(source.replace(gemm, edits[consumer]))
    kernel=export_spec(path, None)['kernel'].with_attr('tensor.webgpu.gemm_accumulation','register')
    lowered=lower_simt_gemm(kernel).script()
    assert ('wgpu_gemm_loop_acc_' in lowered)==(consumer=='none')
    assert 'wgpu_gemm_loop_acc_' not in lower_simt_gemm(
        kernel.with_attr('tensor.webgpu.gemm_accumulation', 'shared')).script()


def test_attention_keeps_shared_intermediates_and_ordered_lowering():
    import tvm
    from tensor.artifacts.portable import export_spec
    from tensor.compiler.webgpu_lowering import lower_simt_gemm
    root=Path(__file__).resolve().parents[2]
    kernel=export_spec(root/'examples/flash_attention.py',None)['kernel']
    default=lower_simt_gemm(kernel.with_attr('tensor.webgpu.gemm_accumulation','register'))
    fallback=lower_simt_gemm(kernel.with_attr('tensor.webgpu.gemm_accumulation','shared'))
    assert 'wgpu_gemm_loop_acc_' not in default.script()
    assert tvm.ir.structural_equal(default.body, fallback.body)


@pytest.mark.parametrize('depth',[512,1024,2048,2560])
@pytest.mark.parametrize('mode',['auto','register','shared'])
def test_whole_loop_default_depth_gate_and_explicit_override(tmp_path,depth,mode):
    from tensor.artifacts.portable import export_spec
    from tensor.compiler.webgpu_lowering import lower_simt_gemm
    root=Path(__file__).resolve().parents[2]
    path=tmp_path/'depth.py'
    path.write_text((root/'examples/webgpu_gemm.py').read_text().replace('K = 37',f'K = {depth}'))
    kernel=export_spec(path,None)['kernel'].with_attr('tensor.webgpu.gemm_accumulation',mode)
    assert ('wgpu_gemm_loop_acc_' in lower_simt_gemm(kernel).script()) == (
        mode=='register' or mode=='auto' and depth>=2048)
    with pytest.raises(ValueError,match='accumulation'):
        lower_simt_gemm(kernel.with_attr('tensor.webgpu.gemm_accumulation','invalid'))


@NATIVE
@pytest.mark.parametrize('transpose_a', [False, True])
@pytest.mark.parametrize('transpose_b', [False, True])
@pytest.mark.parametrize('dtype', ['float16', 'float32'])
def test_whole_loop_accumulators_seeded_transposes_odd_tiles_and_zero_trip(
        tmp_path, transpose_a, transpose_b, dtype):
    m,n,k,bm,bn,bk=9,11,13,7,9,5
    a_shape=(k,m) if transpose_a else (m,k)
    b_shape=(n,k) if transpose_b else (k,n)
    aa_shape=(bk,bm) if transpose_a else (bm,bk)
    bb_shape=(bn,bk) if transpose_b else (bk,bn)
    a_copy='a[tile * BK, by * BM]' if transpose_a else 'a[by * BM, tile * BK]'
    b_copy='b[bx * BN, tile * BK]' if transpose_b else 'b[tile * BK, bx * BN]'
    path=tmp_path/'seeded.py';artifact=path.with_suffix('.tbin')
    path.write_text(f'''import tilelang.language as T
BM, BN, BK = {bm}, {bn}, {bk}
@T.prim_func
def kernel(a:T.Tensor({a_shape},"{dtype}"),b:T.Tensor({b_shape},"{dtype}"),
           seed:T.Tensor(({m},{n}),"float32"),out:T.Tensor(({m},{n}),"float32"),tiles:T.int32):
    T.func_attr({{"tensor.webgpu.gemm_accumulation":"register"}})
    with T.Kernel(T.ceildiv({n},BN),T.ceildiv({m},BM),threads=32) as (bx,by):
        aa=T.alloc_shared({aa_shape},"{dtype}")
        bb=T.alloc_shared({bb_shape},"{dtype}")
        cc=T.alloc_fragment((BM,BN),"float32")
        T.copy(seed[by * BM,bx * BN],cc)
        for tile in T.serial(tiles):
            T.copy({a_copy},aa)
            T.copy({b_copy},bb)
            T.gemm(aa,bb,cc,transpose_A={transpose_a},transpose_B={transpose_b})
        T.copy(cc,out[by * BM,bx * BN])
def tensor_export():return {{"kernel":kernel}}
''')
    tensor.build(path,artifact,provider='webgpu')
    assert b'wgpu_gemm_loop_acc_' in read_artifact(artifact)[1]['kernel.wgsl']
    rng=np.random.default_rng(761)
    a=rng.normal(size=a_shape).astype(dtype);b=rng.normal(size=b_shape).astype(dtype)
    seed=rng.normal(size=(m,n)).astype(np.float32)
    expected=seed+(a.T if transpose_a else a).astype(np.float32)@(b.T if transpose_b else b).astype(np.float32)
    with tensor.Device(provider='webgpu') as device:
        aa,bb,ss=[device.from_numpy(x) for x in (a,b,seed)]
        out=device.zeros((m,n));kernel=device.load(artifact)
        for tiles,reference in ((3,expected),(0,seed),(3,expected)):
            kernel.launch(aa,bb,ss,out,tiles)
            np.testing.assert_allclose(out.to_numpy(),reference,rtol=2e-5,atol=2e-5)


@NATIVE
@pytest.mark.parametrize('operation',['sum','max'])
@pytest.mark.parametrize('axis',[0,1])
def test_parallel_reduction_odd_extents_and_accumulation(tmp_path,operation,axis):
    rows,cols=7,65
    count=cols if axis==0 else rows
    path=tmp_path/'reduce.py';artifact=tmp_path/'reduce.tbin'
    path.write_text(f'''import tilelang.language as T
@T.prim_func
def kernel(x:T.Tensor(({rows},{cols}),"float32"),previous:T.Tensor(({count},),"float32"),out:T.Tensor(({count},),"float32")):
    T.func_attr({{"tensor.webgpu.reduction":"tree"}})
    with T.Kernel(1,threads=128):
        tile=T.alloc_shared(({rows},{cols}),"float32")
        result=T.alloc_shared(({count},),"float32")
        T.copy(x,tile)
        T.copy(previous,result)
        T.reduce_{operation}(tile,result,dim={axis},clear=False)
        T.copy(result,out)
def tensor_export():return {{"kernel":kernel}}
''')
    tensor.build(path,artifact,provider='webgpu')
    rng=np.random.default_rng(814+axis)
    x=rng.normal(size=(rows,cols)).astype(np.float32);previous=rng.normal(size=count).astype(np.float32)
    expected=x.sum(axis=axis)+previous if operation=='sum' else np.maximum(x.max(axis=axis),previous)
    with tensor.Device(provider='webgpu') as device:
        out=device.zeros(count);device.load(artifact).launch(device.from_numpy(x),device.from_numpy(previous),out)
        np.testing.assert_allclose(out.to_numpy(),expected,atol=3e-6,rtol=3e-6)


@NATIVE
@pytest.mark.parametrize('micro',[1,2,4])
@pytest.mark.parametrize('transpose',[False,True])
def test_register_gemm_microtiles_transpose_and_tails(tmp_path,micro,transpose):
    root=Path(__file__).resolve().parents[2]
    text=(root/'examples/webgpu_gemm.py').read_text().replace('TRANSPOSE_B = False',f'TRANSPOSE_B = {transpose}')
    text=text.replace('OUTPUT_DTYPE = "float16"','OUTPUT_DTYPE = "float32"')
    text=text.replace('kernel = linear',f'kernel = linear.with_attr("tensor.webgpu.gemm_microtile", {micro}).with_attr("tensor.webgpu.gemm_accumulation", "register")')
    path=tmp_path/'gemm.py';artifact=tmp_path/'gemm.tbin';path.write_text(text)
    tensor.build(path,artifact,provider='webgpu')
    rng=np.random.default_rng(92);a=rng.normal(size=(33,37)).astype(np.float16)
    b=rng.normal(size=(65,37) if transpose else (37,65)).astype(np.float16);bias=rng.normal(size=65).astype(np.float16)
    expected=np.maximum(a.astype(np.float32)@(b.T if transpose else b).astype(np.float32)+bias,0)
    with tensor.Device(provider='webgpu') as device:
        actual=device.load(artifact)(device.from_numpy(a),device.from_numpy(b),device.from_numpy(bias))
        np.testing.assert_allclose(actual.to_numpy(),expected,rtol=2e-5,atol=2e-5)


@NATIVE
@pytest.mark.parametrize('schedule',[{}, {'lhs_transpose':True,'lhs_pad':1},
    {'dot_width':4,'lhs_transpose':True,'lhs_pad':1}, {'dot_width':4,'unroll':True}])
def test_register_schedule_shared_layout_vector_dots_and_tails(tmp_path,schedule):
    from tensor.compiler.webgpu_lowering import register_matmul_schedule
    m,k,n=33,128,65
    rhs=f'T.if_then_else(bx * 32 + i < {n}, T.cast(w[(bx * 32 + i) * {k} + tile * 64 + j], "float32"), 0)'
    body=register_matmul_schedule(m,k,n,'value',rhs,tile_k=64,**schedule)
    text='import tilelang.language as T\n@T.prim_func\n'+f'def kernel(x:T.Tensor(({m*k},),"float32"),w:T.Tensor(({n*k},),"float16"),out:T.Tensor(({m*n},),"float32")):\n'
    text+='\n'.join('    '+line for line in body.splitlines())+'\ndef tensor_export():return {"kernel":kernel}\n'
    path=tmp_path/'register.py';artifact=path.with_suffix('.tbin');path.write_text(text)
    tensor.build(path,artifact,provider='webgpu')
    rng=np.random.default_rng(109);a=rng.normal(size=(m,k)).astype(np.float16).astype(np.float32)
    b=rng.normal(size=(n,k)).astype(np.float16)
    with tensor.Device(provider='webgpu') as device:
        out=device.zeros(m*n)
        device.load(artifact).launch(device.from_numpy(a.ravel()),device.from_numpy(b.ravel()),out)
        np.testing.assert_allclose(out.to_numpy().reshape(m,n),a@b.astype(np.float32).T,rtol=2e-5,atol=2e-5)


@NATIVE
@pytest.mark.parametrize('force_fallback',[False,True])
@pytest.mark.parametrize('shared_input',[False,True])
@pytest.mark.parametrize('depth,unroll,accumulators',[(256,2,2),(1024,8,8),(2560,20,4)])
def test_streamed_gemv_fused_tail_and_f32_activation_contract(tmp_path,force_fallback,shared_input,depth,unroll,accumulators):
    from tensor_llm.webgpu_kernels import source
    p=dict(r=1,k=depth,o=7,type=1,sg=True,decode_schedule='streamed',lanes=32,threads=64,
           micro_rows=2,dot_width=4,unroll=unroll,accumulators=accumulators,k_layout='striped',shared_input=shared_input)
    text=source('ffn',p)
    if force_fallback:text=text.replace('T.call_extern("uint32", "tensor_subgroup_size")','T.uint32(0)')
    path=tmp_path/'gemv.py';artifact=path.with_suffix('.tbin');path.write_text(text)
    tensor.build(path,artifact,provider='webgpu')
    rng=np.random.default_rng(172);weights=[rng.normal(size=(7,depth)).astype(np.float16) for _ in range(2)]
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);ws=[device.from_numpy(w.ravel()) for w in weights]
        for scale in (1,.01,1e-5):
            x=(rng.normal(size=depth)*scale).astype(np.float32)
            g,u=[w.astype(np.float64)@x.astype(np.float64) for w in weights]
            expected=g/(1+np.exp(-g))*u
            out=device.from_numpy(np.full(7,np.nan,np.float32));inp=device.from_numpy(x)
            kernel.launch(inp,*ws,out);actual=out.to_numpy()
            np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=1e-7)
            inp.release();out.release()


@NATIVE
def test_subgroup_contract_and_packed_intrinsic(tmp_path):
    path=tmp_path/'packed.py';artifact=tmp_path/'packed.tbin'
    path.write_text('''import tilelang.language as T
@T.prim_func
def kernel(x:T.Tensor((128,),"float32"),out:T.Tensor((128,),"float32")):
    with T.Kernel(1,threads=128):
        tx=T.get_thread_binding()
        out[tx]=T.call_extern("float32","subgroupAdd",x[tx])
def tensor_export():return {"kernel":kernel}
''')
    tensor.build(path,artifact,provider='webgpu')
    manifest,files=read_artifact(artifact)
    assert manifest['webgpu']['required_features']==['subgroup']
    import json,zipfile
    from tensor.artifacts.format import ArtifactError
    manifest['webgpu']['required_features']=[]
    bad=tmp_path/'missing-subgroup.tbin'
    with zipfile.ZipFile(bad,'w') as archive:
        archive.writestr('manifest.json',json.dumps(manifest))
        for name,data in files.items():archive.writestr(name,data)
    with pytest.raises(ArtifactError,match='feature'):
        read_artifact(bad)
    with tensor.Device(provider='webgpu') as device:
        if 'subgroup' not in device.info['features']:pytest.skip('adapter lacks subgroups')
        out=device.zeros(128);device.load(artifact).launch(device.ones(128),out)
        values=out.to_numpy();assert np.all(values==values[0]) and values[0] in (4,8,16,32,64,128)


def test_layout_annotations_follow_substituted_buffers(tmp_path):
    """A layout annotation addresses its buffer by Var, and lower_simt_gemm
    substitutes alloc_buffers. The annotation has to be re-keyed onto the
    replacement or TileLang's layout inference cannot find the buffer."""
    from tensor.artifacts.portable import export_spec
    from tensor.compiler.webgpu_lowering import lower_simt_gemm
    import tilelang
    import tvm
    from tvm import tirx as ir
    path=tmp_path/'annotated.py'
    path.write_text("""import tilelang.language as T
from tilelang.layout import make_swizzled_layout
BK=32
@T.prim_func
def kernel(a:T.Tensor((BK,BK),"float16"),b:T.Tensor((BK,BK),"float16"),
           out:T.Tensor((BK,BK),"float16")):
    with T.Kernel(1,1,threads=32):
        aa=T.alloc_shared((BK,BK),"float16")
        bb=T.alloc_shared((BK,BK),"float16")
        T.annotate_layout({bb: make_swizzled_layout(bb, k_major=True, allow_pad=True)})
        T.copy(a,aa)
        T.copy(b,bb)
        T.copy(bb,out)
def tensor_export():return {"kernel":kernel}
""")
    lowered=lower_simt_gemm(export_spec(path,None)['kernel'])
    blocks=[]
    ir.stmt_functor.post_order_visit(lowered.body,
                                     lambda node: blocks.append(node) if isinstance(node, ir.SBlock) else None)
    annotated=[block for block in blocks
               if any(str(key)=='layout_map' for key in block.annotations.keys())]
    assert annotated, 'lower_simt_gemm dropped the layout annotation'
    for block in annotated:
        allocated={buffer.data for buffer in block.alloc_buffers}
        for var in block.annotations['layout_map'].keys():
            assert var in allocated, 'layout annotation still keyed on the replaced buffer'


@NATIVE
@pytest.mark.parametrize('force_fallback',[False,True])
def test_fused_attention_active_tail_and_qk_fallback(tmp_path,force_fallback):
    from tensor_llm.webgpu_kernels import source
    h,kh,d,cap=4,2,64,96
    text=source('attention',dict(r=1,h=h,kh=kh,d=d,cap=cap,sg=True,
                attention_schedule='partitioned_values',channels=32,value_parts=16,fused_scores=True))
    if force_fallback:text=text.replace('T.call_extern("uint32", "tensor_subgroup_size") >= 32','False')
    path=tmp_path/'attention.py';path.write_text(text)
    artifact=tmp_path/'attention.tbin';tensor.build(path,artifact,provider='webgpu')
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact)
        rng=np.random.default_rng(314);q=rng.normal(size=(h,d)).astype(np.float32)
        keys=rng.normal(size=(cap,kh,d)).astype(np.float16);values=rng.normal(size=(cap,kh,d)).astype(np.float16)
        qb=device.from_numpy(q.ravel());kb=device.from_numpy(keys.ravel());vb=device.from_numpy(values.ravel())
        out=device.full(h*d,np.nan);control=device.zeros(2,'int32')
        for position in (0,31,95):
            k=keys.copy();v=values.copy();k[position+1:]=np.nan;v[position+1:]=np.nan
            device.write(kb,k.ravel());device.write(vb,v.ravel());device.write(control,np.array([position,1],np.int32))
            device.write(out,np.full(h*d,np.nan,np.float32))
            kernel.launch(qb,kb,vb,out,control)
            actual=out.to_numpy().reshape(h,d)
            k=keys[:position+1].astype(np.float64)[:,np.arange(h)//2,:].transpose(1,0,2)
            v=values[:position+1].astype(np.float64)[:,np.arange(h)//2,:].transpose(1,0,2)
            scores=(k*q.astype(np.float64)[:,None,:]).sum(axis=2)*d**-.5
            prob=np.exp(scores-scores.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
            expected=(prob[:,:,None]*v).sum(axis=1)
            assert np.isfinite(actual).all()
            np.testing.assert_allclose(actual,expected,rtol=1e-5,atol=3e-7)
