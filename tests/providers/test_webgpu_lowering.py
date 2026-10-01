"""Numerical contracts for reusable WebGPU reductions and register tiles."""
import os
from pathlib import Path
import numpy as np
import pytest
import tensor
from tensor.artifacts.format import read_artifact

NATIVE=pytest.mark.skipif(os.environ.get('TENSOR_WEBGPU')!='1',reason='requires a native WebGPU adapter')


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
    text=text.replace('kernel = linear',f'kernel = linear.with_attr("tensor.webgpu.gemm_microtile", {micro})')
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
