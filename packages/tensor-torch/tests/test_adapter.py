"""Client packaging, FX contracts, and opt-in actual CUDA execution."""
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')
# Source path is only a test aid; entry-point discovery is tested on installed wheels.
sys.path.insert(0, str(Path(__file__).parents[3] / 'packages/tensor-torch/src'))
import tensor_torch as tt

GPU = pytest.mark.skipif(os.environ.get('TENSOR_P4_CUDA') != '1', reason='set TENSOR_P4_CUDA=1')


def test_core_does_not_import_adapter_or_torch():
    result = subprocess.run([sys.executable, '-c', '''
import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'torch','tensor_torch','tilelang','tvm','tvm_ffi'}:
   raise ImportError(fullname)
sys.meta_path.insert(0,Guard())
import tensor
assert not {'torch','tensor_torch','tilelang','tvm','tvm_ffi'} & sys.modules.keys()
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cpu_and_unsupported_nodes_fall_back_without_compiler(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path)
    a, b = torch.randn(129), torch.randn(129)
    with torch.inference_mode():
        function = torch.compile(lambda a,b:torch.relu(a*2+b).sin(), backend=backend, fullgraph=True)
        torch.testing.assert_close(function(a,b), torch.relu(a*2+b).sin())
    assert not backend.report['regions']
    assert backend.report['fallback_nodes']
    assert not list(tmp_path.glob('*.tbin'))


def test_inference_backend_preserves_eager_autograd(tmp_path):
    backend = tt.Backend(cache_dir=tmp_path)
    a = torch.randn(33, requires_grad=True)
    function = torch.compile(lambda a:(a*2).relu(),backend=backend)
    function(a).sum().backward()
    torch.testing.assert_close(a.grad, 2*(a>0).float())
    assert backend.report['autograd'][0]['stage'] == 'fallback'


@GPU
@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16,torch.float32])
def test_fused_pointwise_shapes_nan_broadcast_and_cache(tmp_path, dtype):
    backend = tt.Backend(cache_dir=tmp_path)
    function = torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend,fullgraph=True,dynamic=True)
    with torch.inference_mode():
        for length in (129,257,129):
            a = torch.randn(3,length,device='cuda',dtype=dtype)
            b = torch.randn(length,device='cuda',dtype=dtype)
            a[0,0] = float('nan')
            torch.testing.assert_close(function(a,b),torch.relu(a*2+b),equal_nan=True)
    specs = [s for r in backend.report['regions'] for s in r['specializations']]
    assert len(specs) == 2 and all(s['kind']=='pointwise' for s in specs)
    assert not backend.report['fallback_nodes']
    # A fresh process must execute the same cached graph with all frontend imports blocked.
    script = '''
import builtins,sys,torch,tensor_torch as tt
original_import=builtins.__import__
def guarded_import(name,*args,**kwargs):
 if name.split('.')[0] in {'tilelang','tvm','tvm_ffi'}: raise ImportError('compiler blocked: '+name)
 return original_import(name,*args,**kwargs)
builtins.__import__=guarded_import
b=tt.Backend(cache_dir=sys.argv[1]); dtype=getattr(torch,sys.argv[2])
f=torch.compile(lambda a,b:torch.relu(a*2+b),backend=b,fullgraph=True,dynamic=True)
with torch.inference_mode():
 a=torch.randn(3,129,device='cuda',dtype=dtype);c=torch.randn(129,device='cuda',dtype=dtype)
 torch.testing.assert_close(f(a,c),torch.relu(a*2+c))
assert all(s['cache_hit'] for r in b.report['regions'] for s in r['specializations'])
assert not {'tilelang','tvm','tvm_ffi'} & sys.modules.keys()
'''
    result=subprocess.run([sys.executable,'-c',script,str(tmp_path),str(dtype).removeprefix('torch.')],capture_output=True,text=True,timeout=60)
    assert result.returncode == 0,result.stderr


@GPU
@pytest.mark.parametrize('linear', [False,True])
@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16])
def test_gemm_tail_bias_relu_and_mlp(tmp_path, linear, dtype):
    backend=tt.Backend(cache_dir=tmp_path)
    a=torch.randn(33,64,device='cuda',dtype=dtype)
    w=torch.randn((65,64) if linear else (64,65),device='cuda',dtype=dtype)
    bias=torch.randn(65,device='cuda',dtype=dtype)
    function=lambda a,w,b:torch.relu(torch.nn.functional.linear(a,w,b) if linear else a@w+b)
    with torch.inference_mode():
        compiled=torch.compile(function,backend=backend,fullgraph=True)
        torch.testing.assert_close(compiled(a,w,bias),function(a,w,bias),atol=.005,rtol=.02)
        mlp=lambda a,w1,w2:torch.relu(torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(a,w1)),w2))
        w2=torch.randn(32,65,device='cuda',dtype=dtype)
        wt=w if linear else w.t().contiguous()
        torch.testing.assert_close(torch.compile(mlp,backend=backend,fullgraph=True)(a,wt,w2),mlp(a,wt,w2),atol=.25 if dtype==torch.bfloat16 else .05,rtol=.03)
    assert len([s for r in backend.report['regions'] for s in r['specializations']]) == 3


@GPU
@pytest.mark.parametrize('shape,causal', [((1,2,129,64),True),((2,2,257,64),False),((1,2,128,128),True)])
@pytest.mark.parametrize('dtype', [torch.float16,torch.bfloat16])
def test_fx_attention_online_softmax(tmp_path, shape, causal, dtype):
    backend=tt.Backend(cache_dir=tmp_path)
    args=tuple(torch.randn(shape,device='cuda',dtype=dtype) for _ in range(3))
    function=lambda q,k,v:torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
    with torch.inference_mode():
        output=torch.compile(function,backend=backend,fullgraph=True)(*args)
        torch.testing.assert_close(output,function(*args),atol=.02 if dtype==torch.bfloat16 else .002,rtol=.02)
    assert backend.report['regions'][0]['specializations'][0]['kind']=='attention'



@GPU
def test_graph_break_partial_fallback_and_layout(tmp_path):
    backend=tt.Backend(cache_dir=tmp_path)
    def function(a,b):
        c=torch.relu(a*2+b)
        torch._dynamo.graph_break()
        return torch.relu(c.sin()+b)
    with torch.inference_mode():
        a,b=(torch.randn(257,device='cuda') for _ in range(2))
        compiled=torch.compile(function,backend=backend)
        torch.testing.assert_close(compiled(a,b),function(a,b))
        noncontiguous=a[::2]
        torch.testing.assert_close(compiled(noncontiguous,b[::2]),function(noncontiguous,b[::2]))
    assert backend.report['graphs'] >= 2
    assert any('sin' in n for n in backend.report['fallback_nodes'])
    assert any(r['specializations'] for r in backend.report['regions'])


@GPU
def test_custom_op_fake_dynamic_mutation_and_module_loading(tmp_path):
    import tensor
    artifact=tmp_path/'affine.tbin'
    tensor.build(Path(__file__).parents[3]/'examples/dynamic_affine.py',artifact,compiler='nvrtc')
    (tmp_path/'tensor.json').write_text(json.dumps({'formatVersion':1,'name':'torch-test','version':'0.1.0',
        'tensorAbi':1,'exports':{'affine':{'artifacts':['affine.tbin']}}}))
    from tensor.artifacts.modules import install
    install(tmp_path,cache_dir=tmp_path/'module-cache')
    kernel=tt.load('torch-test::affine',project=tmp_path,module_cache=tmp_path/'module-cache')
    a,b=(torch.randn(129,device='cuda') for _ in range(2))
    assert all(v=='SUCCESS' for v in torch.library.opcheck(kernel._op,([a,b],[2.0])).values())
    output=torch.empty_like(a)
    assert all(v=='SUCCESS' for v in torch.library.opcheck(kernel._into_op,([a,b],[2.0],[output])).values())
    with torch.inference_mode():
        compiled=torch.compile(lambda a,b:kernel(a,b,2.0),backend='aot_eager',dynamic=True,fullgraph=True)
        for size in (129,257):
            aa,bb=(torch.randn(size,device='cuda') for _ in range(2))
            torch.testing.assert_close(compiled(aa,bb),torch.relu(aa*2+bb))
        kernel.into(a,b,2.0,outputs=[output])
        torch.testing.assert_close(output,torch.relu(a*2+b))
        with pytest.raises(ValueError,match='alias'):
            kernel.into(a,b,2.0,outputs=[a])
        with pytest.raises(ValueError,match='contiguous'):
            kernel(a[::2],b[::2],2.0)
        with pytest.raises(ValueError,match='dtype'):
            kernel(a.half(),b.half(),2.0)


@GPU
def test_current_stream_lifetime_and_capture(tmp_path):
    backend=tt.Backend(cache_dir=tmp_path)
    function=torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend,fullgraph=True)
    producer,consumer=torch.cuda.Stream(),torch.cuda.Stream()
    with torch.inference_mode(),torch.cuda.stream(producer):
        a,b=(torch.randn(129,device='cuda') for _ in range(2))
        expected=torch.relu(a*2+b)
        torch.cuda._sleep(1_000_000)
    consumer.wait_stream(producer)
    with torch.inference_mode(),torch.cuda.stream(consumer):
        result=function(a,b)
        observed=result+1
        del result
        # Exercise pointer rebinding and immediate input destruction after submission.
        for _ in range(20):
            x,y=torch.ones(129,device='cuda'),torch.ones(129,device='cuda')
            out=function(x,y)
            del x,y
        torch.testing.assert_close(observed,expected+1)
        torch.testing.assert_close(out,torch.full_like(out,3))
        # Prepared launches must be capturable without host synchronization.
        static_a,static_b=torch.ones(129,device='cuda'),torch.ones(129,device='cuda')
        function(static_a,static_b)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=consumer):
            captured=function(static_a,static_b)
        static_a.fill_(2)
        graph.replay()
        torch.testing.assert_close(captured,torch.full_like(captured,5))


@GPU
def test_aot_autograd_uses_compiled_forward_and_backward_region(tmp_path):
    backend=tt.Backend(cache_dir=tmp_path,training=True)
    function=torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend,fullgraph=True)
    a,b=(torch.randn(129,device='cuda',requires_grad=True) for _ in range(2))
    output=function(a,b)
    output.sum().backward()
    torch.testing.assert_close(a.grad,2*(a*2+b>0).float())
    torch.testing.assert_close(b.grad,(a*2+b>0).float())
    assert {g['stage'] for g in backend.report['autograd']} == {'forward','backward'}
    assert len([r for r in backend.report['regions'] if r['specializations']]) >= 2
    assert any('threshold_backward' in n for n in backend.report['fallback_nodes'])

@GPU
def test_prepared_binding_rejects_changed_storage_and_metadata(tmp_path):
    backend=tt.Backend(cache_dir=tmp_path)
    a,b=(torch.randn(129,device='cuda') for _ in range(2))
    with torch.inference_mode():
        function=torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend)
        function(a,b)
        kernel=tt.load(backend.report['regions'][0]['specializations'][0]['artifact'])
        prepared=kernel.prepare(a,b)
        a.fill_(2)
        prepared()
        torch.testing.assert_close(prepared.outputs[0],torch.relu(a*2+b))
        a.set_(torch.empty_like(a))
        with pytest.raises(ValueError,match='storage changed'):
            prepared()


@GPU
def test_unaligned_views_and_specialization_limit_fall_back(tmp_path):
    backend=tt.Backend(cache_dir=tmp_path,max_specializations=1)
    function=torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend,dynamic=True)
    with torch.inference_mode():
        a,b=(torch.randn(129,device='cuda') for _ in range(2))
        torch.testing.assert_close(function(a,b),torch.relu(a*2+b))
        torch.testing.assert_close(function(a[1:],b[1:]),torch.relu(a[1:]*2+b[1:]))
    assert any('limit' in f for r in backend.report['regions'] for f in r['fallbacks'])
    backend=tt.Backend(cache_dir=tmp_path)
    with torch.inference_mode():
        compiled=torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend)
        torch.testing.assert_close(compiled(a[1:],b[1:]),torch.relu(a[1:]*2+b[1:]))
    assert any('alignment' in f for r in backend.report['regions'] for f in r['fallbacks'])


@GPU
def test_corrupt_fx_cache_is_rebuilt(tmp_path):
    a,b=(torch.randn(129,device='cuda') for _ in range(2))
    function=lambda a,b:torch.relu(a*2+b)
    backend=tt.Backend(cache_dir=tmp_path)
    with torch.inference_mode():
        torch.compile(function,backend=backend)(a,b)
        artifact=Path(backend.report['regions'][0]['specializations'][0]['artifact'])
        artifact.write_bytes(b'corrupt')
        fresh=tt.Backend(cache_dir=tmp_path)
        torch.testing.assert_close(torch.compile(function,backend=fresh)(a,b),function(a,b))
    assert not fresh.report['regions'][0]['specializations'][0]['cache_hit']

@GPU
@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32])
def test_pointwise_activation_precision_and_arithmetic(tmp_path,dtype):
    a=torch.tensor([-30,-12,-1,-0.01,0,.01,1,12,30]+[.1]*120,device='cuda',dtype=dtype)
    b=torch.full_like(a,2)
    backend=tt.Backend(cache_dir=tmp_path)
    function=lambda a,b:torch.tanh(torch.sigmoid(-(a-b)/b))
    with torch.inference_mode():
        compiled=torch.compile(function,backend=backend,fullgraph=True)
        torch.testing.assert_close(compiled(a,b),function(a,b),atol=.001 if dtype==torch.float16 else 1e-6,rtol=.001 if dtype==torch.float16 else 1e-6)
        sigmoid=torch.compile(lambda a:a.sigmoid(),backend=backend,fullgraph=True)
        torch.testing.assert_close(sigmoid(a),a.sigmoid(),atol=1e-7,rtol=.001 if dtype==torch.float16 else 1e-6)
    assert all(r['specializations'] and not r['fallbacks'] for r in backend.report['regions'])

@GPU
def test_runtime_scalars_reuse_preparation_without_changing_results(tmp_path):
    import tensor
    artifact=tmp_path/'affine.tbin'
    tensor.build(Path(__file__).parents[3]/'examples/dynamic_affine.py',artifact,compiler='nvrtc')
    kernel=tt.load(artifact)
    a,b=(torch.randn(129,device='cuda') for _ in range(2))
    with torch.inference_mode():
        for scale in (2.0,.5,-3.0):
            torch.testing.assert_close(kernel(a,b,scale),torch.relu(a*scale+b))
    assert len(kernel._prepared)==1

@GPU
def test_attention_shared_operand_and_unaligned_storage_fallback(tmp_path):
    backend=tt.Backend(cache_dir=tmp_path)
    function=lambda q:torch.nn.functional.scaled_dot_product_attention(q,q,q)
    shape=(1,2,128,64)
    from torch.nn.attention import sdpa_kernel,SDPBackend
    # Flash SDPA itself assumes vector-aligned pointers; use its math provider
    # to validate framework fallback for a deliberately unaligned view.
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        compiled=torch.compile(function,backend=backend,fullgraph=True,dynamic=False)
        aligned=torch.randn(shape,device='cuda',dtype=torch.float16)
        torch.testing.assert_close(compiled(aligned),function(aligned),atol=.002,rtol=.02)
        unaligned=torch.randn(aligned.numel()+1,device='cuda',dtype=torch.float16)[1:].reshape(shape)
        torch.testing.assert_close(compiled(unaligned),function(unaligned),atol=.002,rtol=.02)
    assert any('alignment' in f for r in backend.report['regions'] for f in r['fallbacks'])


def _require_native():
    from tensor_torch.bridge import _executor
    if _executor is None:
        pytest.skip('build with TENSOR_TORCH_BUILD_NATIVE=1 for C++ executor checks')


@GPU
def test_native_prepared_scalars_metadata_streams_and_capture(tmp_path):
    _require_native()
    import tensor
    artifact = tmp_path / 'affine.tbin'
    tensor.build(Path(__file__).parents[3]/'examples/dynamic_affine.py',artifact,compiler='nvrtc')
    kernel = tt.load(artifact)
    a,b = (torch.randn(129,device='cuda') for _ in range(2))
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.inference_mode():
        for scale in (2.0,.5,-3.0):
            with torch.cuda.stream(producer):
                a.fill_(2)
                call = kernel.prepare(a,b,scale)
                assert call._native is not None
            consumer.wait_stream(producer)
            with torch.cuda.stream(consumer):
                call()
                torch.testing.assert_close(call.outputs[0],torch.relu(a*scale+b))
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=consumer):
                    call()
                a.fill_(3)
                graph.replay()
                torch.testing.assert_close(call.outputs[0],torch.relu(a*scale+b))
                call.outputs[0].resize_(128)
                with pytest.raises(ValueError,match='metadata or storage changed'):
                    call()
                call.outputs[0].resize_(129)
                a.resize_(128)
                with pytest.raises(ValueError,match='metadata or storage changed'):
                    call()
                a.resize_(129)


@GPU
def test_native_fx_pointer_rebinding_concurrency_and_alignment(tmp_path):
    _require_native()
    from concurrent.futures import ThreadPoolExecutor
    from tensor_torch.bridge import Kernel, LaunchPlan
    backend = tt.Backend(cache_dir=tmp_path)
    a,b = (torch.randn(129,device='cuda') for _ in range(2))
    with torch.inference_mode():
        torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend,fullgraph=True)(a,b)
        artifact = backend.report['regions'][0]['specializations'][0]['artifact']
        kernel = Kernel(artifact)
        fallback = lambda a,b:torch.relu(a*2+b)
        plan = LaunchPlan(kernel,(a,b),fallback)
        assert plan._native is not None and plan.prepared._native is None
        assert not plan.prepared.tensors
        unaligned = torch.randn(130,device='cuda')[1:]
        torch.testing.assert_close(plan(unaligned,b),fallback(unaligned,b))
        with pytest.raises(ValueError,match='metadata changed'):
            plan(a[:128],b[:128])
        del a,b
    def worker(seed):
        stream = torch.cuda.Stream()
        with torch.inference_mode(),torch.cuda.stream(stream):
            for i in range(30):
                x,y = torch.full((129,),float(seed+i),device='cuda'),torch.ones(129,device='cuda')
                output = plan(x,y)
                del x,y
                torch.testing.assert_close(output,torch.full_like(output,2*(seed+i)+1))
        stream.synchronize()
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(worker,range(3)))


@GPU
def test_native_fixed_and_fx_calls_reject_closed_session(tmp_path):
    _require_native()
    from tensor_torch.bridge import Kernel, LaunchPlan
    a,b = (torch.randn(129,device='cuda') for _ in range(2))
    backend = tt.Backend(cache_dir=tmp_path)
    with torch.inference_mode():
        torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend,fullgraph=True)(a,b)
        kernel = Kernel(backend.report['regions'][0]['specializations'][0]['artifact'])
        call = kernel.prepare(a,b)
        plan = LaunchPlan(kernel,(a,b),lambda a,b:torch.relu(a*2+b))
        assert call._native is not None and plan._native is not None
        tt.close()
        with pytest.raises(RuntimeError,match='closed session'):
            call()
        with pytest.raises(RuntimeError,match='closed session'):
            plan(a,b)


@GPU
def test_native_cuda_failures_preserve_provider_exception(tmp_path):
    _require_native()
    from tensor.providers.cuda import CudaError
    from tensor_torch.bridge import Kernel, _native_plan
    a,b = (torch.randn(129,device='cuda') for _ in range(2))
    backend = tt.Backend(cache_dir=tmp_path)
    with torch.inference_mode():
        torch.compile(lambda a,b:torch.relu(a*2+b),backend=backend,fullgraph=True)(a,b)
        kernel = Kernel(backend.report['regions'][0]['specializations'][0]['artifact'])
        prepared = kernel.prepare(a,b)
        # Zero block width is rejected by the driver before any device execution.
        prepared.launch = {**prepared.launch,'block': [0,1,1]}
        invalid = _native_plan(prepared,(a,b),prepared.outputs,fixed=True)
        with pytest.raises(CudaError,match='cuLaunchKernel failed with CUDA error'):
            invalid()
        prepared()
        torch.testing.assert_close(prepared.outputs[0],torch.relu(a*2+b))
