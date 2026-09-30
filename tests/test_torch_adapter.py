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
sys.path.insert(0, str(Path(__file__).parents[1] / 'packages/tensor-torch/src'))
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
@pytest.mark.parametrize('dtype', [torch.float16,torch.float32])
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
def test_gemm_tail_bias_relu_and_mlp(tmp_path, linear):
    backend=tt.Backend(cache_dir=tmp_path)
    a=torch.randn(33,64,device='cuda',dtype=torch.float16)
    w=torch.randn((65,64) if linear else (64,65),device='cuda',dtype=torch.float16)
    bias=torch.randn(65,device='cuda',dtype=torch.float16)
    function=lambda a,w,b:torch.relu(torch.nn.functional.linear(a,w,b) if linear else a@w+b)
    with torch.inference_mode():
        compiled=torch.compile(function,backend=backend,fullgraph=True)
        torch.testing.assert_close(compiled(a,w,bias),function(a,w,bias),atol=.005,rtol=.02)
        mlp=lambda a,w1,w2:torch.relu(torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(a,w1)),w2))
        w2=torch.randn(32,65,device='cuda',dtype=torch.float16)
        wt=w if linear else w.t().contiguous()
        torch.testing.assert_close(torch.compile(mlp,backend=backend,fullgraph=True)(a,wt,w2),mlp(a,wt,w2),atol=.05,rtol=.02)
    assert len([s for r in backend.report['regions'] for s in r['specializations']]) == 3


@GPU
@pytest.mark.parametrize('shape,causal', [((1,2,129,64),True),((2,2,257,64),False),((1,2,128,128),True)])
def test_fx_attention_online_softmax(tmp_path, shape, causal):
    backend=tt.Backend(cache_dir=tmp_path)
    args=tuple(torch.randn(shape,device='cuda',dtype=torch.float16) for _ in range(3))
    function=lambda q,k,v:torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
    with torch.inference_mode():
        output=torch.compile(function,backend=backend,fullgraph=True)(*args)
        torch.testing.assert_close(output,function(*args),atol=.002,rtol=.02)
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
    tensor.build(Path(__file__).parents[1]/'examples/dynamic_affine.py',artifact,compiler='nvrtc')
    (tmp_path/'tensor.json').write_text(json.dumps({'formatVersion':1,'name':'torch-test','version':'0.1.0',
        'tensorAbi':1,'exports':{'affine':{'artifacts':['affine.tbin']}}}))
    from tensor.modules import install
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
@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
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
