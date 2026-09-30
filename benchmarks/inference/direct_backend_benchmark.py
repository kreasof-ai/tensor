"""Direct Tensor/TileLang/Triton comparison, including identical-cubin controls.

This is an optional research tool. Native TileLang compilation requires NVCC
and a host compiler; neither becomes a Tensor consumer dependency.
"""
from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
from torch.fx import Graph, GraphModule, symbolic_trace
from torch.fx.node import map_arg
from torch._subclasses.fake_tensor import FakeTensorMode

import tensor_torch as tt
from tensor_torch.backend import Region
from tensor_torch.bridge import _executor
from tensor.artifacts.format import read_artifact
from benchmarks.inference.fx_producer import Metadata
from benchmarks.inference.backend_benchmark import cases
from benchmarks.inference.direct_triton_kernels import Operation


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Reordered:
    def __init__(self, operation, order):
        self.operation, self.order = operation, order
        self.__name__ = 'direct_region'

    def __call__(self, *args):
        return self.operation(*(args[i] for i in self.order))


def clone_forward(original, replacements):
    graph, mapping = Graph(), {}
    for node in original.graph.nodes:
        if node.op == 'call_function' and isinstance(node.target, Region):
            mapping[node] = graph.call_function(replacements[node.target],
                map_arg(node.args,lambda n:mapping[n]),map_arg(node.kwargs,lambda n:mapping[n]))
        else:
            mapping[node] = graph.node_copy(node,lambda n:mapping[n])
    return GraphModule(original,graph).forward


def tilelang_pair(path, directory, matched):
    """Use native TVM-FFI allocation/launch, optionally with Tensor's exact cubin.

    Only device compilation is substituted. Native TileLang host codegen,
    argument validation, allocator exchange and runtime submission are intact.
    Check ABI and launch metadata before executing either generated wrapper.
    """
    import tilelang, tvm_ffi
    from tilelang import tvm
    from tvm.target import Target
    from tilelang.jit.kernel import JITKernel
    from tensor.compiler.lowering import device_signature
    from tensor.runtime.signature import resolve_launch
    manifest,files = read_artifact(path)
    primitive = list(tvm.ir.load_json(files['kernel.tirx.json'].decode()).functions.values())[0]
    target = Target({'kind':'cuda','arch':manifest['target']})
    with tilelang.transform.PassContext(opt_level=3), target:
        expected = str(tilelang.lower(primitive,target=target,enable_device_compile=False).kernel_source)
    original = tvm_ffi.get_global_func('tilelang_callback_cuda_compile')
    records = []
    directory.mkdir(parents=True,exist_ok=True)
    def compiler(code,target,pass_config=None):
        image = files['kernel.cubin'] if matched else bytes(original(code,target,pass_config))
        records.append({'cuda_sha256':digest(code.encode()),'cubin_sha256':digest(image),
                        'cuda_equals_reference':code==expected,'cuda_source':code})
        return bytearray(image)
    tvm_ffi.register_global_func('tilelang_callback_cuda_compile',compiler,override=True)
    try:
        output_indices = [i for i,d in enumerate(manifest['arguments']) if d['name'] in manifest['outputs']]
        allocating = JITKernel(primitive,out_idx=output_indices,execution_backend='tvm_ffi',
                              target=target,target_host='c')
        preallocated = JITKernel(primitive,out_idx=None,execution_backend='tvm_ffi',
                                target=target,target_host='c')
    finally:
        tvm_ffi.register_global_func('tilelang_callback_cuda_compile',original,override=True)
    assert len(records)==2
    for kernel,record in zip((allocating,preallocated),records):
        code = record.pop('cuda_source')
        entry,abi,launch = device_signature(primitive,kernel.artifact,code,{})
        assert entry==manifest['entrypoint'] and abi==manifest['abi']
        assert resolve_launch(launch,{})==manifest['launch']
        record['abi_and_launch_equal_tensor'] = True
        if matched:
            assert record['cubin_sha256']==manifest['files']['kernel.cubin']
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'device.cu').write_text(expected)
    return allocating,preallocated,{'matched_tensor_cubin':matched,'frontend_sha256':manifest['files']['kernel.tirx.json'],
        'tensor_cubin_sha256':manifest['files']['kernel.cubin'],
        'variants':dict(zip(('allocating','prepared'),records))}


def prepare_case(name,eager,args,cache,directory):
    # Attention's default argument is specialized into the source, not traced
    # as a bool Proxy (functional SDPA requires a concrete causal flag).
    if name.startswith('sdpa'):
        causal = name.endswith('True')
        def traceable(q,k,v):
            return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
    else:
        traceable = eager
    with FakeTensorMode() as mode:
        graph = symbolic_trace(traceable)
        Metadata(graph).run(*(mode.from_tensor(a) for a in args))
    backend = tt.Backend(cache_dir=cache)
    start = time.perf_counter()
    direct = backend._compile(graph,args)
    tensor_result = direct(*args)
    torch.cuda.current_stream().synchronize()
    setup = {'tensor_direct_seconds':time.perf_counter()-start}
    original = direct.__self__
    Metadata(original).run(*args)
    region_nodes = [n for n in original.graph.nodes if n.op=='call_function' and isinstance(n.target,Region)]
    region_list = [n.target for n in region_nodes]
    assert region_list and all(r.static_plan is not False and r.static_plan is not None for r in region_list)
    replacements = {'tilelang_matched':{},'tilelang_default':{},'triton':{},'triton_jit':{}}
    preallocated = {key:{} for key in replacements}
    evidence = []
    for i,node in enumerate(region_nodes):
        region = node.target
        plan,order = region.static_plan
        # Shape/dtype examples for independent Triton baseline construction.
        inputs = map_arg(node.args,lambda n:n.meta['val'])
        ordered = tuple(inputs[j] for j in order)
        kind = region.record['specializations'][0]['kind']
        start = time.perf_counter()
        for provider,matched in [('tilelang_matched',True),('tilelang_default',False)]:
            alloc,into,record = tilelang_pair(plan.kernel.path,directory/f'region-{i}'/provider,matched)
            replacements[provider][region] = Reordered(alloc,order)
            preallocated[provider][region] = into
            evidence.append({'region':i,'provider':provider,**record})
        setup[f'tilelang_region_{i}_seconds'] = time.perf_counter()-start
        triton_op = Operation('linear' if kind=='gemm' else kind,ordered,
            relu=any(str(n.target).endswith('relu') or n.target==torch.relu for n in region.nodes),
            causal=name.endswith('True'))
        replacements['triton'][region] = Reordered(triton_op.allocating_compiled,order)
        replacements['triton_jit'][region] = Reordered(triton_op,order)
        preallocated['triton'][region] = triton_op
        preallocated['triton_jit'][region] = triton_op
    functions = {'tensor_direct':direct}
    functions.update((provider,clone_forward(original,selected)) for provider,selected in replacements.items())
    # Compile and numerically check all allocating graphs before capture/timing.
    for provider,function in functions.items():
        start = time.perf_counter()
        result = function(*args)
        torch.cuda.current_stream().synchronize()
        setup[provider+'_first_execute_seconds'] = time.perf_counter()-start
        torch.testing.assert_close(result,eager(*args),atol=.01 if name.startswith('mlp') else .002,rtol=.02)
        if provider=='tilelang_matched':
            torch.testing.assert_close(result,tensor_result,atol=0,rtol=0,equal_nan=True)

    # Fixed-storage graphs bind each region's output as the next region's input.
    fixed = {}
    for provider in functions:
        env = dict(zip((n for n in original.graph.nodes if n.op=='placeholder'),args))
        calls = []
        for node in original.graph.nodes:
            if node.op=='placeholder':
                continue
            if node.op=='call_function' and isinstance(node.target,Region):
                region = node.target
                plan,order = region.static_plan
                inputs = map_arg(node.args,lambda n:env[n])
                ordered = tuple(inputs[j] for j in order)
                output = torch.empty(*plan.output_specs[0][0],dtype=plan.output_specs[0][2],device=args[0].device)
                if provider=='tensor_direct':
                    prepared = plan.kernel.prepare(*ordered,outputs=[output])
                    assert prepared._native is not None
                    calls.append(prepared)
                elif provider in ('triton','triton_jit'):
                    op = preallocated[provider][region]
                    launch = op.compiled_into if provider=='triton' else op.into
                    calls.append(lambda launch=launch,ordered=ordered,output=output:launch(ordered,output))
                else:
                    op = preallocated[provider][region]
                    calls.append(lambda op=op,ordered=ordered,output=output:op(*ordered,output))
                env[node] = output
            elif node.op=='output':
                result = map_arg(node.args[0],lambda n:env[n])
            else:
                raise ValueError(f'unexpected noncompiled graph operation: {node}')
        def execute(calls=tuple(calls),result=result):
            for call in calls:
                call()
            return result
        fixed[provider] = execute
        torch.testing.assert_close(execute(),eager(*args),atol=.01 if name.startswith('mlp') else .002,rtol=.02)
        if provider=='tilelang_matched':
            torch.testing.assert_close(execute(),tensor_result,atol=0,rtol=0,equal_nan=True)
    for region,wrapper in replacements['triton_jit'].items():
        op = wrapper.operation
        evidence.append({'provider':'triton','config':op.config,'kernel_sha256':digest(op.compiled.asm['cubin'])})
    return functions,fixed,setup,evidence


def host_samples(functions, *, completion, count, batches=9):
    stream = torch.cuda.current_stream()
    names = list(functions)
    samples = {name:[] for name in names}
    for function in functions.values():
        for _ in range(100):
            function()
    for repetition in range(batches):
        for name in names[repetition%len(names):]+names[:repetition%len(names)]:
            stream.synchronize()
            start = time.perf_counter()
            for _ in range(count):
                functions[name]()
            if completion:
                stream.synchronize()
            samples[name].append((time.perf_counter()-start)*1e6/count)
            if not completion:
                stream.synchronize()
    return {'median_us':{name:statistics.median(values) for name,values in samples.items()},'samples_us':samples}


def gpu_samples(functions,count=50,batches=9):
    stream = torch.cuda.current_stream()
    graphs = {}
    for name,function in functions.items():
        for _ in range(20):function()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            for _ in range(count):function()
        graphs[name] = graph
        for _ in range(10):graph.replay()
        stream.synchronize()
    samples = {name:[] for name in graphs}
    names = list(graphs)
    for repetition in range(batches):
        for name in names[repetition%len(names):]+names[:repetition%len(names)]:
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record();graphs[name].replay();end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*1000/count)
    return {'median_us':{name:statistics.median(values) for name,values in samples.items()},'samples_us':samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    opts = parser.parse_args()
    assert _executor is not None,'build the C++ adapter before benchmarking'
    import importlib.metadata as metadata
    report = {'versions':{name:metadata.version(name) for name in ['torch','tilelang','triton','apache-tvm-ffi']},
              'gpu':torch.cuda.get_device_name(),'cases':[],
              'methodology':{'host_batches':9,'host_submission_calls':200,'allocating_completed_calls':100,
                             'cuda_graph_calls':50,'cuda_graph_batches':9,'provider_order':'rotating',
                             'triton_tuning':'fixed configs; no autotuning; triton uses CompiledKernel runner, triton_jit uses warmed JIT dispatch',
                             'matched_tilelang':'native TVM-FFI host/runtime; Tensor exact cubin supplied through device compiler callback; ABI, entrypoint and launch verified'},
              'source_sha256':{Path(__file__).name:digest(Path(__file__).read_bytes()),
                               'direct_triton_kernels.py':digest(Path(__file__).with_name('direct_triton_kernels.py').read_bytes())}}
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    with torch.inference_mode(),torch.cuda.stream(stream):
        for name,eager,args in cases(opts.quick):
            torch._dynamo.reset()
            start = time.perf_counter()
            functions,fixed,setup,evidence = prepare_case(name,eager,args,opts.cache,opts.out.parent/'kernels'/name)
            compiled = torch.compile(eager,backend=tt.Backend(cache_dir=opts.cache),fullgraph=True,dynamic=False)
            inductor = torch.compile(eager,backend='inductor',fullgraph=True,dynamic=False)
            for function in [compiled,inductor]:
                torch.testing.assert_close(function(*args),eager(*args),atol=.01 if name.startswith('mlp') else .002,rtol=.02)
            allocating = {key:(lambda f=f:f(*args)) for key,f in functions.items()}
            allocating.update({'torch_eager':lambda:eager(*args),'torch_inductor':lambda:inductor(*args),
                               'tensor_compile':lambda:compiled(*args)})
            entry = {'name':name,'setup':setup,'evidence':evidence,
                     'allocating_host':host_samples(allocating,completion=False,count=200),
                     'allocating_completed':host_samples(allocating,completion=True,count=100),
                     'prepared_host':host_samples(fixed,completion=False,count=200),
                     'gpu_prepared':gpu_samples(fixed),
                     'gpu_allocating':gpu_samples(allocating)}
            entry['total_seconds'] = time.perf_counter()-start
            report['cases'].append(entry)
            opts.out.write_text(json.dumps(report,indent=2)+'\n')
            print(name,json.dumps(entry['allocating_completed']['median_us']),flush=True)
    report['geomean_speedup'] = {}
    for mode in ['allocating_host','allocating_completed','prepared_host','gpu_prepared','gpu_allocating']:
        keys = report['cases'][0][mode]['median_us'].keys()
        report['geomean_speedup'][mode] = {key:math.exp(statistics.mean(math.log(
            c[mode]['median_us'][key]/c[mode]['median_us']['tensor_direct']) for c in report['cases'])) for key in keys}
    opts.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['geomean_speedup'],indent=2))


if __name__=='__main__':main()
