"""Larger-shape latency scaling using the direct backend benchmark controls.

Keeps warmed host submission, amortized completed batches, serialized calls,
and captured GPU execution separate. Native TileLang requires a toolkit.
"""
from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import argparse
from contextlib import ExitStack
import importlib.metadata as metadata
import json
import statistics
import time
from pathlib import Path

import torch
import tensor_torch as tt
from tensor_torch.bridge import _executor
from benchmarks.inference.direct_backend_benchmark import digest, prepare_case, host_samples, gpu_samples


def cases():
    def rand(shape, dtype=torch.float16):
        return torch.randn(shape, device='cuda', dtype=dtype)

    for size in (129, 1048576, 4194304, 16777216, 67108864):
        yield (f'pointwise-{size}', lambda a,b:torch.relu(a*2+b),
               (rand((size,),torch.float32), rand((size,),torch.float32)))
    for size in (512, 1024, 2048, 4096):
        yield (f'gemm-{size}-{size}-{size}',
               lambda a,w,b:torch.relu(torch.nn.functional.linear(a,w,b)),
               (rand((size,size)), rand((size,size)), rand((size,))))
    for length in (1024, 2048, 4096, 8192):
        shape = (1,8,length,64)
        for causal in (False, True):
            def attention(q,k,v,causal=causal):
                return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
            yield ('sdpa-'+'-'.join(map(str,shape))+'-'+str(causal), attention,
                   tuple(rand(shape) for _ in range(3)))


def serialized_samples(functions, batches=9, calls=5, *, synchronizers=None, disposers=None):
    """Wall time from entering the callable until its output is ready.

Synchronize after every call; output destruction is outside its timer.
Rotate providers and retain all individual observations, not batch averages.
"""
    stream = torch.cuda.current_stream()
    synchronizers, disposers = synchronizers or {}, disposers or {}
    names = list(functions)
    samples = {name:[] for name in names}
    for name,function in functions.items():
        for _ in range(20):
            output = function()
            if name in synchronizers:
                synchronizers[name]()
            if name in disposers:
                disposers[name](output)
            del output
    stream.synchronize()
    for repetition in range(batches):
        for name in names[repetition%len(names):]+names[:repetition%len(names)]:
            stream.synchronize()
            synchronize = synchronizers.get(name,stream.synchronize)
            synchronize()
            for _ in range(calls):
                start = time.perf_counter()
                output = functions[name]()
                synchronize()
                samples[name].append((time.perf_counter()-start)*1e6)
                if name in disposers:
                    disposers[name](output)
                del output
    return {'median_us':{name:statistics.median(values) for name,values in samples.items()},
            'samples_us':samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--case', action='append', help='Run only these exact case names')
    parser.add_argument('--webgpu', action='store_true', help='Compare identical workloads on the same physical GPU through wgpu')
    parser.add_argument('--webgpu-device', type=int, default=0)
    opts = parser.parse_args()
    assert _executor is not None, 'build the C++ adapter before benchmarking'
    selection = set(opts.case or [])
    remaining = set(selection)
    report = {
        'versions':{name:metadata.version(name) for name in ('torch','tilelang','triton','apache-tvm-ffi')},
        'gpu':torch.cuda.get_device_name(), 'cases':[],
        'methodology':{
            'host_batches':9, 'host_calls_per_batch':20, 'host_warmup_calls':100,
            'serialized_batches':9, 'serialized_calls_per_provider_per_batch':5,
            'serialized_warmup_calls':20,
            'cuda_graph_calls':10, 'cuda_graph_batches':9,
            'provider_order':'rotating', 'seed':42,
            'triton':'fixed schedules, no autotuning; compiled runner and warmed JIT dispatch',
            'tilelang_matched':'native TVM-FFI runtime with exact Tensor cubin; ABI and launch checked',
            'serialized':'wall time for one allocating call plus stream synchronization; output destruction outside timer',
            'batched':'20 queued allocating calls plus one synchronization, divided by 20',
        },
        'source_sha256':{name:digest(Path(__file__).with_name(name).read_bytes())
                         for name in ('scaling_backend_benchmark.py','direct_backend_benchmark.py',
                                      'direct_triton_kernels.py','fx_producer.py')},
    }
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    with ExitStack() as resources, torch.inference_mode(), torch.cuda.stream(stream):
        webgpu = None
        if opts.webgpu:
            import tensor as tx
            import wgpu
            from wgpu.backends import wgpu_native
            from benchmarks.inference.webgpu_scaling import WebGPUCase
            webgpu = resources.enter_context(tx.Device(opts.webgpu_device,provider='webgpu',max_buffer_size=268435456))
            if (webgpu.info['adapter']['adapter_type'] != 'DiscreteGPU' or
                    webgpu.info['name'] != torch.cuda.get_device_name()):
                raise ValueError(f'WebGPU must select the same physical GPU as CUDA: {webgpu.info}')
            report['webgpu'] = {**webgpu.info, 'software_adapter':False,
                                'versions':{'wgpu':wgpu.__version__,'wgpu-native':wgpu_native.__version__}}
            report['methodology']['webgpu'] = {
                'operation_shapes_dtypes':'identical to CUDA; upload copies of the same input tensors',
                'timing':'allocating call plus queue completion; output release outside timer',
                'excluded':'uploads/downloads, AOT compilation and cold pipeline creation',
                'scope':'serialized comparison only; CUDA host batches and CUDA graphs retain CUDA providers',
                'gemm_schedule':'32x32 output tile, K=16, scalar FP32 accumulation',
                'attention_schedule':'online softmax, query/key tiles 8x16, FP32 accumulators',
                'max_buffer_size':268435456,
            }
            report['versions'].update(report['webgpu']['versions'])
            report['source_sha256']['webgpu_scaling.py'] = digest(Path(__file__).with_name('webgpu_scaling.py').read_bytes())
            print('WebGPU physical adapter',json.dumps(report['webgpu']),flush=True)
        for name,eager,args in cases():
            if selection and name not in selection:
                continue
            remaining.discard(name)
            torch._dynamo.reset()
            started = time.perf_counter()
            functions,fixed,setup,evidence = prepare_case(
                name,eager,args,opts.cache,opts.out.parent/'kernels'/name)
            backend = tt.Backend(cache_dir=opts.cache)
            compiled = torch.compile(eager,backend=backend,fullgraph=True,dynamic=False)
            inductor = torch.compile(eager,backend='inductor',fullgraph=True,dynamic=False)
            for function in (compiled,inductor):
                torch.testing.assert_close(function(*args),eager(*args),atol=.002,rtol=.02)
            allocating = {key:(lambda f=f:f(*args)) for key,f in functions.items()}
            allocating.update({'torch_eager':lambda:eager(*args),
                               'torch_inductor':lambda:inductor(*args),
                               'tensor_compile':lambda:compiled(*args)})
            gpu_case = None
            if webgpu is not None:
                gpu_case = WebGPUCase(webgpu,name,args,eager(*args),opts.out.parent/'wgpu'/name,opts.cache/'webgpu')
                evidence.append(gpu_case.evidence)
            serialized = {**allocating, **({'webgpu':gpu_case} if gpu_case is not None else {})}
            entry = {
                'name':name, 'inputs':[{'shape':list(a.shape),'dtype':str(a.dtype)} for a in args],
                'setup':setup, 'evidence':evidence, 'tensor_compile_report':backend.report,
                'allocating_host':host_samples(allocating,completion=False,count=20),
                'allocating_completed':host_samples(allocating,completion=True,count=20),
                'serialized':serialized_samples(serialized,
                    synchronizers={'webgpu':webgpu.synchronize} if webgpu is not None else None,
                    disposers={'webgpu':lambda output:output.release()} if webgpu is not None else None),
                'gpu_allocating':gpu_samples(allocating,count=10),
            }
            entry['total_seconds'] = time.perf_counter()-started
            report['cases'].append(entry)
            opts.out.write_text(json.dumps(report,indent=2)+'\n')
            print(name,json.dumps({mode:entry[mode]['median_us']
                                  for mode in ('serialized','allocating_completed','gpu_allocating')}),flush=True)
            if gpu_case is not None:
                gpu_case.close()
            # Drop fixed-output graphs before moving to the next large profile.
            del fixed,functions,allocating,serialized,gpu_case,compiled,inductor,backend,args
            torch.cuda.empty_cache()
    if remaining:
        raise ValueError(f'unknown cases: {sorted(remaining)}')


if __name__=='__main__':
    main()
