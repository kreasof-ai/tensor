"""Larger-shape latency scaling using the direct backend benchmark controls.

Keeps warmed host submission, amortized completed batches, serialized calls,
and captured GPU execution separate. Native TileLang requires a toolkit.
"""
from __future__ import annotations

import argparse
import importlib.metadata as metadata
import json
import statistics
import time
from pathlib import Path

import torch
import tensor_torch as tt
from tensor_torch.bridge import _executor
from direct_backend_benchmark import digest, prepare_case, host_samples, gpu_samples


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


def serialized_samples(functions, batches=9, calls=5):
    """Wall time from entering the callable until its output is ready.

Synchronize after every call; output destruction is outside its timer.
Rotate providers and retain all individual observations, not batch averages.
"""
    stream = torch.cuda.current_stream()
    names = list(functions)
    samples = {name:[] for name in names}
    for function in functions.values():
        for _ in range(20):
            function()
    stream.synchronize()
    for repetition in range(batches):
        for name in names[repetition%len(names):]+names[:repetition%len(names)]:
            stream.synchronize()
            for _ in range(calls):
                start = time.perf_counter()
                output = functions[name]()
                stream.synchronize()
                samples[name].append((time.perf_counter()-start)*1e6)
                del output
    return {'median_us':{name:statistics.median(values) for name,values in samples.items()},
            'samples_us':samples}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--case', action='append', help='Run only these exact case names')
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
                                      'direct_triton_kernels.py','phase4_producer.py')},
    }
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
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
            entry = {
                'name':name, 'inputs':[{'shape':list(a.shape),'dtype':str(a.dtype)} for a in args],
                'setup':setup, 'evidence':evidence, 'tensor_compile_report':backend.report,
                'allocating_host':host_samples(allocating,completion=False,count=20),
                'allocating_completed':host_samples(allocating,completion=True,count=20),
                'serialized':serialized_samples(allocating),
                'gpu_allocating':gpu_samples(allocating,count=10),
            }
            entry['total_seconds'] = time.perf_counter()-started
            report['cases'].append(entry)
            opts.out.write_text(json.dumps(report,indent=2)+'\n')
            print(name,json.dumps({mode:entry[mode]['median_us']
                                  for mode in ('serialized','allocating_completed','gpu_allocating')}),flush=True)
            # Drop fixed-output graphs before moving to the next large profile.
            del fixed,functions,allocating,compiled,inductor,backend,args
            torch.cuda.empty_cache()
    if remaining:
        raise ValueError(f'unknown cases: {sorted(remaining)}')


if __name__=='__main__':
    main()
