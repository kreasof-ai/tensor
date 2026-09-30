"""Actual FX-to-NVRTC inference performance, with honest eager/Inductor baselines."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch
import tensor_torch as tt


def end_to_end(function, args, count=100):
    samples=[]
    for _ in range(7):
        torch.cuda.synchronize()
        started=time.perf_counter()
        for _ in range(count):
            function(*args)
        torch.cuda.synchronize()
        samples.append((time.perf_counter()-started)*1e6/count)
    return statistics.median(samples)


def gpu(function,args,count=50):
    stream=torch.cuda.current_stream()
    for _ in range(10):function(*args)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(count):
            output=function(*args)
    for _ in range(10):graph.replay()
    times=[]
    for _ in range(7):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record();graph.replay();end.record();end.synchronize()
        times.append(start.elapsed_time(end)*1000/count)
    return statistics.median(times)


def cases(quick):
    rand=lambda shape,dtype=torch.float16:torch.randn(shape,device='cuda',dtype=dtype)
    for size in (129,1048576) if quick else (129,257,1048576,4194304):
        yield f'pointwise-{size}',lambda a,b:torch.relu(a*2+b),(rand((size,),torch.float32),rand((size,),torch.float32))
    for m,n,k in ([(33,65,64)] if quick else [(33,65,64),(128,128,128),(512,512,512)]):
        yield f'gemm-{m}-{n}-{k}',lambda a,w,b:torch.relu(torch.nn.functional.linear(a,w,b)),(rand((m,k)),rand((n,k)),rand((n,)))
    yield 'mlp-128-256-128',lambda a,w1,b1,w2,b2:torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(a,w1,b1)),w2,b2),(rand((128,128)),rand((256,128)),rand((256,)),rand((128,256)),rand((128,)))
    for shape in ([(1,8,129,64)] if quick else [(1,8,128,64),(1,8,129,64),(2,4,257,64),(1,8,512,64),(1,8,1024,64),(1,8,512,128)]):
        for causal in (False,True):
            def attention(q,k,v,causal=causal):
                return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=causal)
            yield 'sdpa-'+'-'.join(map(str,shape))+'-'+str(causal),attention,tuple(rand(shape) for _ in range(3))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--quick',action='store_true')
    opts=parser.parse_args()
    torch.manual_seed(42)
    torch.cuda.init()
    stream=torch.cuda.Stream()
    from tensor_torch.bridge import _submit
    report={'native_submission':_submit is not None,'torch':torch.__version__,'device':torch.cuda.get_device_name(),'cases':[],
            'timing':'median 7 batches; warm end-to-end includes Python submission, output allocation and stream completion; GPU times use 50-call CUDA graph replay',
            'targets':{'max_geomean_inductor_ratio':1.10,'max_case_inductor_ratio':1.25,'selected_eager_speedup':1.25,'prepared_submission_us':15,'cached_prepare_seconds':.1,'cold_pointwise_seconds':10,'cold_gemm_seconds':30}}
    package=Path(tt.__file__).parent
    report['implementation_sha256']={str(p.relative_to(package)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(package.rglob('*')) if p.is_file() and p.suffix in {'.py','.c','.so','.pyd'}}
    report['benchmark_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode(),torch.cuda.stream(stream):
        for name,eager,args in cases(opts.quick):
            torch._dynamo.reset()
            backend=tt.Backend(cache_dir=opts.cache)
            function=torch.compile(eager,backend=backend,fullgraph=True,dynamic=False)
            started=time.perf_counter();result=function(*args);stream.synchronize()
            cold=time.perf_counter()-started
            torch.testing.assert_close(result,eager(*args),atol=.01 if name.startswith('mlp') else .002,rtol=.02)
            inductor=torch.compile(eager,backend='inductor',fullgraph=True,dynamic=False)
            torch.testing.assert_close(inductor(*args),eager(*args),atol=.01 if name.startswith('mlp') else .002,rtol=.02)
            # Interleave providers to reduce drift from clocks and warm allocator state.
            functions={'tensor':function,'eager':eager,'inductor':inductor}
            for f in functions.values():
                for _ in range(20):f(*args)
            end={key:end_to_end(f,args) for key,f in functions.items()}
            device={key:gpu(f,args) for key,f in functions.items()}
            cached=tt.Backend(cache_dir=opts.cache)
            warm=torch.compile(eager,backend=cached,fullgraph=True,dynamic=False)
            started=time.perf_counter();warm(*args);stream.synchronize();warm_prepare=time.perf_counter()-started
            assert all(s['cache_hit'] for r in cached.report['regions'] for s in r['specializations'])
            entry={'name':name,'cold_graph_seconds':cold,'cached_graph_seconds':warm_prepare,
                   'end_to_end_us':end,'gpu_us':device,'tensor_over_inductor':end['tensor']/end['inductor'],
                   'eager_speedup':end['eager']/end['tensor'],'report':backend.report}
            if name.startswith('sdpa'):
                from torch.nn.attention import sdpa_kernel,SDPBackend
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    entry['forced_flash']={'end_to_end_us':end_to_end(eager,args),'gpu_us':gpu(eager,args)}
            report['cases'].append(entry)
            opts.out.write_text(json.dumps(report,indent=2)+'\n')
            print(name, json.dumps(end), 'ratio',round(entry['tensor_over_inductor'],3),flush=True)
        # Prepared submission: fixed .tbin arguments, allocations outside timing.
        pointwise=next(c for c in report['cases'] if c['name'].startswith('pointwise'))
        kernel=tt.load(pointwise['report']['regions'][0]['specializations'][0]['artifact'])
        a,b=(torch.randn(129,device='cuda') for _ in range(2))
        prepared=kernel.prepare(a,b)
        samples=[]
        for _ in range(7):
            stream.synchronize();start=time.perf_counter()
            for _ in range(1000):prepared()
            samples.append((time.perf_counter()-start)*1000)
            stream.synchronize()
        report['prepared_submission_us']=statistics.median(samples)
    ratios=[c['tensor_over_inductor'] for c in report['cases']]
    report['geomean_inductor_ratio']=math.exp(statistics.mean(map(math.log,ratios)))
    report['max_inductor_ratio']=max(ratios)
    report['acceptance']={'geomean':report['geomean_inductor_ratio']<=1.10,'worst_case':max(ratios)<=1.25,
                          'selected_eager_speedup':any(c['eager_speedup']>=1.25 for c in report['cases']),
                          'prepared_submission':report['prepared_submission_us']<=15,
                          'cached_prepare':all(c['cached_graph_seconds']<=.1 for c in report['cases']),
                          'cold_pointwise':all(c['cold_graph_seconds']<=10 for c in report['cases'] if c['name'].startswith('pointwise')),
                          'cold_gemm':all(c['cold_graph_seconds']<=30 for c in report['cases'] if c['name'].startswith(('gemm','mlp')))}
    opts.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'cases','timing'}},indent=2))


if __name__=='__main__':main()
