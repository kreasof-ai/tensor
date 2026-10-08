"""LLT CUDA bottleneck attribution and isolated, correctness-gated GEMM search."""
import argparse
import hashlib
import json
import os
import sys
import time
import statistics
from pathlib import Path
import torch
from tensor_torch.llt import Operators
from tensor.compiler.search import ScheduleSearch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/research/data/llt-optimization'
BUILD=ROOT/'build/llt-optimization'


def save(name, data):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def model_fixture(ops, width=768, layers=12, loops=1, vocab=256, sequence=1024):
    llt=Path(os.environ.get('LLT_CHECKOUT',ROOT.parent/'loop-latent-transformer'))
    sys.path.insert(0,str(llt/'experiments/l40s'))
    from model import Config
    from tensor_model import BackendTransformer
    torch.manual_seed(9501)
    c=Config(width=width,heads=width//64,layers=layers,loops=loops,rank=64,
             vocab=vocab,max_seq=sequence+4,gelu='none')
    model=BackendTransformer(c,ops).cuda().eval()
    return model,torch.randint(vocab,(1,sequence+1),device='cuda')


def profile(a):
    ops=Operators(ROOT/'build/llt-qualification/artifacts')
    model,tokens=model_fixture(ops)
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        # Same prepared serving weights as the LLT study.
        for block in model.blocks:block.to(dtype=torch.bfloat16)
        model.down.to(dtype=torch.bfloat16);model.output.to(dtype=torch.bfloat16)
        _,state=model.prefill(tokens[:,:-1])
        def decode():
            model.rewind(state,tokens.shape[1]-1)
            return model.decode_token(tokens[:,-1:],state)
        for _ in range(3):decode()
        torch.cuda.synchronize()
        original=ops.call
        def traced(factory,parameters,outputs,*args,**kwargs):
            label=parameters.get('kind',factory) if isinstance(parameters,dict) else factory
            shape=(','.join(f'{k}={parameters[k]}' for k in ('m','k','c') if k in parameters)
                   if isinstance(parameters,dict) else '')
            with torch.profiler.record_function('tensor:'+label+(':'+shape if shape else '')):
                return original(factory,parameters,outputs,*args,**kwargs)
        ops.call=traced
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
            decode()
        ops.call=original
        BUILD.mkdir(parents=True,exist_ok=True)
        p.export_chrome_trace(str(BUILD/'baseline-decode-trace.json'))
        rows=[]
        for e in p.key_averages():
            if e.key.startswith('tensor:'):
                rows.append(dict(operation=e.key,count=e.count,cpu_total_us=e.cpu_time_total,
                                 device_total_us=e.device_time_total,self_device_total_us=e.self_device_time_total))
        rows.sort(key=lambda x:x['device_total_us'],reverse=True)
        save('baseline-profile',dict(status='passed',config=vars(model.c),operators=rows,
            protocol='CUPTI attribution; profiler overhead excluded from benchmark comparisons'))
        print(json.dumps(rows[:12],indent=2))


def graph_measure(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):result=fn()
    times=[]
    for _ in range(9):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):graph.replay()
        end.record();end.synchronize()
        times.append(start.elapsed_time(end)/20)
    del graph,result
    torch._C._cuda_clearCublasWorkspaces()
    return dict(samples_ms=times,median_ms=statistics.median(times))


def search(a):
    ops=Operators(BUILD/'artifacts')
    rows=[]
    # Decode, forward/prefill, dX, dW, and vocabulary projection kernels.
    cases=[(1,3072,768,False,True),(1,768,3072,False,True),
           (1024,768,3072,False,True),(1024,3072,768,False,False),
           (3072,1024,768,True,False),(1024,768,50304,False,True)]
    if a.case is not None:cases=[cases[a.case]]
    with torch.inference_mode():
        for m,k,n,ta,tb in cases:
            torch.manual_seed(9502)
            x=torch.randn((k,m) if ta else (m,k),device='cuda',dtype=torch.bfloat16)*.05
            w=torch.randn((n,k) if tb else (k,n),device='cuda',dtype=torch.bfloat16)*.05
            expected=(x.float().T if ta else x.float())@(w.float().T if tb else w.float())
            p=dict(kind='gemm',m=m,k=k,c=n,ta=ta,tb=tb,dtype='bfloat16',out_dtype='bfloat16')
            baseline=lambda:ops.training(p,['out'],x,w)
            torch.testing.assert_close(baseline().float(),expected,atol=.005,rtol=.035)
            row=dict(parameters=p,baseline=graph_measure(baseline),candidates=[])
            spaces={'mma':dict(bm=(16,32,64,128),bn=(32,64,128),bk=(32,64,128),threads=(128,256),stages=(1,2,3))}
            seeds=[dict(family='mma',bm=32,bn=64,bk=32,threads=128,stages=2),
                   dict(family='mma',bm=64,bn=128,bk=64,threads=128,stages=3)]
            if m<=8 and tb and not ta:
                spaces['gemv']=dict(rows=(1,2,4,8,16),chunk=(64,128,256,512),threads=(64,128,256))
                seeds=[dict(family='gemv',rows=4,chunk=128,threads=128),
                       dict(family='gemv',rows=8,chunk=256,threads=128)]+seeds
            def legal(s):
                if s['family']=='gemv':return s['rows']*s['chunk']>=s['threads']
                return (s['bm']+s['bn'])*s['bk']*2*s['stages']<=96*1024
            engine=ScheduleSearch(seeds,spaces=spaces,width=4,legal=legal)
            started=time.perf_counter()
            for trial in range(a.candidates):
                if time.perf_counter()-started>a.seconds:break
                try:s=engine.next()
                except StopIteration:break
                candidate=dict(schedule=s)
                try:
                    params={**p,'schedule':s}
                    fn=lambda:ops.call('make_kernel',params,['out'],x,w,module='tensor_torch.templates.llt_gemm')
                    actual=fn()
                    torch.testing.assert_close(actual.float(),expected,atol=.005,rtol=.035)
                    candidate.update(status='passed',timing=graph_measure(fn),max_error=(actual.float()-expected).abs().max().item())
                    engine.record(s,candidate['timing']['median_ms'])
                    print('Search',m,k,n,trial,s,round(candidate['timing']['median_ms']*1000,2),'us',flush=True)
                    del actual
                except Exception as error:
                    candidate.update(status='rejected',reason=str(error)[-2500:])
                    print('Rejected',m,k,n,trial,s,str(error)[-120:],flush=True)
                row['candidates'].append(candidate)
                save('search-'+str(m)+'-'+str(k)+'-'+str(n),dict(status='running',case=row))
            passed=[c for c in row['candidates'] if c['status']=='passed']
            assert passed,'no valid candidate'
            winner=min(passed,key=lambda c:c['timing']['median_ms'])
            fn=lambda:ops.call('make_kernel',{**p,'schedule':winner['schedule']},['out'],x,w,module='tensor_torch.templates.llt_gemm')
            row.update(winner=winner['schedule'],winner_recheck=graph_measure(fn),elapsed_seconds=time.perf_counter()-started)
            row['speedup']=row['baseline']['median_ms']/row['winner_recheck']['median_ms']
            rows.append(row)
            save('search-'+str(m)+'-'+str(k)+'-'+str(n),dict(status='passed',case=row))
            print('Winner',m,k,n,row['winner'],'speedup',row['speedup'],flush=True)
    save('search-summary' if a.case is None else 'search-summary-'+str(a.case),dict(status='passed',cases=rows,coverage=ops.report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('profile','search'))
    parser.add_argument('--candidates',type=int,default=20)
    parser.add_argument('--seconds',type=float,default=240)
    parser.add_argument('--case',type=int)
    a=parser.parse_args()
    globals()[a.phase](a)
