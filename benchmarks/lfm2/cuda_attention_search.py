"""Discover CUDA decode attention schedules against independent float64 math."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensor
from tensor.compiler.search import ScheduleSearch
from tensor.compiler.cuda_schedules import ATTENTION_SPACES
from tensor.providers.cuda_graph import CudaGraph
from tensor.compiler.tuning import measure_cuda
from tensor_llm.cuda_kernels import warp_partial_source, merge_source, grouped_source
from benchmarks.lfm2.cuda_format_search import compile_source


def search(out, max_candidates=7):
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError('positive candidate budget required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[2]
    sources={name:dict(sha256=hashlib.sha256((repo/name).read_bytes()).hexdigest(),text=(repo/name).read_text()) for name in ('benchmarks/lfm2/cuda_attention_search.py', 'src/tensor/compiler/search.py', 'src/tensor/compiler/cuda_schedules.py', 'src/tensor/compiler/cuda_lowering.py', 'packages/tensor-llm/src/tensor_llm/cuda_kernels.py')}
    p=dict(h=32,kh=8,d=64,cap=8576,splits=16)
    rng=np.random.default_rng(2243);q=rng.normal(size=(32,64)).astype(np.float32)
    k=rng.normal(size=(8576,8,64)).astype(np.float16);v=rng.normal(size=k.shape).astype(np.float16)
    mergepath=compile_source(merge_source(p,16),out/'kernels');observations=[]
    with tensor.Device() as device:
        dq,dk,dv=[device.from_numpy(x.ravel()) for x in (q,k,v)]
        parts=device.empty(32*16*66);result=device.empty(32*64);control=device.zeros(2,dtype='int32')
        merge=device.load(mergepath)
        for length in (1,7,128,512,2048,4096,6143,6144,8192):
            discovery=ScheduleSearch([dict(family='partial',splits=16),dict(family='grouped',stage=32,warps=2)],
                                     spaces=ATTENTION_SPACES,width=4)
            host=np.array([length-1,1],np.int32)
            device.driver.call('cuMemcpyHtoD_v2',control.pointer,host.ctypes.data,host.nbytes)
            expected=[]
            for head in range(32):
                scores=k[:length,head//4].astype(np.float64)@q[head].astype(np.float64)/8
                probs=np.exp(scores-scores.max());probs/=probs.sum()
                expected.append(probs@v[:length,head//4].astype(np.float64))
            expected=np.asarray(expected).ravel();row=dict(length=length,candidates={})
            for index in range(max_candidates):
                try:config=discovery.next()
                except StopIteration:break
                text=(warp_partial_source(p,config['splits']) if config['family']=='partial'
                      else grouped_source(p,config['stage'],config['warps']))
                kernel=None;name=f'candidate-{index}'
                try:
                    kernel=device.load(compile_source(text,out/'kernels'))
                    def launch():kernel.launch(dq,dk,dv,parts,control);merge.launch(parts,result)
                    launch();actual=result.to_numpy();np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=2e-6)
                    def batch():
                        for _ in range(20):launch()
                    with CudaGraph(device,batch,resources=(dq,dk,dv,parts,result,control,kernel,merge)) as graph:
                        timing=measure_cuda(device,graph.launch,warmup=3,samples=5,repeats=1)
                    for metric in ('gpu_seconds','completed_seconds'):timing[metric]=[x/20 for x in timing[metric]]
                    for metric in ('median_gpu_seconds','median_completed_seconds'):timing[metric]/=20
                    row['candidates'][name]=dict(status='passed',config=config,timing=timing,
                                                source_sha256=hashlib.sha256(text.encode()).hexdigest())
                    discovery.record(config,timing['median_gpu_seconds'])
                    print(length,config,round(timing['median_gpu_seconds']*1e6,2),'us',flush=True)
                except Exception as exc:
                    row['candidates'][name]=dict(status='rejected',config=config,reason=str(exc))
                    print(length,config,'rejected',str(exc)[:160],flush=True)
                finally:
                    if kernel is not None:kernel.release()
            passed=[name for name,value in row['candidates'].items() if value['status']=='passed']
            assert passed,row
            row['selected']=min(passed,key=lambda name:row['candidates'][name]['timing']['median_gpu_seconds'])
            observations.append(row)
            report=dict(schema='tensor.lfm2-cuda-attention-search.v1',adapter=device.info,cases=observations,sources=sources,
                        protocol=dict(seed=2243,gpu_concurrency=1,warmups=3,samples=5,graph_repetitions=20,times_divided_by=20,
                                      discovery='tensor.compiler.search.ScheduleSearch',max_candidates_per_shape=max_candidates,
                                      reference='independent NumPy float64 softmax',rtol=3e-5,atol=2e-6))
            (out/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('build/lfm2-compiler-cleanup/attention-search'))
    p.add_argument('--candidates',type=int,default=7)
    args=p.parse_args();search(args.out,args.candidates)
