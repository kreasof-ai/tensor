"""Test shared K/V for grouped-query CUDA decode; retain only measured gains."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import json
import ast
import hashlib
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.cuda_graph import CudaGraph
from tensor.compiler.tuning import measure_cuda
from tensor_llm.cuda_kernels import warp_partial_source, merge_source, grouped_source
from tensor_llm.kernels import emit
from benchmarks.lfm2.cuda_format_search import compile_source


def prelude(text):
    return next(n.value.value for n in ast.walk(ast.parse(text)) if isinstance(n,ast.keyword) and n.arg=='prelude')


def adaptive_source(p,threshold=6144):
    """Two ordinary heads per CTA below the threshold, shared GQA above it."""
    h,kh,d,cap,splits=(p[k] for k in ('h','kh','d','cap','splits'))
    assert (h,kh,d)==(32,8,64)
    ordinary=prelude(warp_partial_source(p,splits))
    ordinary=ordinary.replace('warp_attention','pair_attention')
    ordinary=ordinary.replace('maxima[4], sums[4], values[4][64]','maxima[8], sums[8], values[8][64]')
    ordinary=ordinary.replace('head=blockIdx.x,','head=blockIdx.x*2+warp/4,')
    ordinary=ordinary.replace('token=begin+warp;','token=begin+warp%4;').replace('if(warp==0)','if(warp%4==0)')
    for i in range(4):ordinary=ordinary.replace(f'maxima[{i}]',f'maxima[warp/4*4+{i}]')
    ordinary=ordinary.replace('maxima[i]','maxima[warp/4*4+i]').replace('sums[i]','sums[warp/4*4+i]').replace('values[i]','values[warp/4*4+i]')
    cpp=ordinary+'\n'+prelude(grouped_source(p,32,2))+f'''
__device__ __forceinline__ void adaptive_attention(const float* q,const void* k,const void* v,float* parts,const int* pos) {{
    if(pos[0]+1 >= {threshold}) {{
        if(blockIdx.x < {kh})grouped_attention(q,k,v,parts,pos);
    }} else pair_attention(q,k,v,parts,pos);
}}
'''
    return emit([('q',h*d,'float32'),('kc',cap*kh*d,'float16'),('vc',cap*kh*d,'float16'),
                 ('parts',h*splits*66,'float32'),('pos',2,'int32')],f'''with T.Kernel({h//2}, {splits}, threads=256,prelude={cpp!r}) as (head, split):
    T.evaluate(T.call_extern("void", "adaptive_attention", T.address_of(q[0]), T.address_of(kc[0]), T.address_of(vc[0]), T.address_of(parts[0]), T.address_of(pos[0])))''')




def search(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=dict(h=32,kh=8,d=64,cap=8576,splits=16)
    variants={'before':(None,warp_partial_source(p,16))}
    variants['adaptive']=(dict(threshold=6144,stage=32,warps=2),adaptive_source(p))
    for stage in (16,32,64):
        for warps in (2,4):variants[f'{stage}-{warps}']=(dict(stage=stage,warps=warps),grouped_source(p,stage,warps))
    rng=np.random.default_rng(2243);q=rng.normal(size=(32,64)).astype(np.float32)
    k=rng.normal(size=(8576,8,64)).astype(np.float16);v=rng.normal(size=k.shape).astype(np.float16)
    paths={name:compile_source(text,out/'kernels') for name,(_,text) in variants.items()}
    mergepath=compile_source(merge_source(p,16),out/'kernels');observations=[]
    with tensor.Device() as device:
        dq,dk,dv=[device.from_numpy(x.ravel()) for x in (q,k,v)]
        parts=device.empty(32*16*66);result=device.empty(32*64);control=device.zeros(2,dtype='int32')
        merge=device.load(mergepath);kernels={name:device.load(path) for name,path in paths.items()}
        for length in (1,7,128,512,2048,4096,6143,6144,8192):
            host=np.array([length-1,1],np.int32)
            device.driver.call('cuMemcpyHtoD_v2',control.pointer,host.ctypes.data,host.nbytes)
            expected=[]
            for head in range(32):
                scores=k[:length,head//4].astype(np.float64)@q[head].astype(np.float64)/8
                probs=np.exp(scores-scores.max());probs/=probs.sum()
                expected.append(probs@v[:length,head//4].astype(np.float64))
            expected=np.asarray(expected).ravel();row=dict(length=length,candidates={})
            for name,kernel in kernels.items():
                def launch():kernel.launch(dq,dk,dv,parts,control);merge.launch(parts,result)
                launch();actual=result.to_numpy();np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=2e-6)
                def batch():
                    for _ in range(20):launch()
                with CudaGraph(device,batch,resources=(dq,dk,dv,parts,result,control,kernel,merge)) as graph:
                    timing=measure_cuda(device,graph.launch,warmup=3,samples=5,repeats=1)
                for key in ('gpu_seconds','completed_seconds'):timing[key]=[x/20 for x in timing[key]]
                for key in ('median_gpu_seconds','median_completed_seconds'):timing[key]/=20
                row['candidates'][name]=dict(config=variants[name][0],timing=timing,
                                            source_sha256=hashlib.sha256(variants[name][1].encode()).hexdigest())
                print(length,name,round(timing['median_gpu_seconds']*1e6,2),'us',flush=True)
            observations.append(row)
        report=dict(schema='tensor.lfm2-cuda-format-search.v1',adapter=device.info,cases=observations,
                    protocol=dict(seed=2243,gpu_concurrency=1,warmups=3,samples=5,graph_repetitions=20,times_divided_by=20,
                                  baseline='generic packed schedule, not the frozen full-model before profile',
                                  reference='independent NumPy float64 softmax',rtol=3e-5,atol=2e-6))
        (out/'search.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':search('build/lfm2-cuda-formats/attention-search')
