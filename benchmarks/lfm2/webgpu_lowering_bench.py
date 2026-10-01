"""Compare compiler GEMM microtiles and ordered/tree reductions on native WGSL."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.runtime.abi import BoundCall


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);records=[]
    root=Path(__file__).resolve().parents[2];rng=np.random.default_rng(817)
    for micro in (1,2,4):
        text=(root/'examples/webgpu_gemm.py').read_text().replace('M = 33','M = 128').replace('N = 65','N = 256').replace('K = 37','K = 512')
        text=text.replace('OUTPUT_DTYPE = "float16"','OUTPUT_DTYPE = "float32"').replace('USE_BIAS = True','USE_BIAS = False').replace('RELU = True','RELU = False')
        text=text.replace('kernel = linear',f'kernel = linear.with_attr("tensor.webgpu.gemm_microtile", {micro})')
        a=rng.normal(size=(128,512)).astype(np.float16);b=rng.normal(size=(512,256)).astype(np.float16)
        records.append(measure(out,f'gemm-{micro}',text,[a,b],a.astype(np.float32)@b.astype(np.float32)))
    for mode in ('ordered','tree'):
        text=f'''import tilelang.language as T
@T.prim_func
def kernel(x:T.Tensor((64,1024),"float32"),out:T.Tensor((64,),"float32")):
    T.func_attr({{"tensor.webgpu.reduction":"{mode}"}})
    with T.Kernel(64,threads=128) as block:
        square=T.alloc_shared((1024,),"float32")
        total=T.alloc_shared((1,),"float32")
        for i in T.Parallel(1024):square[i]=x[block,i]*x[block,i]
        T.reduce_sum(square,total,dim=0)
        out[block]=total[0]
def tensor_export():return {{"kernel":kernel,"outputs":["out"]}}
'''
        x=rng.normal(size=(64,1024)).astype(np.float32)
        records.append(measure(out,f'reduce-{mode}',text,[x],np.sum(x.astype(np.float64)**2,axis=1)))
    (out/'report.json').write_text(json.dumps({'protocol':'five medians after warmup, ten prepared launches and completion per sample','records':records},indent=2))


def measure(out,name,text,inputs,expected):
    path=out/(name+'.py');artifact=path.with_suffix('.tbin');path.write_text(text);artifact.unlink(missing_ok=True)
    tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache')
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);args=[device.from_numpy(x) for x in inputs];output=device.zeros(expected.shape)
        values,symbols,launch=kernel._bind((*args,output),{},include_outputs=True)
        plan=device.prepare_plan([(kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True))])
        plan.launch();np.testing.assert_allclose(output.to_numpy(),expected,rtol=3e-4,atol=1e-4)
        samples=[]
        for repeat in range(6):
            start=time.perf_counter()
            for _ in range(10):plan.launch()
            output.to_numpy();elapsed=(time.perf_counter()-start)/10
            if repeat:samples.append(elapsed)
        row={'case':name,'median_seconds':statistics.median(samples),'samples_seconds':samples};print(row,flush=True);return row


if __name__=='__main__':run('build/webgpu-compiler-lowering-bench')
