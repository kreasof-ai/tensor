"""Measure packed decode schedules while streaming all layers' actual weights.

Each prepared plan visits every matching layer once. This avoids choosing a
schedule from a single repeatedly cached matrix that fits in GPU cache.
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.runtime.abi import BoundCall
from tensor_llm import GGUF
from tensor_llm.common.gguf import dequantize
from tensor_llm.lfm2.kernels.webgpu import source


BASELINE={'gemv_lanes':16,'gemv_threads':128,'gemv_dot':False}
CONFIGS=[BASELINE]+[{'gemv_lanes':lanes,'gemv_threads':threads,'gemv_dot':False}
              for lanes,threads in ((8,64),(8,128),(8,256),(16,64),(16,256),(32,64),(32,128),(32,256))]
CONFIGS += [dict(gemv_lanes=lanes,gemv_threads=threads,gemv_accumulators=4,gemv_dot=False)
            for lanes,threads in ((8,128),(16,128),(32,128),(32,256))]
DOTS=[BASELINE, {'gemv_lanes':32,'gemv_dot':False}]+[dict(gemv_lanes=lanes,gemv_threads=threads,gemv_dot=True)
                                    for lanes,threads in ((16,128),(32,64),(32,128),(32,256))]


def run(model,out,configs=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    gguf=GGUF(model);records=[]
    for suffix,kind in (('ffn_down.weight','linear'),('ffn_gate.weight','ffn'),('shortconv.in_proj.weight','linear')):
        infos=[t for t in gguf.tensors.values() if t.name.endswith(suffix) and t.type==2]
        if not infos:continue
        o,k=infos[0].shape
        x=(np.random.default_rng(29).normal(size=k)*.01).astype(np.float32)
        decoded=dequantize(gguf.packed(infos[0].name),2).reshape(o,k).astype(np.float64)
        expected=decoded@x.astype(np.float64)
        tolerance=np.sum(np.abs(decoded*x),axis=1)*3e-6+1e-10
        if kind=='ffn':
            up=dequantize(gguf.packed(infos[0].name.replace('_gate.','_up.')),2).reshape(o,k).astype(np.float64)@x.astype(np.float64)
            expected=expected/(1+np.exp(-expected))*up
            tolerance=1e-7+5e-5*np.abs(expected)
        with tensor.Device(provider='webgpu') as device:
            input=device.from_numpy(x);output=device.zeros(o)
            weights=[]
            for info in infos:
                buffers=[device.from_numpy(gguf.packed(info.name).view(np.uint32))]
                if kind=='ffn':buffers.append(device.from_numpy(gguf.packed(info.name.replace('_gate.','_up.')).view(np.uint32)))
                weights.append(buffers)
            for i,config in enumerate(CONFIGS if configs is None else configs):
                p=dict(r=1,k=k,o=o,type=2,sg=True,**config)
                path=out/f'{suffix}-{i}.py';artifact=path.with_suffix('.tbin')
                path.write_text(source(kind,p));tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache')
                kernel=device.load(artifact)
                kernel.launch(input,*weights[0],output)
                actual=output.to_numpy();error=float(np.max(np.abs(actual-expected)))
                if not np.all(np.isfinite(actual)) or not np.all(np.abs(actual-expected)<=tolerance):
                    row={'weight':suffix,'kind':kind,'schedule':config,'status':'failed','maximum_absolute_error':error}
                else:
                    calls=[]
                    for buffers in weights:
                        values,symbols,launch=kernel._bind((input,*buffers,output),{},include_outputs=True)
                        calls.append((kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True)))
                    plan=device.prepare_plan(calls);samples=[]
                    # Shader compilation leaves this desktop GPU idle. Warm it
                    # continuously before timing, including the smaller model.
                    deadline=time.perf_counter()+1
                    while time.perf_counter()<deadline:
                        for _ in range(10):plan.launch()
                        device.synchronize()
                    for repeat in range(8):
                        start=time.perf_counter()
                        for _ in range(10):plan.launch()
                        device.synchronize();elapsed=(time.perf_counter()-start)/10
                        if repeat:samples.append(elapsed)
                    row={'weight':suffix,'kind':kind,'shape':[o,k],'layers':len(infos),'schedule':config,'status':'ok',
                         'samples_seconds':samples,'median_seconds':statistics.median(samples),'maximum_absolute_error':error}
                row['source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
                row['artifact_sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
                records.append(row);print(row,flush=True)
            adapter=device.info
    report={'model':str(Path(model).resolve()),'adapter':adapter,'records':records,
            'protocol':'FP32 inputs and dequantized FP32 weights; 1 second continuous warmup after compilation, 1 discarded batch then 7 samples of 10 prepared plan launches, each streaming every matching layer; completion included'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--preset',choices=('scalar','dot'),default='scalar')
    args=parser.parse_args();run(args.model,args.out,DOTS if args.preset=='dot' else CONFIGS)
