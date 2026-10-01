"""Prefill shared layouts and scalar/vector dot scheduling on actual GGUF weights."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.runtime.abi import BoundCall
from tensor_llm import GGUF
from tensor_llm.gguf import dequantize
from tensor_llm.webgpu_kernels import source


def run(model,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    gguf=GGUF(model);records=[]
    configs=[{}, {'lhs_pad':1}, {'lhs_pad':8}, {'lhs_transpose':True,'lhs_pad':1},
             {'dot_width':4}, {'dot_width':4,'lhs_pad':1},
             {'dot_width':4,'lhs_transpose':True,'lhs_pad':1},
             {'dot_width':4,'unroll':True}]
    for name in ('blk.0.ffn_down.weight','blk.0.ffn_gate.weight'):
        info=gguf.tensors[name];o,k=info.shape;raw=gguf.packed(name)
        weights=(raw.view(np.float16) if info.type==1 else dequantize(raw,info.type)).reshape(o,k)
        x=(np.random.default_rng(29).normal(size=(32,k))*.01).astype(np.float32)
        expected=x.astype(np.float16).astype(np.float32)@weights.astype(np.float16).astype(np.float32).T
        for i,config in enumerate(configs):
            p=dict(r=32,k=k,o=o,type=info.type,dot_width=1);p.update(config)
            path=out/f'{name}-{i}.py';artifact=path.with_suffix('.tbin');path.write_text(source('linear',p))
            artifact.unlink(missing_ok=True);tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache')
            with tensor.Device(provider='webgpu') as device:
                kernel=device.load(artifact);output=device.zeros(32*o)
                args=(device.from_numpy(x.ravel()),device.from_numpy(raw.view(np.float16 if info.type==1 else np.uint32)),output)
                values,symbols,launch=kernel._bind(args,{},include_outputs=True)
                plan=device.prepare_plan([(kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True))])
                plan.launch();actual=output.to_numpy().reshape(32,o)
                np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=1e-5)
                samples=[]
                for repeat in range(8):
                    start=time.perf_counter()
                    for _ in range(20):plan.launch()
                    output.to_numpy();elapsed=(time.perf_counter()-start)/20
                    if repeat:samples.append(elapsed)
                row={'weight':name,'type':info.type,'schedule':config,'samples_seconds':samples,
                     'median_seconds':statistics.median(samples),'maximum_absolute_error':float(np.max(np.abs(actual-expected)))}
                records.append(row);print(row,flush=True)
            (out/'report.json').write_text(json.dumps({'model':str(model),'records':records,
                'protocol':'seven samples after warmup, twenty launches and completion per sample; identical inputs'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.model,a.out)
