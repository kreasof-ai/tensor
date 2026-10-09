"""Bounded prefill tile sweep against exact FP16-operand matrix products."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.runtime.abi import BoundCall
from tensor_llm import GGUF
from tensor_llm.common.gguf import dequantize
from tensor_llm.lfm2.kernels.webgpu import source


def run(model,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    gguf=GGUF(model);results=[]
    for name in ('blk.0.ffn_down.weight','blk.0.ffn_gate.weight'):
        info=gguf.tensors[name];o,k=info.shape
        raw=gguf.packed(name);weights=(raw.view(np.float16) if info.type==1 else dequantize(raw,info.type)).reshape(o,k)
        rng=np.random.default_rng(29);x=(rng.normal(size=(32,k))*.01).astype(np.float32)
        expected=x.astype(np.float16).astype(np.float32)@weights.astype(np.float16).astype(np.float32).T
        for tile in ((16,32,32),(32,32,32),(16,64,32),(32,64,32),(16,32,64),(32,32,64),(16,64,64),(32,64,64)):
            p=dict(r=32,k=k,o=o,type=info.type,tile=tile)
            path=out/f'{name}-{tile[0]}-{tile[1]}-{tile[2]}.py';artifact=path.with_suffix('.tbin')
            text=source('linear',p)
            if path.exists() and path.read_text()!=text:artifact.unlink(missing_ok=True)
            path.write_text(text)
            if not artifact.exists():tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache')
            with tensor.Device(provider='webgpu') as device:
                kernel=device.load(artifact);input=device.from_numpy(x.flatten());weight=device.from_numpy(raw.view(np.float16 if info.type==1 else np.uint32));output=device.zeros(32*o)
                values,symbols,launch=kernel._bind((input,weight,output),{},include_outputs=True)
                plan=device.prepare_plan([(kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True))])
                plan.launch();actual=output.to_numpy().reshape(32,o)
                np.testing.assert_allclose(actual,expected,rtol=3e-4,atol=1e-5)
                samples=[]
                for repeat in range(6):
                    start=time.perf_counter()
                    for _ in range(10):plan.launch()
                    output.to_numpy();elapsed=(time.perf_counter()-start)/10
                    if repeat:samples.append(elapsed)
                row={'weight':name,'type':info.type,'tile':tile,'samples_seconds':samples,'median_seconds':statistics.median(samples),'maximum_absolute_error':float(np.max(np.abs(actual-expected)))}
                results.append(row);print(row,flush=True)
    (out/'report.json').write_text(json.dumps({'model':str(model),'results':results,'protocol':'five samples after warmup, ten launches plus completion/readback per sample'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.model,a.out)
