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


# The eight configurations swept for the 230M report. Left unchanged so its
# recorded evidence keeps its meaning.
LEGACY=[{}, {'lhs_pad':1}, {'lhs_pad':8}, {'lhs_transpose':True,'lhs_pad':1},
        {'dot_width':4}, {'dot_width':4,'lhs_pad':1},
        {'dot_width':4,'lhs_transpose':True,'lhs_pad':1},
        {'dot_width':4,'unroll':True}]

# Output/register tile sweep. r=32 bounds tile_m at 32. The 2.6B FFN shapes
# (k,o in {2048,10752}) divide evenly by 32/64/128, so every tile here is
# tail-free. Larger tile_n/tile_k raise workgroup storage pressure against the
# adapter's 32 KiB limit, which is part of what this sweep measures.
TILES=[{'dot_width':4},{'dot_width':1},
       {'tile':(16,64,64),'dot_width':4},{'tile':(16,64,128),'dot_width':4},
       {'tile':(16,128,32),'dot_width':4},{'tile':(32,32,64),'dot_width':4},
       {'tile':(32,64,32),'dot_width':4},{'tile':(32,64,64),'dot_width':4},
       {'tile':(32,64,128),'dot_width':4},{'tile':(32,128,32),'dot_width':4},
       {'tile':(32,128,64),'dot_width':4},{'tile':(32,64,64),'dot_width':1}]

PRESETS={'legacy':LEGACY,'tiles':TILES}

# Rejected: an [n][k] weight layout, so the four lanes of a dot_width load sit
# adjacent instead of tile_n apart. Measured 1.4-1.9x SLOWER, and it failed the
# ffn_down correctness gate. Cause: threads index the weight column as tx%nr,
# so under [n][k] all 16 distinct columns are 64 elements apart and collide on
# the same LDS bank, while [k][n] keeps consecutive columns adjacent. Making
# this work needs a swizzle, not a layout swap.


def run(model,out,preset='legacy'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    gguf=GGUF(model);records=[]
    configs=PRESETS[preset]
    for name in ('blk.0.ffn_down.weight','blk.0.ffn_gate.weight'):
        info=gguf.tensors[name];o,k=info.shape;raw=gguf.packed(name)
        weights=(raw.view(np.float16) if info.type==1 else dequantize(raw,info.type)).reshape(o,k)
        x=(np.random.default_rng(29).normal(size=(32,k))*.01).astype(np.float32)
        expected=x.astype(np.float16).astype(np.float32)@weights.astype(np.float16).astype(np.float32).T
        for i,config in enumerate(configs):
            p=dict(r=32,k=k,o=o,type=info.type,dot_width=1);p.update(config)
            path=out/f'{name}-{preset}-{i}.py';artifact=path.with_suffix('.tbin');path.write_text(source('linear',p))
            artifact.unlink(missing_ok=True)
            try:
                tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache')
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
                    row={'weight':name,'type':info.type,'schedule':config,'status':'ok','tile':p.get('tile',(16,32,64)),
                         'dot_width':p['dot_width'],'samples_seconds':samples,'median_seconds':statistics.median(samples),
                         'maximum_absolute_error':float(np.max(np.abs(actual-expected)))}
            except Exception as error:
                # Larger tiles can exceed the adapter's 32 KiB workgroup storage
                # limit, and any retiling changes FP32 summation order. Record the
                # failure and keep sweeping rather than aborting the run.
                row={'weight':name,'type':info.type,'schedule':config,'status':'failed','tile':p.get('tile',(16,32,64)),
                     'dot_width':p['dot_width'],'error':f'{type(error).__name__}: {str(error)[-300:]}'}
            records.append(row);print(row,flush=True)
            (out/'report.json').write_text(json.dumps({'model':str(model),'preset':preset,'records':records,
                'protocol':'seven samples after warmup, twenty launches and completion per sample; identical inputs'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--preset',choices=tuple(PRESETS),default='legacy')
    a=p.parse_args();run(a.model,a.out,a.preset)
