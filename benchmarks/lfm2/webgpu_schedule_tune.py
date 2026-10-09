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
from tensor_llm.common.gguf import dequantize
from tensor_llm.lfm2.kernels.webgpu import source


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

# Rejected, three times over. The prefill inner loop builds each 4-wide weight
# operand from rhs[k*4+lane, col], so the four lanes are tile_n elements apart
# and can only ever be four scalar LDS reads. Three routes to adjacent addresses
# were tried; all three are closed now, on measurement rather than assumption.
#
#  1. TileLang's own swizzle. make_swizzled_layout and the make_{half,quarter}
#     bank-swizzled layouts compile, pass the correctness gate with byte-identical
#     max abs err to baseline, and reach WGSL -- and are 3.8-4.8 percent SLOWER
#     on ffn_gate. Enabling tirx.disable_vectorize for one diagnostic run does not
#     change that. make_full_bank_swizzled_layout still fails on a real TileLang
#     contiguity constraint: continuous % (vector_size * 8) == 0 with continuous=32
#     and vector_size=8 for f16, so it needs a contiguous dimension >= 64 and does
#     not apply at tile_n=32.
#
#     This route was first recorded as upstream-blocked and that was wrong. TileLang
#     0.1.14 lowers a swizzled shared buffer on target=webgpu fine. The failure was
#     ours: lower_simt_gemm replaced alloc_buffers while passing annotations through
#     unchanged, so the layout map kept pointing at Vars the rebuilt block no longer
#     allocated. remap_annotations in webgpu_lowering.py now re-keys them. If you
#     see "buffer rhs is not found in the block" again, suspect the lowering pass
#     before filing anything upstream.
#
#  2. A hand-rolled blocked staging tile, rhs shaped (tile_k/4, tile_n, 4) so
#     the lane becomes the innermost axis. It compiles and it is faster at the
#     same tile -- ffn_gate 0.948 ms blocked against 1.025 ms plain at
#     (32,32,32) -- but it is wrong. Max absolute error against the FP16-operand
#     reference rises from 3.4e-08 to 1.1e-05, and all six ffn_down variants fail
#     their gate outright. Compared against an FP32-operand reference the result
#     matches neither contract, so the staged value is not merely skipping the
#     FP16 rounding: a 3-D shared store changes what TileLang emits in a way this
#     template does not control.
#
#     If you re-measure this, hold the tile fixed. The first sweep only ran a
#     plain baseline at (32,32,64), so its 0.948-vs-1.081 ms reading mixed a
#     tile-shape win with a staging win; plain (32,32,32) is 1.025 ms, and only
#     about 0.077 ms of that gap is the staging. Do not quote the larger figure.
#
# The faster blocked number is not real performance and must not be quoted.
# Attribution is in docs/research/lfm2-2.6b-q4_0-matched-run.md.
#
# Also rejected: an [n][k] weight layout. Measured 1.4-1.9x SLOWER, and it
# failed the ffn_down correctness gate. Cause: threads index the weight column
# as tx%nr, so under [n][k] all 16 distinct columns are 64 elements apart and
# collide on one LDS bank, while [k][n] keeps consecutive columns adjacent.
# Fixing that needs a swizzle, not a layout swap -- and point 1 shows the swizzle
# is already correct here and buys nothing.


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
