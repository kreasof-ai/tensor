"""Ablate queue fencing, separate copy, and mapping flush with identical shaders."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,math
from pathlib import Path
import numpy as np
import wgpu
from tensor_llm import LFM2
from benchmarks.lfm2 import decode_search_compare as compare


def legacy_readback(device,buffer):
    device.synchronize()
    staging=device._readback_staging(buffer)
    encoder=device._gpu.create_command_encoder()
    encoder.copy_buffer_to_buffer(buffer._storage,0,staging,0,buffer._allocated)
    device._gpu.queue.submit([encoder.finish()])
    staging.map_sync(wgpu.MapMode.READ)
    try:
        return np.frombuffer(staging.read_mapped(copy=False),dtype=buffer.dtype,
                             count=math.prod(buffer.shape)).reshape(buffer.shape).copy()
    finally:staging.unmap()


def install_legacy_readback(engine):
    for plan in (*engine.prepared.values(),*engine.greedy.values()):
        launch=plan.launch
        def legacy(*,readback=None,launch=launch):
            launch()
            if readback is not None:return legacy_readback(engine.device,readback)
        plan.launch=legacy
    return engine


def install_python_submission(engine):
    for plan in (*engine.prepared.values(),*engine.greedy.values()):plan._submit_native=None
    return engine


def run(a):
    compare.run(a.model,a.bundle,a.searched or a.bundle,a.reference,a.fixtures,a.out,a.repeats,
                runtime_bundle=a.bundle if a.searched else None,
                baseline_wrapper=install_python_submission if a.ablation=='submission' else install_legacy_readback,
                max_buffer_size=a.max_buffer_size)
    import json
    path=a.out/'report.json';report=json.loads(path.read_text())
    report['protocol']['runtime_ablation']=('tensor_before: Python command encoder/pass/finish/submit wrappers with native dispatch encoding; tensor_searched: native fresh command creation/encoding/copy/submission/release. Identical shaders, ordered map completion and owned host snapshots.' if a.ablation=='submission' else 'tensor_before: original queue fence, separate copy submission, public map_sync empty flush; tensor_runtime (or tensor_searched in a runtime-only run): same shaders with combined compute/copy submission and map completion only; tensor_searched with --searched: extended kernel search too')
    path.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','reference','fixtures','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--searched',type=Path);p.add_argument('--repeats',type=int,default=7)
    p.add_argument('--max-buffer-size',type=int)
    p.add_argument('--ablation',choices=('readback','submission'),default='readback');run(p.parse_args())
