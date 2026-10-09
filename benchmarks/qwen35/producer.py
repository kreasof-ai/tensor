"""Produce native CUDA batch artifacts for the Qwen3.5 FP8 checkpoint."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from tensor.compiler.build import build_artifact
from tensor_llm.qwen35.checkpoint import Qwen35Checkpoint
from tensor_llm.qwen35.decode import implementation_hashes
from tensor_llm.qwen35.artifacts import requirements
from tensor_llm.qwen35.kernels.decode import source


def produce(checkpoint,out,*,slots=8,context=48000,target='sm_89',splits=16,kv_dtype='bfloat16'):
    c=Qwen35Checkpoint(checkpoint)
    c.config.state_bytes(slots=slots,context=context)
    if slots not in (1,2,4,8):raise ValueError('slots must be 1, 2, 4 or 8')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema='tensor.qwen35-batch.v1',status='building',slots=slots,context=context,
        target=target,splits=splits,kv_dtype=kv_dtype,config=asdict(c.config),implementation=implementation_hashes(),kernels={})
    (out/'inference.json').unlink(missing_ok=True)
    for key,(kind,p) in requirements(c.config,slots,context,splits=splits,kv_dtype=kv_dtype).items():
        entry=out/(key+'.py');artifact=out/(key+'.tbin')
        text=source(kind,p)
        if not artifact.is_file() or not entry.is_file() or entry.read_text()!=text:
            artifact.unlink(missing_ok=True);entry.write_text(text)
            build_artifact(entry,artifact,target=target,compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
        manifest['kernels'][key]=dict(kind=kind,parameters=p,path=artifact.name,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        print(kind,p,flush=True)
    manifest['status']='built'
    manifest['implementation']=implementation_hashes()
    temporary=out/'inference.json.tmp';temporary.write_text(json.dumps(manifest,indent=2)+'\n')
    temporary.replace(out/'inference.json')
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--slots',type=int,default=8)
    p.add_argument('--context',type=int,default=48000)
    p.add_argument('--kv-dtype',choices=('bfloat16','fp8'),default='bfloat16')
    a=p.parse_args();produce(a.checkpoint,a.out,slots=a.slots,context=a.context,kv_dtype=a.kv_dtype)
