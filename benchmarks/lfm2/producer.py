"""Compile the LFM2 packed-weight plan through Tensor/NVRTC or WGSL."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,time
import tensor
from tensor_llm import GGUF
from tensor_llm.config import Config
from tensor_llm.model import requirements,valid_rows,WEBGPU_PROFILES
from tensor_llm.cuda_kernels import CUDA_PROFILES
from tensor_llm.kernels import source
from tensor_llm.provenance import implementation_hashes
from tensor.artifacts.format import read_artifact
from tensor.compiler.search import ScheduleProfile


def produce(model,out,*,context=8448,rows=None,target=None,provider='cuda',webgpu_profile='portable',cuda_profile='default',schedule_profile=None):
    if provider not in ('cuda','webgpu'):raise ValueError('requires CUDA or WebGPU')
    rows=rows or ((1,32) if provider=='webgpu' else (1,128))
    target=target or ('webgpu-portable-v1' if provider=='webgpu' else 'sm_86')
    gguf=GGUF(model);config=Config.from_gguf(gguf)
    if cuda_profile not in CUDA_PROFILES:raise ValueError('unsupported CUDA kernel profile')
    if not valid_rows(provider,rows,cuda_profile if provider=='cuda' else webgpu_profile) or type(context) is not int or not 1<=context<=config.max_context:
        raise ValueError('requires the provider row profile and a supported positive context')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    capacity=(context+max(rows)+63)//64*64;wanted=requirements(gguf,capacity,rows,provider=provider,webgpu_profile=webgpu_profile,cuda_profile=cuda_profile);records={};start=time.perf_counter()
    kernel_source=source
    selected_profile=None
    if schedule_profile is not None:
        selected_profile=ScheduleProfile(json.loads(Path(schedule_profile).read_text()))
        selected_profile.select('',{},provider=provider,target=target)
    elif provider=='cuda' and cuda_profile=='optimized' and target=='sm_86':
        path=Path(__file__).with_name('profiles')/'cuda-sm86-lfm2.5-2.6b.json'
        selected_profile=ScheduleProfile(json.loads(path.read_text()))
    if provider=='cuda' and cuda_profile=='optimized':
        from tensor_llm.cuda_kernels import source as kernel_source
    if provider=='webgpu':
        from tensor_llm.webgpu_kernels import source as kernel_source
        from tensor.compiler.webgpu import build_webgpu
        compiler_path=Path(build_webgpu.__code__.co_filename)
        lowering_hash=hashlib.sha256(compiler_path.with_name('webgpu_lowering.py').read_bytes()).hexdigest()
        producer_hash=hashlib.sha256(compiler_path.read_bytes()).hexdigest()
    (out/'src').mkdir(exist_ok=True);(out/'artifacts').mkdir(exist_ok=True)
    for i,(key,(kind,p)) in enumerate(wanted.items()):
        schedule=selected_profile.select(kind,p,provider=provider,target=target) if selected_profile else {}
        text=kernel_source(kind,{**p,**schedule});src=out/'src'/(key+'.py');artifact=out/'artifacts'/(key+'.tbin')
        if not src.exists() or src.read_text()!=text:
            src.write_text(text);artifact.unlink(missing_ok=True)
        if artifact.exists():
            manifest,_=read_artifact(artifact)
            if (manifest['target'] != target
                or provider=='cuda' and manifest['compiler'].get('lowering_sha256')!=hashlib.sha256(Path(tensor.__file__).parent.joinpath('compiler/cuda_lowering.py').read_bytes()).hexdigest()
                or provider=='webgpu' and (manifest['compiler'].get('lowering_sha256')!=lowering_hash or manifest['compiler'].get('producer_sha256')!=producer_hash)):artifact.unlink()
        if not artifact.exists():tensor.build(src,artifact,compiler='wgsl' if provider=='webgpu' else 'nvrtc',provider=provider,target=target,cache_dir=out/'compiler-cache')
        records[key]={'kind':kind,'parameters':p,'artifact':artifact.relative_to(out).as_posix(),
                      'schedule':schedule,
                      'sha256':hashlib.file_digest(artifact.open('rb'),'sha256').hexdigest()}
        print(f'{i+1}/{len(wanted)} {kind} {p}',flush=True)
    result={'schema':'tensor.lfm2-inference.v1','config':asdict(config),'capacity':capacity,'rows':rows,'target':target,
            'provider':provider,'implementation':implementation_hashes(provider),'kernels':records,'build_seconds':time.perf_counter()-start}
    if provider=='webgpu':result['webgpu_profile']=webgpu_profile
    else:result['cuda_profile']=cuda_profile
    if selected_profile:result['schedule_profile']={**selected_profile.data,'sha256':selected_profile.sha256}
    (out/'inference.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--context',type=int,default=8448);p.add_argument('--target')
    p.add_argument('--provider',choices=('cuda','webgpu'),default='cuda')
    p.add_argument('--webgpu-profile',choices=WEBGPU_PROFILES,default='portable')
    p.add_argument('--cuda-profile',choices=CUDA_PROFILES,default='default')
    p.add_argument('--schedule-profile',type=Path,help='target-bound producer schedule profile JSON')
    chunks=p.add_mutually_exclusive_group()
    chunks.add_argument('--prefill-chunk',type=int,choices=(32,64,128))
    chunks.add_argument('--prefill-chunks',type=int,nargs='+',choices=(32,64,128))
    args=p.parse_args();produce(args.model,args.out,context=args.context,target=args.target,provider=args.provider,webgpu_profile=args.webgpu_profile,cuda_profile=args.cuda_profile,schedule_profile=args.schedule_profile,
                              rows=(1,*args.prefill_chunks) if args.prefill_chunks else (1,args.prefill_chunk) if args.prefill_chunk else None)

if __name__=='__main__':main()
