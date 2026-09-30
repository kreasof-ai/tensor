"""Compile the LFM2 packed-weight forward plan through Tensor/NVRTC."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
from pathlib import Path
from dataclasses import asdict
import argparse,hashlib,json,time
import tensor
from tensor_llm import GGUF
from tensor_llm.config import Config
from tensor_llm.model import requirements
from tensor_llm.kernels import source
from tensor_llm.provenance import implementation_hashes
from tensor.artifacts.format import read_artifact


def produce(model,out,*,context=8448,rows=(1,128),target='sm_86'):
    if rows != (1,128) or type(context) is not int or context < 1:
        raise ValueError('requires rows=(1,128) and a positive context')
    gguf=GGUF(model);config=Config.from_gguf(gguf);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    capacity=(context+max(rows)+63)//64*64;wanted=requirements(gguf,capacity,rows);records={};start=time.perf_counter()
    (out/'src').mkdir(exist_ok=True);(out/'artifacts').mkdir(exist_ok=True)
    for i,(key,(kind,p)) in enumerate(wanted.items()):
        text=source(kind,p);src=out/'src'/(key+'.py');artifact=out/'artifacts'/(key+'.tbin')
        if not src.exists() or src.read_text()!=text:
            src.write_text(text);artifact.unlink(missing_ok=True)
        if artifact.exists():
            manifest,_=read_artifact(artifact)
            if manifest['target'] != target:artifact.unlink()
        if not artifact.exists():tensor.build(src,artifact,compiler='nvrtc',target=target,cache_dir=out/'compiler-cache')
        records[key]={'kind':kind,'parameters':p,'artifact':artifact.relative_to(out).as_posix(),
                      'sha256':hashlib.file_digest(artifact.open('rb'),'sha256').hexdigest()}
        print(f'{i+1}/{len(wanted)} {kind} {p}',flush=True)
    result={'schema':'tensor.lfm2-inference.v1','config':asdict(config),'capacity':capacity,'rows':rows,'target':target,
            'implementation':implementation_hashes(),'kernels':records,'build_seconds':time.perf_counter()-start}
    (out/'inference.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--context',type=int,default=8448);p.add_argument('--target',default='sm_86')
    args=p.parse_args();produce(args.model,args.out,context=args.context,target=args.target)

if __name__=='__main__':main()
