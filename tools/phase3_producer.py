"""Package the five already-built profiles into a relocatable module graph."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

ROOT=Path(__file__).resolve().parents[1]
EXAMPLES=('elementwise','gemm_relu','dynamic_affine','dynamic_gemm','scalar_offset')


def produce(artifacts,out):
    from tensor.artifact import read_artifact
    from tensor.modules import pack
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    for module,names in (('tensor-base',('elementwise',)),('tensor-ops',EXAMPLES[1:])):
        path=out/'modules'/module
        (path/'src').mkdir(parents=True)
        (path/'artifacts').mkdir()
        exports={}
        for name in names:
            shutil.copyfile(ROOT/f'examples/{name}.py',path/f'src/{name}.py')
            shutil.copyfile(Path(artifacts)/f'{name}.tbin',path/f'artifacts/{name}.tbin')
            manifest,_=read_artifact(path/f'artifacts/{name}.tbin')
            assert manifest['source_sha256']==hashlib.sha256((path/f'src/{name}.py').read_bytes()).hexdigest()
            exports[name]={'source':f'src/{name}.py','portable':f'artifacts/{name}.tbin','artifacts':[f'artifacts/{name}.tbin']}
        manifest={'formatVersion':1,'name':module,'version':'0.1.0','tensorAbi':1,'exports':exports,
                  'capabilities':['contiguous'],
                  'dependencies':{'tensor-base':{'path':'../tensor-base','version':'0.1.0'}} if module=='tensor-ops' else {}}
        (path/'tensor.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    result=pack(out/'modules/tensor-ops',out/'ops.tpack',cache_dir=out/'module-cache')
    repeat=pack(out/'modules/tensor-ops',out/'repeat.tpack',cache_dir=out/'module-cache')
    assert result['sha256']==repeat['sha256']
    (out/'repeat.tpack').unlink()
    report={'status':'passed','hostname':socket.gethostname(),
            'revision':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
            'package_sha256':result['sha256'],'package_bytes':result['bytes'],'deterministic':True,
            'modules':result['modules'],'examples':list(EXAMPLES)}
    (out/'phase3-producer.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=produce(args.artifacts,args.out)
    print(json.dumps({'status':result['status'],'package_sha256':result['package_sha256'],'modules':list(result['modules'])}))
