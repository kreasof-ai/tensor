"""Build and optionally tune a standalone nanoGPT training bundle through NVRTC."""

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

from pathlib import Path
import argparse
from dataclasses import asdict
import hashlib
import json
import platform
import socket
import subprocess
import time
import numpy as np
import tensor as tx
from tensor_nn.nanogpt import GPTConfig,requirements
from tensor_nn.kernels import source
from tensor_nn.provenance import implementation_hashes
from tensor.artifacts.modules import pack
from tensor.compiler.tuning import tune


def produce(directory, config, *, target='sm_86', autotune=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    module=directory/'module';(module/'src').mkdir(parents=True);(module/'artifacts').mkdir()
    started=time.perf_counter();records={};exports={}
    schedules=[(32,64,32,2),(64,64,32,2),(32,64,32,1)]
    device=tx.Device() if autotune else None
    if device: device.__enter__()
    try:
        for index,(key,(kind,p)) in enumerate(requirements(config).items()):
            candidates={};builds={};paths={}
            choices=schedules if autotune and kind in ('gemm','gemm_gelu','gemm_residual') else [None]
            for candidate,schedule in enumerate(choices):
                name=f'k_{key}_{candidate}'
                text=source(kind,p,schedule);src=module/'src'/(name+'.py');src.write_text(text)
                out=module/'artifacts'/(name+'.tbin')
                build=tx.build(src,out,target=target,compiler='nvrtc',cache_dir=directory/'compiler-cache')
                builds[str(candidate)]={'schedule':schedule,'build_seconds':build['seconds'],'cache_hit':build['cache_hit']}
                paths[str(candidate)]=(src,out)
                if len(choices)>1: candidates[str(candidate)]=device.load(out)
            selected='0';tuning=None
            if candidates:
                b,m,k,n=p['batch'],p['m'],p['k'],p['cols']
                inputs=[device.randn((b*m*k,),'float16',seed=1),device.randn((b*k*n,),'float16',seed=2)]
                output=device.empty((b*m*n,),'float16')
                extra=[]
                if kind=='gemm_gelu':extra=[device.empty((b*m*n,),'float16')]
                elif kind=='gemm_residual':extra=[device.randn((b*m*n,),'float16',seed=3)]
                inputs.extend(extra)
                candidates['0'].launch(*inputs,output)
                reference=output.to_numpy()
                checks=[(extra[0],extra[0].to_numpy())] if kind=='gemm_gelu' else []
                tuning=tune(candidates,inputs,output,reference,checks=checks)
                tuning['reference']='default Tensor schedule; independently checked by training/reference validation'
                selected=tuning['selected']
                for value in [*inputs,output]:value.release()
                for kernel in candidates.values():kernel.release()
            src,out=paths[selected]
            records[key]={'kind':kind,'parameters':p,'artifact':out.relative_to(directory).as_posix(),
                'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'schedule':builds[selected]['schedule'],
                'builds':builds,'tuning':tuning}
            export='k_'+key
            exports[export]={'source':src.relative_to(module).as_posix(),'portable':out.relative_to(module).as_posix(),
                             'artifacts':[out.relative_to(module).as_posix()]}
            print(f'{index+1}/{len(requirements(config))} {kind} {p} selected={selected}',flush=True)
        (module/'tensor.json').write_text(json.dumps({'formatVersion':1,'name':'tensor-nanogpt-training','version':'0.1.0',
            'tensorAbi':1,'capabilities':['contiguous'],'exports':exports},indent=2)+'\n')
        package=pack(module,directory/'training.tpack',cache_dir=directory/'module-cache')
        report={'schema':'tensor.manual-nanogpt.v2','config':asdict(config),'target':target,'kernels':records,
            'implementation_sha256':implementation_hashes(),
            'producer':{'hostname':socket.gethostname(),'platform':platform.platform(),
                        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()},
            'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                            ['packages/tensor-nn/src/tensor_nn/kernels.py','packages/tensor-nn/src/tensor_nn/nanogpt.py',
                             'src/tensor/runtime/manual.py','src/tensor/compiler/tuning.py','benchmarks/nanogpt/producer.py']},
            'autotuning':{'enabled':autotune,'adapter':device.info if device else None,
                         'search_space':schedules if autotune else None},
            'module_sha256':package['sha256'],'build_and_tune_seconds':time.perf_counter()-started}
        (directory/'training.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
    finally:
        if device:device.__exit__(None,None,None)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--target',default='sm_86')
    parser.add_argument('--diagnostic',action='store_true');parser.add_argument('--tune',action='store_true')
    args=parser.parse_args()
    config=GPTConfig.diagnostic() if args.diagnostic else GPTConfig()
    result=produce(args.out,config,target=args.target,autotune=args.tune)
    print(json.dumps({'kernels':len(result['kernels']),'build_and_tune_seconds':result['build_and_tune_seconds']}))

if __name__=='__main__':main()
