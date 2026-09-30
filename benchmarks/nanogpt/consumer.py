"""Execute ten standalone GPT updates with compiler/framework imports prohibited."""

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

from pathlib import Path
import argparse
import hashlib
import importlib.abc
from importlib.metadata import distributions,version
import json
import platform
import socket
import sys
import time


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in {'torch','tilelang','tvm','tvm_ffi','triton'}:
            raise ImportError('compiler/framework import prohibited: '+fullname)


def consume(bundle,reference,out):
    sys.meta_path.insert(0,Guard())
    packages=sorted(d.metadata['Name'] for d in distributions())
    forbidden={'torch','tilelang','apache-tvm-ffi','triton'}
    if forbidden & {p.lower().replace('_','-') for p in packages}:
        raise RuntimeError('install the supplied Tensor wheel into a clean consumer environment')
    import numpy as np
    import tensor as tx
    from tensor_nn import NanoGPT
    fixture=json.loads(Path(reference).read_text())
    if fixture['status']!='passed' or len(fixture['steps'])!=10:
        raise ValueError('requires a successful ten-update reference fixture')
    manifest_bytes=(Path(bundle)/'training.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest()!=fixture['training_manifest_sha256']:
        raise ValueError('reference fixture belongs to a different training bundle')
    report={'schema':'tensor.phase6-consumer.v1','status':'running','packages':packages,
        'hostname':socket.gethostname(),'platform':platform.platform(),'steps':[]}
    started=time.perf_counter()
    with tx.Device() as device:
        model=NanoGPT(bundle,device);cfg=model.config
        rng=np.random.default_rng(123)
        data=rng.integers(0,cfg.vocab,(10,cfg.batch,cfg.sequence+1),dtype=np.int32)
        data[:,:,:min(cfg.sequence,4)]=7
        for index,batch in enumerate(data):
            model.set_batch(np.ascontiguousarray(batch[:,:-1]),np.ascontiguousarray(batch[:,1:]))
            begin=time.perf_counter();model.step();device.synchronize();elapsed=time.perf_counter()-begin
            loss=model.read_loss();norm=model.read_norm();expected=fixture['steps'][index]
            np.testing.assert_allclose(loss,expected['loss'],rtol=0.001,atol=0.002)
            np.testing.assert_allclose(norm,expected['norm'],rtol=0.02,atol=0.002)
            report['steps'].append({'step':index+1,'loss':loss,'gradient_norm':norm,'completed_seconds':elapsed,'status':'passed'})
            print(f'passed clean consumer update {index+1}, loss={loss:.6f}',flush=True)
        for name,values in fixture['final_state_samples'].items():
            param=model.parameters[name];indices=values['indices']
            for field in ('weight','moment','variance'):
                actual=getattr(param,field).to_numpy()[indices]
                np.testing.assert_allclose(actual,values[field],rtol=0.003,atol=3e-6 if field!='variance' else 1e-8)
        report.update(status='passed',adapter=device.info,config=vars(cfg),owned_device_bytes=model.allocated_bytes,
            compiler_import_guard=True,compiler_imports=sorted({'torch','tilelang','tvm','tvm_ffi','triton'} & set(sys.modules)),
            versions={name:version(name) for name in ('numpy','tensor-workspace','tensor-nn')},
            training_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            reference_sha256=hashlib.sha256(Path(reference).read_bytes()).hexdigest(),
            final_state_samples_checked=True,seconds=time.perf_counter()-started)
    Path(out).write_text(json.dumps(report,indent=2)+'\n');return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.parent.mkdir(parents=True,exist_ok=True);consume(args.bundle,args.reference,args.out)

if __name__=='__main__':main()
