"""Compare manual nanoGPT losses, gradients, parameters and AdamW state to Torch."""

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
import torch
import tensor as tx
from tensor_nn import NanoGPT
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.nanogpt.reference import Reference


def difference(actual,expected):
    actual=np.asarray(actual,dtype=np.float32);expected=np.asarray(expected,dtype=np.float32)
    delta=actual-expected
    return {'maximum_absolute_error':float(np.max(np.abs(delta))),
            'relative_l2_error':float(np.linalg.norm(delta.astype(np.float64).reshape(-1))/max(np.linalg.norm(expected.astype(np.float64).reshape(-1)),1e-12))}


def batches(config,steps,seed=123):
    rng=np.random.default_rng(seed)
    data=rng.integers(0,config.vocab,(steps,config.batch,config.sequence+1),dtype=np.int32)
    # Repeated token ids exercise scatter accumulation and tied-weight gradients.
    data[:,:,:min(config.sequence,4)]=7
    return [(np.ascontiguousarray(x[:,:-1]),np.ascontiguousarray(x[:,1:])) for x in data]


def validate(directory, *, steps=10, out=None):
    started=time.perf_counter();report={'schema':'tensor.phase6-training-validation.v1','status':'running','steps':[]}
    def save():
        if out:
            Path(out).parent.mkdir(parents=True,exist_ok=True)
            Path(out).write_text(json.dumps(report,indent=2)+'\n')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    with tx.Device() as device:
        model=NanoGPT(directory,device);cfg=model.config;reference=Reference(cfg)
        report.update(optimizer_reference_uses_identical_gradients=True,
                      gradient_reference='PyTorch autograd before replacing its gradients; matched state at each update',
                      config=vars(cfg),adapter=device.info,parameter_count=sum(p.weight.shape[0] for p in model.parameters.values()),
                      owned_device_bytes=model.allocated_bytes,precision='FP16 compute / FP32 master weights, reductions, gradients and AdamW',
                      training_manifest_sha256=hashlib.sha256((Path(directory)/'training.json').read_bytes()).hexdigest())
        for step,(x,y) in enumerate(batches(cfg,steps)):
            model.set_batch(x,y);model.forward_backward();device.synchronize()
            loss=model.read_loss()
            tx_x=torch.from_numpy(x.astype(np.int64)).cuda();tx_y=torch.from_numpy(y.astype(np.int64)).cuda()
            expected_loss=float(reference.forward_backward(tx_x,tx_y).detach())
            entry={'step':step+1,'loss':loss,'reference_loss':expected_loss,'gradients':{},'parameters':{},'moments':{},'variances':{}}
            report['steps'].append(entry);save()
            try:
                np.testing.assert_allclose(loss,expected_loss,atol=0.002,rtol=0.001)
                for name,param in model.parameters.items():
                    actual=param.grad.to_numpy().reshape(param.shape)/cfg.loss_scale
                    expected=reference.mapping()[name].grad.detach().float().cpu().numpy()/cfg.loss_scale
                    entry['gradients'][name]=difference(actual,expected)
                    np.testing.assert_allclose(actual,expected,atol=3e-5,rtol=0.03,err_msg=f'gradient {name}, step {step+1}')
                # Isolate optimizer arithmetic: AdamW can amplify tiny gradient
                # sign differences into a full learning-rate update. Gradients
                # were independently checked above before this replacement.
                for name,param in model.parameters.items():
                    same=param.grad.to_numpy().reshape(param.shape)
                    reference.mapping()[name].grad.copy_(torch.from_numpy(same).cuda())
                model.update();device.synchronize();entry['norm']=model.read_norm()
                expected_norm=float(reference.update());entry['reference_norm']=expected_norm
                np.testing.assert_allclose(entry['norm'],expected_norm,atol=0.002,rtol=0.02)
                for name,param in model.parameters.items():
                    weight=reference.mapping()[name]
                    state=reference.optimizer.state[weight]
                    for field,buffer,expected in [('parameters',param.weight,weight),('moments',param.moment,state['exp_avg']),('variances',param.variance,state['exp_avg_sq'])]:
                        actual=buffer.to_numpy().reshape(param.shape);expected=expected.detach().cpu().numpy()
                        entry[field][name]=difference(actual,expected)
                        if field=='parameters': atol,rtol=2e-7,2e-5
                        elif field=='moments': atol,rtol=2e-7,2e-5
                        else: atol,rtol=1e-10,3e-5
                        np.testing.assert_allclose(actual,expected,atol=atol,rtol=rtol,err_msg=f'{field} {name}, step {step+1}')
                entry['status']='passed'
                print(f'passed step {step+1}: loss={loss:.6f} reference={expected_loss:.6f} norm={entry["norm"]:.6f}',flush=True)
            except BaseException as error:
                entry['status']='failed';report['status']='failed';report['error']=str(error);save();raise
            save()
        report['final_state_samples']={}
        for name,param in model.parameters.items():
            indices=np.unique(np.linspace(0,param.weight.shape[0]-1,min(64,param.weight.shape[0]),dtype=np.int64))
            report['final_state_samples'][name]={'indices':indices.tolist(),**{field:getattr(param,field).to_numpy()[indices].tolist() for field in ('weight','moment','variance')}}
        report.update(status='passed',seconds=time.perf_counter()-started,
                      tolerances={'loss':{'atol':0.002,'rtol':0.001},'gradients':{'atol':3e-5,'rtol':0.03},
                                  'parameters':{'atol':2e-7,'rtol':2e-5},'moments':{'atol':2e-7,'rtol':2e-5},'variances':{'atol':1e-10,'rtol':3e-5}})
        save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=10);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();validate(args.bundle,steps=args.steps,out=args.out)

if __name__=='__main__':main()
