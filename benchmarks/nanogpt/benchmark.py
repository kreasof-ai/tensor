"""Repeat ten complete nanoGPT updates, restoring all training state per window."""

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

from pathlib import Path
import argparse
import gc
import hashlib
import json
import platform
import statistics
import sys
import time
import numpy as np
import torch
import tensor as tx
from tensor_nn import NanoGPT,GPTConfig
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from benchmarks.nanogpt.reference import Reference
from benchmarks.nanogpt.validate import batches


def benchmark(bundle, *, windows=5, steps=10, out=None, providers=None):
    manifest=json.loads((Path(bundle)/'training.json').read_text());cfg=GPTConfig(**manifest['config'])
    data=batches(cfg,steps)
    providers=providers or ['tensor','torch_eager_explicit','torch_inductor_explicit','torch_eager_sdpa','torch_inductor_sdpa','torch_eager_sdpa_fused_adamw','torch_inductor_sdpa_fused_adamw']
    report={'schema':'tensor.phase6-training-benchmark.v1','status':'running','config':vars(cfg),'platform':platform.platform(),
        'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(),
        'training_manifest_sha256':hashlib.sha256((Path(bundle)/'training.json').read_bytes()).hexdigest(),
        'batch_sha256':[hashlib.sha256(x.tobytes()+y.tobytes()).hexdigest() for x,y in data],
        'build_and_tune_seconds':manifest['build_and_tune_seconds'],'providers':{},
        'methodology':{'steps_per_window':steps,'windows':windows,'warmup_steps':2,
            'state':'initial weights and empty AdamW state restored before every timed window and after warmup',
            'timed':'forward + loss + backward + unscale/global clip + AdamW + stream completion',
            'excluded':'token upload, weight/state reset, logging and loss downloads',
            'cold':'first ten updates include any consumer JIT; Tensor producer build/tune time reported separately',
            'attention':'explicit controls match FP16 score/probability boundaries; SDPA controls use native fused attention',
            'gemm_accumulation':'FP32; PyTorch FP16 reduced-precision reductions disabled',
            'data':'fixed seeded synthetic token batches; throughput/correctness benchmark, no convergence claim',
            'optimizer':'AdamW beta=(0.9,0.95), eps=1e-8, lr=0.0006, matrix decay=0.1, norm decay=0, global norm limit=1',
            'precision':'FP16 compute; FP32 master weights, normalization/softmax reductions, gradients, AdamW state; scale=128'}}
    def save():
        if out:Path(out).write_text(json.dumps(report,indent=2)+'\n')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    for provider in providers:
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter();device=None
        if provider=='tensor':
            device=tx.Device();device.__enter__();model=NanoGPT(bundle,device)
            synchronize=device.synchronize;reset=model.reset
            def step(index):
                model.set_batch(*data[index]);begin=time.perf_counter();model.step();synchronize()
                elapsed=time.perf_counter()-begin;loss=model.read_loss();model.read_norm();return elapsed,loss
        else:
            attention='sdpa' if 'sdpa' in provider else 'explicit'
            model=Reference(cfg,compiled='inductor' in provider,attention=attention,fused_optimizer=provider.endswith('fused_adamw'))
            gpu_data=[(torch.from_numpy(x.astype(np.int64)).cuda(),torch.from_numpy(y.astype(np.int64)).cuda()) for x,y in data]
            synchronize=torch.cuda.synchronize;reset=model.reset
            def step(index):
                synchronize();begin=time.perf_counter()
                model.forward_backward(*gpu_data[index]);model.update();synchronize()
                elapsed=time.perf_counter()-begin;loss=float(model.loss.detach())
                if not np.isfinite(loss):raise FloatingPointError('nonfinite baseline loss')
                return elapsed,loss
        try:
            construction=time.perf_counter()-started
            cold=[step(i) for i in range(steps)]
            cold_elapsed=time.perf_counter()-started
            reset()
            for i in range(2):step(i%steps)
            reset();timings=[];trajectories=[]
            for window in range(windows):
                reset();values=[step(i) for i in range(steps)]
                timings.append([v[0] for v in values]);trajectories.append([v[1] for v in values])
                print(f'{provider} window {window+1}: {statistics.mean(timings[-1])*1000:.3f} ms/update',flush=True)
            means=[statistics.mean(values) for values in timings]
            report['providers'][provider]={'status':'passed','attention_backend':attention if not device else 'explicit',
                'optimizer_backend':'Tensor fused kernels' if device else 'PyTorch fused AdamW' if provider.endswith('fused_adamw') else 'PyTorch single-tensor AdamW',
                'median_seconds_per_update':statistics.median(means),
                'tokens_per_second':cfg.rows/statistics.median(means),'window_means_seconds':means,
                'step_seconds':timings,'loss_trajectories':trajectories,'construction_seconds':construction,
                'cold_step_seconds':[v[0] for v in cold],'cold_losses':[v[1] for v in cold],
                'first_ten_seconds_from_construction':cold_elapsed,
                'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            if device:
                report['providers'][provider].update(owned_device_bytes=model.allocated_bytes,
                    first_ten_seconds_including_producer=manifest['build_and_tune_seconds']+cold_elapsed,
                    launches_per_update=model.library.launch_count/(steps*(windows+1)+2))
            save()
        finally:
            if device:device.__exit__(None,None,None)
            del model;gc.collect();torch.cuda.empty_cache()
    if 'tensor' in report['providers']:
        expected=report['providers']['tensor']['loss_trajectories'][0]
        for name,value in report['providers'].items():
            delta=max(abs(a-b) for trajectory in value['loss_trajectories'] for a,b in zip(expected,trajectory))
            value['independent_maximum_loss_difference']=delta
        if 'torch_eager_explicit' in report['providers']:
            for trajectory in report['providers']['torch_eager_explicit']['loss_trajectories']:
                np.testing.assert_allclose(trajectory,expected,rtol=0.001,atol=0.002)
            report['independent_eager_loss_trajectory_check']='passed'
    report['status']='passed';save();return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--windows',type=int,default=5)
    parser.add_argument('--providers',nargs='+')
    args=parser.parse_args();args.out.parent.mkdir(parents=True,exist_ok=True)
    benchmark(args.bundle,windows=args.windows,out=args.out,providers=args.providers)

if __name__=='__main__':main()
