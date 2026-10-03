"""Validate fused searched FFNs on independent inputs before model benchmarking."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,json,statistics
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.webgpu import Device
from tensor_llm import GGUF
from tensor_llm.model import webgpu_parameters
from tensor_llm.webgpu_kernels import source
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,Timer,bind,oracle,check


def run(model,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);gguf=GGUF(model)
    gate=np.array(gguf.array('blk.0.ffn_gate.weight',dtype=np.float16),copy=True)
    up=np.array(gguf.array('blk.0.ffn_up.weight',dtype=np.float16),copy=True)
    o,k=gate.shape;r=32
    params=webgpu_parameters('ffn',dict(r=r,k=k,o=o,type=1),'searched')
    specs={'baseline':dict(r=r,k=k,o=o,type=1,sg=True),
           'searched_m16':{**params,'tile':(16,8,64)},'searched':params}
    # Fusion doubles accumulators and scratch: measure its actual effect.
    results={};device=Device();device._adapter=TimestampAdapter(device._adapter)
    with device:
        args=(device.full(r*k,0),device.from_numpy(gate.ravel()),device.from_numpy(up.ravel()),device.full(r*o,np.nan))
        for name,p in specs.items():
            path=out/(name+'.py');artifact=path.with_suffix('.tbin')
            path.write_text(source('ffn',p))
            artifact.unlink(missing_ok=True)
            tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache')
            kernel=device.load(artifact);plan=bind(device,kernel,args);validation=[]
            for seed,scale in ((29,.01),(101,.01),(202,1),(303,1e-5)):
                inputs=(np.random.default_rng(seed).normal(size=(r,k))*scale).astype(np.float32)
                g,gb=oracle(inputs,gate);u,ub=oracle(inputs,up)
                silu=g/(1+np.exp(-g));expected=silu*u
                bound=np.abs(u)*1.1*gb+np.abs(silu)*ub+1.1*gb*ub+np.abs(expected)*2e-6+1e-10
                device._gpu.queue.write_buffer(args[0]._storage,0,inputs.tobytes())
                device._gpu.queue.write_buffer(args[-1]._storage,0,np.full(r*o,np.nan,np.float32).tobytes())
                plan.launch();actual=args[-1].to_numpy().reshape(r,o)
                validation.append({'seed':seed,'scale':scale,**check(actual,expected,bound)})
            # Time the ordinary search fixture, rather than the last subnormal
            # validation case: rounding branches can depend on operand values.
            inputs=(np.random.default_rng(29).normal(size=(r,k))*.01).astype(np.float32)
            device._gpu.queue.write_buffer(args[0]._storage,0,inputs.tobytes())
            timer=Timer(device,plan)
            for _ in range(3):timer.sample(100)
            samples=[timer.sample(20)/20 for _ in range(7)]
            results[name]={'parameters':p,'validation':validation,'gpu_samples_seconds':samples,
                           'median_gpu_seconds':statistics.median(samples)}
            print(name,results[name],flush=True)
            timer.close();plan.close();kernel._dispose()
        for arg in args:arg.release()
        report={'status':'passed','adapter':device.info,'records':results,'timing_input':{'seed':29,'scale':.01},
                'protocol':'3 warmup batches of 100; median of 7 batches of 20; GPU timestamps at 10 ns period'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model,a.out)
