"""Benchmark-only timestamp/host profiling of the installed WebGPU LFM2 plan.

Requests native timestamp features only on this profiling device. Per-dispatch
timestamps can perturb scheduling; ordinary matched forward timings remain the
acceptance benchmark. Native results are ticks; --timestamp-period-ns must be
the adapter's Vulkan timestampPeriod (vulkaninfo), not a wall-clock estimate.
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time
import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import write_timestamp
from tensor.providers.webgpu import Device
from tensor_llm import LFM2


class TimestampAdapter:
    def __init__(self,adapter):self.adapter=adapter
    def __getattr__(self,name):return getattr(self.adapter,name)
    def request_device_sync(self,**kwargs):
        kwargs['required_features']+=['timestamp-query','timestamp-query-inside-passes']
        return self.adapter.request_device_sync(**kwargs)


def profile(model,bundle,out,repeats=10,timestamp_period_ns=0):
    if timestamp_period_ns<=0:raise ValueError('requires the measured Vulkan timestampPeriod')
    device=Device();device._adapter=TimestampAdapter(device._adapter)
    with device,LFM2(model,bundle,device,context=512) as engine:
        prompt=engine.tokenizer.chat('What is 2 + 2?');rows=[]
        for r in (1,32):
            plan=engine.prepared[r];n=len(plan.nodes)
            query=device._gpu.create_query_set(type='timestamp',count=2*n)
            result=device._gpu.create_buffer(size=16*n,usage=wgpu.BufferUsage.QUERY_RESOLVE|wgpu.BufferUsage.COPY_SRC)
            elapsed=[];host=[];whole=[]
            for mode in ('host','whole','dispatch'):
                samples=[]
                for repeat in range(repeats+1):
                    engine.reset();engine.forward(np.resize(prompt,128 if r==1 else 96))
                    start=time.perf_counter();device.synchronize();sync=time.perf_counter()-start
                    start=time.perf_counter()
                    engine._write(engine.workspaces[r]['tokens'],np.resize(prompt,r).astype(np.int32))
                    engine._write(engine.control,np.array([engine.position,r],np.int32))
                    upload=time.perf_counter()-start
                    start=time.perf_counter()
                    if mode=='host':plan.launch()
                    else:
                        encoder=device._gpu.create_command_encoder();compute=encoder.begin_compute_pass()
                        if mode=='whole':write_timestamp(compute,query,0)
                        for i,(pipeline,group,grid) in enumerate(plan.nodes):
                            if mode=='dispatch':write_timestamp(compute,query,2*i)
                            compute.set_pipeline(pipeline);compute.set_bind_group(0,group)
                            compute.dispatch_workgroups(*grid)
                            if mode=='dispatch':write_timestamp(compute,query,2*i+1)
                        if mode=='whole':write_timestamp(compute,query,1)
                        compute.end();encoder.resolve_query_set(query,0,2*n if mode=='dispatch' else 2,result,0)
                        device._gpu.queue.submit([encoder.finish()])
                    encode=time.perf_counter()-start
                    start=time.perf_counter();logits=engine.logits.to_numpy();read=time.perf_counter()-start
                    if not np.isfinite(logits).all():raise AssertionError('nonfinite profile output')
                    if mode=='host':value={'pre_sync_ms':sync*1000,'upload_ms':upload*1000,'encode_submit_ms':encode*1000,'completion_logits_ms':read*1000}
                    else:
                        data=np.frombuffer(device._gpu.queue.read_buffer(result),dtype=np.uint64)
                        value=(data[1::2]-data[::2]).astype(np.float64)*timestamp_period_ns/1e6
                        if mode=='whole':value=float(value[0])
                    if repeat:samples.append(value)
                if mode=='host':host={key:statistics.median(v[key] for v in samples) for key in samples[0]}
                elif mode=='whole':whole=statistics.median(samples)
                else:elapsed=np.median(samples,axis=0).tolist()
            kinds={id(engine.kernels[key]):record['kind'] for key,record in engine.manifest['kernels'].items()}
            weights={id(buffer):name for name,buffer in engine.weights.items()};nodes=[];totals=defaultdict(float)
            for i,((kernel,call),ms) in enumerate(zip(plan.calls,elapsed)):
                name=kinds[id(kernel)];totals[name]+=ms
                nodes.append({'index':i,'kind':name,'weight':next((weights[id(b)] for b in call.storage if id(b) in weights),None),'gpu_ms':ms})
            row={'rows':r,'position':128 if r==1 else 96,'dispatches':n,'host_median':host,'whole_gpu_ms':whole,
                 'instrumented_dispatch_sum_ms':sum(elapsed),'gpu_by_kind_ms':dict(sorted(totals.items(),key=lambda p:-p[1])),
                 'nodes':nodes}
            rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='nodes'},indent=2),flush=True)
            result.destroy();query.destroy()
        report={'adapter':device.info,'timestamp_period_ns':timestamp_period_ns,'repeats':repeats,'profile':rows,'protocol':'same-pass native timestamp instrumentation; each sample replays a fresh prefix; host completion includes GPU execution'}
        Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--repeats',type=int,default=10)
    p.add_argument('--timestamp-period-ns',required=True,type=float)
    a=p.parse_args();profile(a.model,a.bundle,a.out,a.repeats,a.timestamp_period_ns)
