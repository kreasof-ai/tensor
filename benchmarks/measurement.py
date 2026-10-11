"""Timestamp measurement and prepared binding for generic WebGPU benchmarks."""

import numpy as np
import wgpu
from wgpu.backends.wgpu_native.extras import write_timestamp
from tensor.runtime.abi import BoundCall

class Timer:
    def __init__(self,device,plan,period=10):
        self.device=device;self.plan=plan;self.period=period
        self.query=device._gpu.create_query_set(type='timestamp',count=2)
        self.result=device._gpu.create_buffer(size=16,usage=wgpu.BufferUsage.QUERY_RESOLVE|wgpu.BufferUsage.COPY_SRC)
    def sample(self,dispatches=1):
        pipeline,group,grid=self.plan.nodes[0];gpu=self.device._gpu
        encoder=gpu.create_command_encoder();compute=encoder.begin_compute_pass()
        write_timestamp(compute,self.query,0);compute.set_pipeline(pipeline);compute.set_bind_group(0,group)
        for _ in range(dispatches):compute.dispatch_workgroups(*grid)
        write_timestamp(compute,self.query,1);compute.end()
        encoder.resolve_query_set(self.query,0,2,self.result,0);gpu.queue.submit([encoder.finish()])
        ticks=np.frombuffer(gpu.queue.read_buffer(self.result),dtype=np.uint64)
        return int(ticks[1]-ticks[0])*self.period*1e-9
    def close(self):self.result.destroy();self.query.destroy()

def bind(device,kernel,args):
    values,symbols,launch=kernel._bind(args,{},include_outputs=True)
    return device.prepare_plan([(kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True))])
