"""CUDA event timings for a fixed sequence of native calls in one graph."""
import ctypes as ct
from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall


def time_calls(device,calls,*,repeats=3,samples=5):
    if repeats<1 or samples<1:raise ValueError('positive timing repetitions required')
    bound=[];resources=[]
    for kernel,args in calls:
        storage,symbols,launch=kernel._bind(tuple(args),{},include_outputs=True)
        bound.append((kernel,BoundCall(device,kernel.manifest,storage,symbols,launch,validated=True)))
        resources.extend((kernel,*args))
    def submit():
        for _ in range(repeats):
            for kernel,call in bound:device._launch(kernel,call)
    start,end=ct.c_void_p(),ct.c_void_p()
    device.driver.call('cuEventCreate',ct.byref(start),0)
    try:
        device.driver.call('cuEventCreate',ct.byref(end),0)
        with CudaGraph(device,submit,resources=tuple(resources)) as graph:
            graph.launch();device.synchronize();result=[]
            for _ in range(samples):
                device.driver.call('cuEventRecord',start,device.stream);graph.launch()
                device.driver.call('cuEventRecord',end,device.stream);device.synchronize()
                milliseconds=ct.c_float()
                device.driver.call('cuEventElapsedTime',ct.byref(milliseconds),start,end)
                result.append(milliseconds.value/repeats)
        return result
    finally:
        if end.value:device.driver.call('cuEventDestroy_v2',end)
        device.driver.call('cuEventDestroy_v2',start)
