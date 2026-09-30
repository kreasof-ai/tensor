"""Explicit correctness-checked tuning of precompiled Tensor kernel candidates."""
from __future__ import annotations
import ctypes as c
import statistics
import time
import numpy as np


def measure_cuda(device, callback, *, warmup=3, samples=7, repeats=10):
    """Return CUDA-event and completed-call samples for an allocation-free callback."""
    if device.info['provider'] != 'cuda':
        raise ValueError('CUDA event measurement requires the CUDA provider')
    if any(type(v) is not int or v < 1 for v in (samples,repeats)) or type(warmup) is not int or warmup < 0:
        raise ValueError('invalid measurement counts')
    begin,end=c.c_void_p(),c.c_void_p()
    device.driver.call('cuEventCreate',c.byref(begin),0)
    try:
        device.driver.call('cuEventCreate',c.byref(end),0)
        for _ in range(warmup): callback()
        device.synchronize()
        gpu,completed=[],[]
        for _ in range(samples):
            start=time.perf_counter()
            device.driver.call('cuEventRecord',begin,device.stream)
            for _ in range(repeats): callback()
            device.driver.call('cuEventRecord',end,device.stream)
            device.driver.call('cuEventSynchronize',end)
            completed.append((time.perf_counter()-start)/repeats)
            value=c.c_float()
            device.driver.call('cuEventElapsedTime',c.byref(value),begin,end)
            gpu.append(value.value/1000/repeats)
        return {'gpu_seconds':gpu,'completed_seconds':completed,
                'median_gpu_seconds':statistics.median(gpu),
                'median_completed_seconds':statistics.median(completed),
                'warmup':warmup,'samples':samples,'repeats':repeats}
    finally:
        device.synchronize()
        if end.value: device.driver.call('cuEventDestroy_v2',end)
        device.driver.call('cuEventDestroy_v2',begin)


def tune(candidates, inputs, output, reference, *, rtol=0.02, atol=0.002, checks=()):
    """Select a schedule by measured GPU latency, rejecting incorrect candidates.

    candidates maps configuration labels to loaded Executables. The caller owns
    compilation, the search space and an independently validated reference.
    Inputs/output are already allocated; transfers stay outside the timing.
    """
    if not candidates: raise ValueError('tuning needs candidates')
    reference=np.asarray(reference)
    if not np.isfinite(reference).all(): raise ValueError('tuning reference must be finite')
    results={}
    for name,kernel in candidates.items():
        try:
            kernel.launch(*inputs,output)
            actual=output.to_numpy()
            np.testing.assert_allclose(actual,reference,rtol=rtol,atol=atol)
            for buffer,expected in checks:
                np.testing.assert_allclose(buffer.to_numpy(),expected,rtol=rtol,atol=atol)
            results[name]={'status':'passed','timing':measure_cuda(kernel.device,lambda:kernel.launch(*inputs,output)),
                           'maximum_absolute_error':float(np.max(np.abs(actual.astype(np.float32)-reference.astype(np.float32))))}
        except (AssertionError,ValueError) as error:
            results[name]={'status':'rejected','reason':str(error)}
    valid=[name for name,result in results.items() if result['status']=='passed']
    if not valid: raise ValueError('all tuning candidates failed correctness')
    selected=min(valid,key=lambda name:results[name]['timing']['median_gpu_seconds'])
    return {'selected':selected,'candidates':results,'rtol':rtol,'atol':atol}
