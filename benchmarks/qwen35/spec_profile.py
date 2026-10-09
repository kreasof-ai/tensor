"""CUDA event timings for one uncaptured chunk plan, grouped by operation."""
import ctypes as ct
import json
from pathlib import Path


def profile(executor,out,bundle):
    executor.model._check();device=executor.device;driver=device.driver
    reverse={id(v):k for k,v in executor.kernels.items()}
    kinds={k:row['kind'] for k,row in json.loads((Path(bundle)/'prefill.json').read_text())['kernels'].items()}
    pairs=[]
    try:
        for kernel,bound in executor.plan:
            start,end=ct.c_void_p(),ct.c_void_p()
            driver.call('cuEventCreate',ct.byref(start),0);driver.call('cuEventCreate',ct.byref(end),0)
            pairs.append((reverse.get(id(kernel),'target-head'),start,end))
            driver.call('cuEventRecord',start,device.stream)
            device._launch(kernel,bound)
            driver.call('cuEventRecord',end,device.stream)
        device.synchronize();records=[];groups={}
        for key,start,end in pairs:
            value=ct.c_float();driver.call('cuEventElapsedTime',ct.byref(value),start,end)
            records.append(dict(key=key,milliseconds=float(value.value)))
            kind=kinds.get(key,key)
            groups[kind]=groups.get(kind,0.)+value.value
        result=dict(total_kernel_milliseconds=sum(r['milliseconds'] for r in records),
                    groups=groups,records=records)
        Path(out).write_text(json.dumps(result,indent=2)+'\n')
        print('Verification kernel profile',sorted(groups.items(),key=lambda r:-r[1]),flush=True)
        return result
    finally:
        for _,start,end in pairs:
            if start.value:driver.call('cuEventDestroy_v2',start)
            if end.value:driver.call('cuEventDestroy_v2',end)
