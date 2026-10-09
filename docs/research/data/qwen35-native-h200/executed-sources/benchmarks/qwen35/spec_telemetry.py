"""Low-overhead NVML telemetry for native cohort measurements."""
import ctypes as ct
import json
from pathlib import Path
import threading
import time


class Memory(ct.Structure):
    _fields_=[('total',ct.c_ulonglong),('free',ct.c_ulonglong),('used',ct.c_ulonglong)]


class Utilization(ct.Structure):
    _fields_=[('gpu',ct.c_uint),('memory',ct.c_uint)]


class Monitor:
    def __init__(self,out,*,interval=.25):
        self.out=Path(out);self.interval=interval;self.stop=threading.Event();self.rows=[];self.error=None

    def __enter__(self):
        try:
            self.lib=ct.CDLL('libnvidia-ml.so.1');self.handle=ct.c_void_p()
            for name,args in (('nvmlInit_v2',[]),('nvmlShutdown',[]),
                    ('nvmlDeviceGetHandleByIndex_v2',[ct.c_uint,ct.POINTER(ct.c_void_p)]),
                    ('nvmlDeviceGetMemoryInfo',[ct.c_void_p,ct.POINTER(Memory)]),
                    ('nvmlDeviceGetUtilizationRates',[ct.c_void_p,ct.POINTER(Utilization)])):
                f=getattr(self.lib,name);f.argtypes=args;f.restype=ct.c_int
            if self.lib.nvmlInit_v2()!=0:raise RuntimeError('NVML initialization failed')
            if self.lib.nvmlDeviceGetHandleByIndex_v2(0,ct.byref(self.handle))!=0:raise RuntimeError('NVML GPU 0 unavailable')
            self.origin=time.perf_counter();self.thread=threading.Thread(target=self.sample,daemon=True);self.thread.start()
        except Exception as error:self.error=str(error)
        return self

    def sample(self):
        while not self.stop.is_set():
            memory,util=Memory(),Utilization()
            if self.lib.nvmlDeviceGetMemoryInfo(self.handle,ct.byref(memory))==0:
                self.lib.nvmlDeviceGetUtilizationRates(self.handle,ct.byref(util))
                self.rows.append(dict(seconds=time.perf_counter()-self.origin,used_bytes=int(memory.used),
                    total_bytes=int(memory.total),gpu_utilization_percent=int(util.gpu)))
            self.stop.wait(self.interval)

    def __exit__(self,*exc):
        self.stop.set()
        if hasattr(self,'thread'):self.thread.join();self.lib.nvmlShutdown()
        self.out.parent.mkdir(parents=True,exist_ok=True)
        self.out.write_text(''.join(json.dumps(row)+'\n' for row in self.rows))

    def summary(self):
        return dict(samples=len(self.rows),error=self.error,
            peak_used_bytes=max((r['used_bytes'] for r in self.rows),default=None),
            peak_gpu_utilization_percent=max((r['gpu_utilization_percent'] for r in self.rows),default=None))
