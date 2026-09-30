"""Driver-only CUDA Graph capture for fixed-buffer Tensor launch plans."""
from __future__ import annotations
import ctypes as c
from .cuda import CudaError


class CudaGraph:
    """Own a captured graph and validate its retained resources before replay.

The caller closes the graph before its Device session. Capture submits only
already-loaded kernels; allocation, image loading, copies and host reads must
happen outside the callback. Device-buffer contents may change between replays.
"""
    def __init__(self,device,submit,*,resources=()):
        device._check();self.device=device;self.resources=tuple(resources)
        self.graph=c.c_void_p();self.executable=c.c_void_p();self._closed=False
        ptr=c.c_void_p; integer=c.c_int
        signatures={'cuStreamBeginCapture':[ptr,integer],'cuStreamEndCapture':[ptr,c.POINTER(ptr)],
                    'cuGraphInstantiateWithFlags':[c.POINTER(ptr),ptr,c.c_ulonglong],
                    'cuGraphLaunch':[ptr,ptr],'cuGraphDestroy':[ptr],'cuGraphExecDestroy':[ptr]}
        for name,args in signatures.items():
            try:function=getattr(device.driver.lib,name)
            except AttributeError as error:raise CudaError(f'CUDA driver is missing {name}') from error
            function.argtypes=args;function.restype=integer
        self._generation=device._generation
        device.driver.call('cuStreamBeginCapture',device.stream,1) # Thread local.
        try:submit()
        except BaseException:
            # End even an invalidated capture; preserve the callback exception.
            try:device.driver.call('cuStreamEndCapture',device.stream,c.byref(self.graph))
            except CudaError:pass
            if self.graph.value:device.driver.call('cuGraphDestroy',self.graph)
            self._closed=True
            raise
        try:
            device.driver.call('cuStreamEndCapture',device.stream,c.byref(self.graph))
            device.driver.call('cuGraphInstantiateWithFlags',c.byref(self.executable),self.graph,0)
        except BaseException:
            if self.graph.value:device.driver.call('cuGraphDestroy',self.graph)
            self._closed=True
            raise

    def launch(self):
        self.device._check()
        if self._closed or self._generation!=self.device._generation:raise CudaError('CUDA graph is released or belongs to an old session')
        for resource in self.resources:
            if getattr(resource,'_released',False) or getattr(resource,'_generation',self._generation)!=self._generation:
                raise CudaError('CUDA graph references a released resource')
        self.device.driver.call('cuGraphLaunch',self.executable,self.device.stream)

    def close(self):
        if self._closed:return
        self.device._check();self.device.synchronize()
        try:self.device.driver.call('cuGraphExecDestroy',self.executable)
        finally:
            self.device.driver.call('cuGraphDestroy',self.graph);self._closed=True
            self.resources=()

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
