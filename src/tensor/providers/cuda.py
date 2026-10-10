"""Small CUDA runtime for trusted Tensor cubins, independent of the compiler."""

from __future__ import annotations

import ctypes as c
import os
from pathlib import Path

from tensor.artifacts.format import ArtifactError, read_artifact
from tensor.runtime import Buffer, Executable, Session, TensorRuntimeError, bench
from tensor.runtime.abi import CAPABILITIES, StreamDescriptor, check_requirement


class CudaError(TensorRuntimeError):
    pass


class CudaUnavailable(CudaError):
    pass


class _Driver:
    def __init__(self) -> None:
        if c.sizeof(c.c_void_p) != 8:
            raise CudaUnavailable("CUDA requires 64-bit Python")
        try:
            self.lib = c.WinDLL("nvcuda.dll") if os.name == "nt" else c.CDLL("libcuda.so.1")
        except OSError as exc:
            raise CudaUnavailable("NVIDIA CUDA driver unavailable") from exc
        ptr, integer, uint, size, address = c.c_void_p, c.c_int, c.c_uint, c.c_size_t, c.c_uint64
        signatures = {
            "cuInit": [uint], "cuDeviceGetCount": [c.POINTER(integer)],
            "cuDeviceGet": [c.POINTER(integer), integer],
            "cuDeviceGetAttribute": [c.POINTER(integer), integer, integer],
            "cuDeviceGetName": [ptr, integer, integer],
            "cuCtxGetCurrent": [c.POINTER(ptr)], "cuCtxSetCurrent": [ptr],
            "cuDevicePrimaryCtxRetain": [c.POINTER(ptr), integer],
            "cuDevicePrimaryCtxRelease_v2": [integer],
            "cuStreamCreate": [c.POINTER(ptr), uint], "cuStreamSynchronize": [ptr],
            "cuStreamDestroy_v2": [ptr],
            "cuStreamGetCtx": [ptr, c.POINTER(ptr)],
            "cuEventCreate": [c.POINTER(ptr), uint], "cuEventRecord": [ptr, ptr],
            "cuEventSynchronize": [ptr], "cuEventElapsedTime": [c.POINTER(c.c_float), ptr, ptr],
            "cuEventDestroy_v2": [ptr], "cuStreamWaitEvent": [ptr, ptr, uint],
            "cuPointerGetAttribute": [ptr, integer, address],
            "cuMemAlloc_v2": [c.POINTER(address), size], "cuMemFree_v2": [address],
            "cuMemcpyHtoD_v2": [address, ptr, size], "cuMemcpyDtoH_v2": [ptr, address, size],
            "cuMemcpyDtoD_v2": [address, address, size],
            "cuModuleLoadData": [c.POINTER(ptr), ptr], "cuModuleUnload": [ptr],
            "cuModuleGetFunction": [c.POINTER(ptr), ptr, c.c_char_p],
            "cuFuncSetAttribute": [ptr, integer, integer],
            "cuLaunchKernel": [ptr, uint, uint, uint, uint, uint, uint, uint, ptr,
                               c.POINTER(ptr), c.POINTER(ptr)],
            "cuGetErrorName": [integer, c.POINTER(c.c_char_p)],
            "cuGetErrorString": [integer, c.POINTER(c.c_char_p)],
        }
        for name, args in signatures.items():
            try:
                function = getattr(self.lib, name)
            except AttributeError as exc:
                raise CudaUnavailable(f"CUDA driver is missing {name}") from exc
            function.argtypes = args
            function.restype = integer
        self.call("cuInit", 0)

    def call(self, name: str, *args: object) -> None:
        code = getattr(self.lib, name)(*args)
        if code:
            label, detail = c.c_char_p(), c.c_char_p()
            self.lib.cuGetErrorName(code, c.byref(label))
            self.lib.cuGetErrorString(code, c.byref(detail))
            error = CudaError(f"{name}: {(label.value or b'CUDA_ERROR').decode()} ({code}): "
                              f"{(detail.value or b'').decode()}")
            error.code = code
            raise error

    def device_info(self, ordinal: int) -> tuple[int, dict]:
        count, device = c.c_int(), c.c_int()
        self.call("cuDeviceGetCount", c.byref(count))
        if not 0 <= ordinal < count.value:
            raise CudaUnavailable(f"CUDA device {ordinal} unavailable ({count.value} detected)")
        self.call("cuDeviceGet", c.byref(device), ordinal)
        major, minor = c.c_int(), c.c_int()
        self.call("cuDeviceGetAttribute", c.byref(major), 75, device)
        self.call("cuDeviceGetAttribute", c.byref(minor), 76, device)
        name = c.create_string_buffer(256)
        self.call("cuDeviceGetName", name, len(name), device)
        return device.value, {"ordinal": ordinal, "name": name.value.decode(errors="replace"),
                              "arch": f"sm_{major.value}{minor.value}"}




class Device(Session):
    device_type = 2
    error = CudaError
    capabilities = CAPABILITIES | {"events", "async_launch", "external_streams", "gpu_dlpack", "bfloat16_storage"}

    """Retain the CUDA primary context and own or borrow a stream for the session."""

    def __init__(self, ordinal: int = 0, *, stream: int | None = None):
        if stream is not None and (type(stream) is not int or not 0 <= stream < 1 << 64):
            raise ValueError("stream must be a CUDA stream integer handle")
        self._external_stream = stream
        self.owns_stream = stream is None
        self.driver = _Driver()
        self.ordinal = ordinal
        self.device, self.info = self.driver.device_info(ordinal)
        self.info["provider"] = "cuda"
        self.limits = {}
        for field, attributes in (("block", (2, 3, 4)), ("grid", (5, 6, 7))):
            self.limits[field] = []
            for attribute in attributes:
                limit = c.c_int()
                self.driver.call("cuDeviceGetAttribute", c.byref(limit), attribute, self.device)
                self.limits[field].append(limit.value)
        self.context, self.previous, self.stream = c.c_void_p(), c.c_void_p(), c.c_void_p()
        self._buffers: set[Buffer] = set()
        self._modules: list[c.c_void_p] = []
        self._consumers: set[int] = set()
        self._events = set()
        self._generation = 0
        self._open = False
        self._stream_active = False

    def __enter__(self) -> Device:
        if self._open:
            raise CudaError("device session is already open")
        try:
            self.driver.call("cuCtxGetCurrent", c.byref(self.previous))
            self.driver.call("cuDevicePrimaryCtxRetain", c.byref(self.context), self.device)
            self.driver.call("cuCtxSetCurrent", self.context)
            if self.owns_stream:
                self.driver.call("cuStreamCreate", c.byref(self.stream), 1)
            else:
                self.stream = self._foreign_stream(self._external_stream)
            self._stream_active = True
            self._generation += 1
            self._start_session()
            self._open = True
            return self
        except BaseException:
            self._cleanup()
            raise

    def _check(self) -> None:
        if not self._open:
            raise CudaError("CUDA device session is closed")
        self.driver.call("cuCtxSetCurrent", self.context)

    def synchronize(self) -> None:
        self._check()
        self.driver.call("cuStreamSynchronize", self.stream)
        for handle in self._consumers:
            self.driver.call("cuStreamSynchronize", c.c_void_p(handle))

    def _foreign_stream(self, handle: int) -> c.c_void_p:
        if type(handle) is not int or not 0 <= handle < 1 << 64:
            raise ValueError("stream must be a CUDA stream integer handle")
        stream, context = c.c_void_p(handle), c.c_void_p()
        self.driver.call("cuStreamGetCtx", stream, c.byref(context))
        if context.value != self.context.value:
            raise ValueError("foreign stream must belong to this device's CUDA primary context")
        return stream

    def _order_streams(self, producer, consumer) -> None:
        if producer.value == consumer.value:
            return
        event = c.c_void_p()
        self.driver.call("cuEventCreate", c.byref(event), 2)  # Disable timing.
        try:
            self.driver.call("cuEventRecord", event, producer)
            self.driver.call("cuStreamWaitEvent", consumer, event, 0)
        finally:
            self.driver.call("cuEventDestroy_v2", event)

    def wait_for(self, producer_stream: int) -> None:
        """Order this stream after work already queued on a foreign stream."""
        self._check()
        self._order_streams(self._foreign_stream(producer_stream), self.stream)

    def handoff(self, consumer_stream: int) -> None:
        """Order a foreign consumer stream after work already queued here."""
        self._check()
        self._order_streams(self.stream, self._foreign_stream(consumer_stream))
        self._consumers.add(consumer_stream)

    def record_event(self):
        from tensor.runtime import Event
        self._check()
        handle = c.c_void_p()
        self.driver.call("cuEventCreate", c.byref(handle), 2)
        try:
            self.driver.call("cuEventRecord", handle, self.stream)
        except BaseException:
            self.driver.call("cuEventDestroy_v2", handle)
            raise
        event = Event(self, handle)
        self._events.add(event)
        return event

    def wait(self, event):
        self._check()
        event._check(self)
        self._check()  # event validation can activate the producer's context.
        self.driver.call("cuStreamWaitEvent", self.stream, event.handle, 0)

    def _dispose_event(self, event):
        self.driver.call("cuEventDestroy_v2", event.handle)

    def stream_descriptor(self):
        return StreamDescriptor(self.device_type, self.ordinal, self.stream.value or 0)

    def _download(self, buffer):
        from tensor.runtime.dtypes import decode_bfloat16
        result = self._download_storage(buffer)
        return decode_bfloat16(result) if str(buffer.dtype) == "bfloat16" else result

    def _download_storage(self, buffer):
        import numpy as np
        from tensor.runtime.dtypes import storage_dtype
        result = np.empty(buffer.shape, dtype=storage_dtype(buffer.dtype))
        self.driver.call("cuMemcpyDtoH_v2", c.c_void_p(result.ctypes.data), buffer.pointer, buffer.nbytes)
        return result

    def _dispose_buffer(self, buffer):
        if buffer._owner is not None:
            buffer._owner.release()
            buffer._owner = None
        else:
            self.driver.call("cuMemFree_v2", buffer.pointer)

    def _launch(self, executable, call):
        # Bind on the host first; activate the provider context once immediately
        # before submission, including after user-defined scalar conversions.
        self._check()
        descriptor = call.descriptor
        shared = descriptor.shared_memory_bytes
        if shared > executable._shared_limit:
            self.driver.call("cuFuncSetAttribute", executable.function, 8, shared)
            executable._shared_limit = shared
        self.driver.call("cuLaunchKernel", executable.function,
                         *descriptor.grid, *descriptor.block, shared,
                         c.c_void_p(descriptor.stream.handle), call.cuda_parameters(), None)

    def empty(self, shape, dtype="float32") -> Buffer:
        import numpy as np

        self._check()
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if not shape or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("buffer shape must contain positive integers")
        from tensor.runtime.dtypes import dtype as normalize_dtype
        dtype = normalize_dtype(dtype)
        if dtype.hasobject or dtype.itemsize < 1:
            raise ValueError("buffer dtype must have a fixed byte size")
        nbytes = dtype.itemsize
        for extent in shape:
            nbytes *= extent
        if nbytes >= 1 << (c.sizeof(c.c_size_t)*8):
            raise ValueError("buffer allocation size exceeds size_t")
        pointer = c.c_uint64()
        self.driver.call("cuMemAlloc_v2", c.byref(pointer), nbytes)
        buffer = Buffer(self, pointer.value, shape, dtype)
        self._buffers.add(buffer)
        return buffer

    def from_numpy(self, array, dtype=None) -> Buffer:
        import numpy as np
        from tensor.runtime.dtypes import encode_bfloat16

        host = encode_bfloat16(array) if str(dtype) == "bfloat16" else np.ascontiguousarray(array, dtype=dtype)
        buffer = self.empty(host.shape, dtype if dtype is not None else host.dtype)
        try:
            self.driver.call("cuMemcpyHtoD_v2", buffer.pointer,
                             c.c_void_p(host.ctypes.data), host.nbytes)
            # The launch stream is non-blocking; complete the default-stream copy first.
            self.driver.call("cuStreamSynchronize", None)
            return buffer
        except BaseException:
            buffer.release()
            raise

    def from_dlpack(self, source) -> Buffer:
        import numpy as np

        self._check()
        if not hasattr(source, "__dlpack_device__") or not hasattr(source, "__dlpack__"):
            raise TypeError("DLPack source must implement __dlpack__ and __dlpack_device__")
        kind, ordinal = source.__dlpack_device__()
        if kind == 1:
            return self.from_numpy(np.from_dlpack(source))
        if kind != 2 or ordinal != self.ordinal:
            raise BufferError("DLPack source must be on the CPU or this CUDA device")
        from tensor.runtime.dlpack import borrow

        # DLPack uses 1 for the legacy default stream, rather than CUDA's null handle.
        owner, pointer, shape, dtype = borrow(source, stream=self.stream.value or 1, ordinal=self.ordinal)
        try:
            context, actual_device = c.c_void_p(), c.c_int()
            self.driver.call("cuPointerGetAttribute", c.byref(context), 1, pointer)
            self.driver.call("cuPointerGetAttribute", c.byref(actual_device), 9, pointer)
            if context.value != self.context.value or actual_device.value != self.ordinal:
                raise BufferError("DLPack pointer must belong to this device's primary context")
            buffer = Buffer(self, pointer, shape, dtype, owner=owner)
            self._buffers.add(buffer)
            return buffer
        except BaseException:
            self.synchronize()
            owner.release()
            raise

    def full(self, shape, fill_value, dtype="float32") -> Buffer:
        import numpy as np

        return self.from_numpy(np.full(shape, fill_value, dtype="float32" if str(dtype) == "bfloat16" else dtype), dtype=dtype)

    def zeros(self, shape, dtype="float32") -> Buffer:
        return self.full(shape, 0, dtype)

    def ones(self, shape, dtype="float32") -> Buffer:
        return self.full(shape, 1, dtype)

    def randn(self, shape, dtype="float32", *, seed=None) -> Buffer:
        import numpy as np

        return self.from_numpy(np.random.default_rng(seed).standard_normal(shape), dtype=dtype)

    def arange(self, stop: int, dtype="float32") -> Buffer:
        import numpy as np

        return self.from_numpy(np.arange(stop, dtype="float32" if str(dtype) == "bfloat16" else dtype), dtype=dtype)

    def load(self, artifact: str | Path) -> Executable:
        self._check()
        manifest, files = read_artifact(artifact)
        if manifest.get("provider", "cuda") != "cuda":
            raise ArtifactError("artifact requires a different runtime provider")
        if "runtime_abi" in manifest:
            check_requirement(manifest["runtime_abi"], self.capabilities)
        from tensor.runtime.cuda_target import matches_device
        if not matches_device(manifest["target"], self.info["arch"]):
            raise ArtifactError(f"artifact target {manifest['target']} does not match device {self.info['arch']}")
        module, function = c.c_void_p(), c.c_void_p()
        image = c.create_string_buffer(files["kernel.cubin"])
        self.driver.call("cuModuleLoadData", c.byref(module), image)
        try:
            self.driver.call("cuModuleGetFunction", c.byref(function), module,
                             manifest["entrypoint"].encode())
        except BaseException:
            self.driver.call("cuModuleUnload", module)
            raise
        self._modules.append(module)
        return Executable(self, manifest, module, function, image)

    def _unload_executable(self, executable):
        self.driver.call("cuModuleUnload", executable.module)
        self._modules.remove(executable.module)

    def _cleanup(self) -> None:
        if any(getattr(buffer, "_dlpack_exports", 0) for buffer in self._buffers):
            raise CudaError("release DLPack consumers before closing this device session")
        errors = []

        def call(name, *values):
            try:
                self.driver.call(name, *values)
            except CudaError as exc:
                errors.append(exc)

        if self.context.value:
            call("cuCtxSetCurrent", self.context)
        if self._stream_active:
            call("cuStreamSynchronize", self.stream)
        for handle in self._consumers:
            call("cuStreamSynchronize", c.c_void_p(handle))
        for event in list(self._events):
            try:
                event.release()
            except CudaError as exc:
                errors.append(exc)
        for buffer in list(self._buffers):
            try:
                call("cuCtxSetCurrent", self.context)
                buffer._dispose()
            except (CudaError, RuntimeError) as exc:
                errors.append(exc)
        for executable in list(getattr(self, "_executables", {}).values()):
            try:
                call("cuCtxSetCurrent", self.context)
                executable._dispose()
            except CudaError as exc:
                errors.append(exc)
        if self._stream_active and self.owns_stream:
            call("cuStreamDestroy_v2", self.stream)
        if self.context.value:
            call("cuDevicePrimaryCtxRelease_v2", self.device)
            call("cuCtxSetCurrent", self.previous)
        self._open = False
        self._stream_active = False
        self._buffers.clear()
        self._modules.clear()
        self._consumers.clear()
        self._events.clear()
        self.stream, self.context = c.c_void_p(), c.c_void_p()
        if errors:
            raise errors[0]

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._cleanup()
        except CudaError:
            if exc_type is None:
                raise
