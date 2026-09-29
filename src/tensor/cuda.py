"""Small CUDA runtime for trusted Tensor cubins, independent of the compiler."""

from __future__ import annotations

import ctypes as c
import os
import statistics
import time
from pathlib import Path

from tensor.artifact import ArtifactError, read_artifact
from tensor.signature import (SCALAR_TYPES, bind_shapes, buffer_argument,
                              evaluate, resolve_launch, resolve_shape, scalar_value)


class CudaError(RuntimeError):
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
            "cuEventDestroy_v2": [ptr], "cuStreamWaitEvent": [ptr, ptr, uint],
            "cuPointerGetAttribute": [ptr, integer, address],
            "cuMemAlloc_v2": [c.POINTER(address), size], "cuMemFree_v2": [address],
            "cuMemcpyHtoD_v2": [address, ptr, size], "cuMemcpyDtoH_v2": [ptr, address, size],
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


class Buffer:
    """Contiguous owned or DLPack-borrowed allocation, valid within its session."""

    def __init__(self, device: Device, pointer: int, shape: tuple[int, ...], dtype: object, *, owner=None):
        self.device, self.pointer, self.shape, self.dtype = device, pointer, shape, dtype
        self._owner = owner
        self.owned = owner is None
        self.nbytes = self.dtype.itemsize
        for extent in shape:
            self.nbytes *= extent
        self.strides = []
        stride = self.dtype.itemsize
        for extent in reversed(shape):
            self.strides.insert(0, stride)
            stride *= extent
        self.strides = tuple(self.strides)
        self._released = False

    def _check(self) -> None:
        self.device._check()
        if self._released:
            raise CudaError("buffer has been released")

    def to_numpy(self):
        import numpy as np

        self._check()
        result = np.empty(self.shape, dtype=self.dtype)
        self.device.synchronize()
        self.device.driver.call("cuMemcpyDtoH_v2", c.c_void_p(result.ctypes.data),
                                self.pointer, self.nbytes)
        return result

    def to_bytes(self) -> bytes:
        """Return a host-side byte snapshot of this device buffer."""
        return self.to_numpy().tobytes()

    def release(self) -> None:
        if not self._released:
            self._check()
            self.device.synchronize()
            self._dispose()

    def _dispose(self) -> None:
        if self._released:
            return
        if self._owner is not None:
            self._owner.release()
            self._owner = None
        else:
            self.device.driver.call("cuMemFree_v2", self.pointer)
        self._released = True
        self.device._buffers.remove(self)


class Executable:
    def __init__(self, device: Device, manifest: dict, module: c.c_void_p,
                 function: c.c_void_p, image: c.Array):
        self.device, self.manifest, self.module, self.function, self._image = (
            device, manifest, module, function, image)
        self._shared_limit = 0

    def _bind(self, args: tuple, kwargs: dict, *, include_outputs: bool):
        descriptors = [item for item in self.manifest["arguments"]
                       if include_outputs or item["name"] not in self.manifest.get("outputs", [])]
        if len(args) > len(descriptors):
            raise ValueError(f"kernel expects at most {len(descriptors)} arguments, received {len(args)}")
        values = dict(zip((item["name"] for item in descriptors), args))
        names = {item["name"] for item in descriptors}
        dimensions = {}
        for name, value in kwargs.items():
            if name in values:
                raise ValueError(f"argument {name} supplied twice")
            if name in names:
                values[name] = value
            elif name in self.manifest.get("symbols", {}):
                dimensions[name] = value
            else:
                raise ValueError(f"unknown kernel argument {name}")
        missing = names - values.keys()
        if missing:
            raise ValueError(f"missing kernel arguments: {sorted(missing)}")
        for descriptor in descriptors:
            if buffer_argument(descriptor):
                value = values[descriptor["name"]]
                if not isinstance(value, Buffer):
                    raise TypeError(f"{descriptor['name']} must be a Tensor Buffer")
                value._check()
                if value.device is not self.device:
                    raise ValueError(f"{descriptor['name']} belongs to a different CUDA device session")
                # V1 omitted alignment metadata; retain CUDA allocation alignment there.
                alignment = descriptor.get("alignment", 256)
                if value.pointer % alignment:
                    raise ValueError(f"{descriptor['name']} needs {alignment}-byte pointer alignment")
        symbols, values = bind_shapes(self.manifest, values, dimensions)
        launch = resolve_launch(self.manifest["launch"], symbols)
        for field in ("grid", "block"):
            if any(value > limit for value, limit in zip(launch[field], self.device.limits[field])):
                raise ValueError(f"launch {field} exceeds this device's limits")
        return values, symbols, launch

    def launch(self, *arguments, **bindings) -> None:
        """Launch with all frontend arguments; symbolic dimensions are inferred."""
        self.device._check()
        values, symbols, launch = self._bind(arguments, bindings, include_outputs=True)
        abi = self.manifest.get("abi", [dict(item, kind="buffer") for item in self.manifest["arguments"]])
        holders = []
        for descriptor in abi:
            name = descriptor["name"]
            if buffer_argument(descriptor):
                holders.append(c.c_uint64(values[name].pointer))
            else:
                value = values[name] if name in values else evaluate({"var": name}, symbols)
                holders.append(SCALAR_TYPES[descriptor["dtype"]](scalar_value(value, descriptor["dtype"], name)))
        params = (c.c_void_p * len(holders))(*(c.addressof(item) for item in holders))
        shared = launch["shared_memory_bytes"]
        if shared > self._shared_limit:
            # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
            self.device.driver.call("cuFuncSetAttribute", self.function, 8, shared)
            self._shared_limit = shared
        self.device.driver.call("cuLaunchKernel", self.function,
                                *launch["grid"], *launch["block"], shared,
                                self.device.stream, params, None)

    def prepare(self, *inputs, **bindings):
        """Allocate declared outputs without launching, for repeated benchmarks."""
        self.device._check()
        outputs = self.manifest.get("outputs", [])
        if not outputs:
            raise ValueError("artifact has no declared outputs; call launch with every buffer")
        supplied, symbols, _ = self._bind(inputs, bindings, include_outputs=False)
        generated = {}
        try:
            for item in self.manifest["arguments"]:
                if item["name"] in outputs:
                    generated[item["name"]] = self.device.empty(resolve_shape(item["shape"], symbols), item["dtype"])
        except BaseException:
            for buffer in generated.values():
                buffer.release()
            raise
        supplied.update(generated)
        ordered = tuple(supplied[item["name"]] for item in self.manifest["arguments"])
        # Explicit scalars are already present in ordered; only inferred symbols go in kwargs.
        implicit = {name: value for name, value in symbols.items() if name not in supplied}
        return ordered, implicit, generated

    def __call__(self, *inputs, **bindings):
        """Allocate declared outputs, launch, and return one Buffer or a tuple."""
        ordered, dimensions, generated = self.prepare(*inputs, **bindings)
        try:
            self.launch(*ordered, **dimensions)
        except BaseException:
            for buffer in generated.values():
                buffer.release()
            raise
        outputs = self.manifest["outputs"]
        result = tuple(generated[name] for name in outputs)
        return result[0] if len(result) == 1 else result


class Device:
    """Retain the CUDA primary context and own or borrow a stream for the session."""

    def __init__(self, ordinal: int = 0, *, stream: int | None = None):
        if stream is not None and (type(stream) is not int or not 0 <= stream < 1 << 64):
            raise ValueError("stream must be a CUDA stream integer handle")
        self._external_stream = stream
        self.owns_stream = stream is None
        self.driver = _Driver()
        self.ordinal = ordinal
        self.device, self.info = self.driver.device_info(ordinal)
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

    def empty(self, shape, dtype="float32") -> Buffer:
        import numpy as np

        self._check()
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if not shape or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("buffer shape must contain positive integers")
        dtype = np.dtype(dtype)
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

    def from_numpy(self, array) -> Buffer:
        import numpy as np

        host = np.ascontiguousarray(array)
        buffer = self.empty(host.shape, host.dtype)
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
        from tensor.dlpack import borrow

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

        return self.from_numpy(np.full(shape, fill_value, dtype=dtype))

    def zeros(self, shape, dtype="float32") -> Buffer:
        return self.full(shape, 0, dtype)

    def ones(self, shape, dtype="float32") -> Buffer:
        return self.full(shape, 1, dtype)

    def randn(self, shape, dtype="float32", *, seed=None) -> Buffer:
        import numpy as np

        return self.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(dtype))

    def arange(self, stop: int, dtype="float32") -> Buffer:
        import numpy as np

        return self.from_numpy(np.arange(stop, dtype=dtype))

    def load(self, artifact: str | Path) -> Executable:
        self._check()
        manifest, files = read_artifact(artifact)
        if manifest["target"] != self.info["arch"]:
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

    def _cleanup(self) -> None:
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
        for buffer in list(self._buffers):
            try:
                call("cuCtxSetCurrent", self.context)
                buffer._dispose()
            except (CudaError, RuntimeError) as exc:
                errors.append(exc)
        for module in reversed(self._modules):
            call("cuCtxSetCurrent", self.context)
            call("cuModuleUnload", module)
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
        self.stream, self.context = c.c_void_p(), c.c_void_p()
        if errors:
            raise errors[0]

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._cleanup()
        except CudaError:
            if exc_type is None:
                raise


def bench(executable: Executable, buffers: tuple, *, warmup: int = 10,
          iters: int = 100, **bindings) -> dict:
    """Measure host launch plus stream synchronization, with preallocated buffers."""
    if type(warmup) is not int or warmup < 0 or type(iters) is not int or iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")
    device = executable.device
    for _ in range(warmup):
        executable.launch(*buffers, **bindings)
        device.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        executable.launch(*buffers, **bindings)
        device.synchronize()
        samples.append(time.perf_counter() - start)
    enqueue = []
    for _ in range(iters):
        start = time.perf_counter()
        executable.launch(*buffers, **bindings)
        enqueue.append(time.perf_counter() - start)
    device.synchronize()
    return {"warmup": warmup, "iters": iters,
            "median_launch_and_sync_seconds": statistics.median(samples),
            "min_launch_and_sync_seconds": min(samples),
            "median_host_enqueue_seconds": statistics.median(enqueue)}
