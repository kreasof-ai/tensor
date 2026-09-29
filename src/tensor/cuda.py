"""Small CUDA runtime for trusted Tensor cubins, independent of the compiler."""

from __future__ import annotations

import ctypes as c
import os
import statistics
import time
from pathlib import Path

from tensor.artifact import ArtifactError, read_artifact


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
            "cuCtxCreate_v2": [c.POINTER(ptr), uint, integer], "cuCtxDestroy_v2": [ptr],
            "cuStreamCreate": [c.POINTER(ptr), uint], "cuStreamSynchronize": [ptr],
            "cuStreamDestroy_v2": [ptr],
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
    """Owned contiguous device allocation; its address is valid while the device is open."""

    def __init__(self, device: Device, pointer: int, shape: tuple[int, ...], dtype: object):
        self.device, self.pointer, self.shape, self.dtype = device, pointer, shape, dtype
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
            self.device.driver.call("cuMemFree_v2", self.pointer)
            self._released = True
            self.device._buffers.remove(self)


class Executable:
    def __init__(self, device: Device, manifest: dict, module: c.c_void_p,
                 function: c.c_void_p, image: c.Array):
        self.device, self.manifest, self.module, self.function, self._image = (
            device, manifest, module, function, image)

    def launch(self, *buffers: Buffer) -> None:
        self.device._check()
        descriptors = self.manifest["arguments"]
        if len(buffers) != len(descriptors):
            raise ValueError(f"kernel expects {len(descriptors)} buffers, received {len(buffers)}")
        for value, descriptor in zip(buffers, descriptors):
            if not isinstance(value, Buffer):
                raise TypeError(f"{descriptor['name']} must be a Tensor Buffer")
            value._check()
            if value.device is not self.device:
                raise ValueError(f"{descriptor['name']} belongs to a different CUDA device session")
            if value.shape != tuple(descriptor["shape"]) or str(value.dtype) != descriptor["dtype"]:
                raise ValueError(f"{descriptor['name']} needs shape {descriptor['shape']} and dtype {descriptor['dtype']}")
        addresses = [c.c_uint64(value.pointer) for value in buffers]
        params = (c.c_void_p * len(addresses))(*(c.addressof(item) for item in addresses))
        launch = self.manifest["launch"]
        self.device.driver.call("cuLaunchKernel", self.function,
                                *launch["grid"], *launch["block"], launch["shared_memory_bytes"],
                                self.device.stream, params, None)

    def __call__(self, *inputs: Buffer):
        """Allocate declared outputs, launch, and return one Buffer or a tuple."""
        outputs = self.manifest.get("outputs", [])
        if not outputs:
            raise ValueError("artifact has no declared outputs; call launch with every buffer")
        arguments = self.manifest["arguments"]
        input_descriptors = [item for item in arguments if item["name"] not in outputs]
        if len(inputs) != len(input_descriptors):
            raise ValueError(f"kernel expects {len(input_descriptors)} input buffers")
        supplied = dict(zip((item["name"] for item in input_descriptors), inputs))
        generated = {item["name"]: self.device.empty(item["shape"], item["dtype"])
                     for item in arguments if item["name"] in outputs}
        supplied.update(generated)
        self.launch(*(supplied[item["name"]] for item in arguments))
        result = tuple(generated[name] for name in outputs)
        return result[0] if len(result) == 1 else result


class Device:
    """Own one CUDA context and stream; use as a context manager."""

    def __init__(self, ordinal: int = 0):
        self.driver = _Driver()
        self.ordinal = ordinal
        self.device, self.info = self.driver.device_info(ordinal)
        self.context, self.previous, self.stream = c.c_void_p(), c.c_void_p(), c.c_void_p()
        self._buffers: set[Buffer] = set()
        self._modules: list[c.c_void_p] = []
        self._open = False

    def __enter__(self) -> Device:
        if self._open:
            raise CudaError("device session is already open")
        try:
            self.driver.call("cuCtxGetCurrent", c.byref(self.previous))
            self.driver.call("cuCtxCreate_v2", c.byref(self.context), 0, self.device)
            self.driver.call("cuStreamCreate", c.byref(self.stream), 1)
            self._open = True
            return self
        except BaseException:
            self._cleanup()
            raise

    def _check(self) -> None:
        if not self._open:
            raise CudaError("CUDA device session is closed")

    def synchronize(self) -> None:
        self._check()
        self.driver.call("cuStreamSynchronize", self.stream)

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

        try:
            array = np.from_dlpack(source)
        except (BufferError, RuntimeError, TypeError) as exc:
            raise ValueError("this CUDA provider accepts CPU DLPack producers; GPU borrowing needs a stream adapter") from exc
        return self.from_numpy(array)

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
            shared = manifest["launch"]["shared_memory_bytes"]
            if shared:
                # CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8.
                self.driver.call("cuFuncSetAttribute", function, 8, shared)
        except BaseException:
            self.driver.call("cuModuleUnload", module)
            raise
        self._modules.append(module)
        return Executable(self, manifest, module, function, image)

    def _cleanup(self) -> None:
        errors = []
        calls = []
        if self.stream.value:
            calls.append(("cuStreamSynchronize", self.stream))
        calls += [("cuMemFree_v2", buffer.pointer) for buffer in self._buffers if not buffer._released]
        calls += [("cuModuleUnload", module) for module in reversed(self._modules)]
        if self.stream.value:
            calls.append(("cuStreamDestroy_v2", self.stream))
        if self.context.value:
            calls.extend((("cuCtxDestroy_v2", self.context), ("cuCtxSetCurrent", self.previous)))
        for name, value in calls:
            try:
                self.driver.call(name, value)
            except CudaError as exc:
                errors.append(exc)
        self._open = False
        self._buffers.clear()
        self._modules.clear()
        self.stream, self.context = c.c_void_p(), c.c_void_p()
        if errors:
            raise errors[0]

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._cleanup()
        except CudaError:
            if exc_type is None:
                raise


def bench(executable: Executable, buffers: tuple[Buffer, ...], *, warmup: int = 10,
          iters: int = 100) -> dict:
    """Measure host launch plus stream synchronization, with preallocated buffers."""
    if type(warmup) is not int or warmup < 0 or type(iters) is not int or iters < 1:
        raise ValueError("warmup must be non-negative and iters must be positive")
    device = executable.device
    for _ in range(warmup):
        executable.launch(*buffers)
        device.synchronize()
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        executable.launch(*buffers)
        device.synchronize()
        samples.append(time.perf_counter() - start)
    enqueue = []
    for _ in range(iters):
        start = time.perf_counter()
        executable.launch(*buffers)
        enqueue.append(time.perf_counter() - start)
    device.synchronize()
    return {"warmup": warmup, "iters": iters,
            "median_launch_and_sync_seconds": statistics.median(samples),
            "min_launch_and_sync_seconds": min(samples),
            "median_host_enqueue_seconds": statistics.median(enqueue)}
