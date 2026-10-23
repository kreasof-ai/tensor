"""CUDA driver bindings and errors."""

from __future__ import annotations

import ctypes as c
import os

from tensor.runtime.errors import TensorRuntimeError


class CudaError(TensorRuntimeError):
    pass


class CudaUnavailable(CudaError):
    pass


class Memcpy3D(c.Structure):
    """CUDA_MEMCPY3D_v2; depth one also supports arbitrary linear row pitches."""

    _fields_ = [
        ("srcXInBytes", c.c_size_t), ("srcY", c.c_size_t),
        ("srcZ", c.c_size_t), ("srcLOD", c.c_size_t),
        ("srcMemoryType", c.c_int), ("srcHost", c.c_void_p),
        ("srcDevice", c.c_uint64), ("srcArray", c.c_void_p),
        ("reserved0", c.c_void_p), ("srcPitch", c.c_size_t),
        ("srcHeight", c.c_size_t),
        ("dstXInBytes", c.c_size_t), ("dstY", c.c_size_t),
        ("dstZ", c.c_size_t), ("dstLOD", c.c_size_t),
        ("dstMemoryType", c.c_int), ("dstHost", c.c_void_p),
        ("dstDevice", c.c_uint64), ("dstArray", c.c_void_p),
        ("reserved1", c.c_void_p), ("dstPitch", c.c_size_t),
        ("dstHeight", c.c_size_t),
        ("WidthInBytes", c.c_size_t), ("Height", c.c_size_t),
        ("Depth", c.c_size_t),
    ]


class _Driver:
    def __init__(self) -> None:
        if c.sizeof(c.c_void_p) != 8:
            raise CudaUnavailable("CUDA requires 64-bit Python")
        try:
            self.lib = (
                c.WinDLL("nvcuda.dll") if os.name == "nt" else c.CDLL("libcuda.so.1")
            )
        except OSError as exc:
            raise CudaUnavailable("NVIDIA CUDA driver unavailable") from exc
        ptr, integer, uint, size, address = (
            c.c_void_p,
            c.c_int,
            c.c_uint,
            c.c_size_t,
            c.c_uint64,
        )
        signatures = {
            "cuInit": [uint],
            "cuDeviceGetCount": [c.POINTER(integer)],
            "cuDeviceGet": [c.POINTER(integer), integer],
            "cuDeviceGetAttribute": [c.POINTER(integer), integer, integer],
            "cuDeviceGetName": [ptr, integer, integer],
            "cuCtxGetCurrent": [c.POINTER(ptr)],
            "cuCtxSetCurrent": [ptr],
            "cuDevicePrimaryCtxRetain": [c.POINTER(ptr), integer],
            "cuDevicePrimaryCtxRelease_v2": [integer],
            "cuStreamCreate": [c.POINTER(ptr), uint],
            "cuStreamSynchronize": [ptr],
            "cuStreamDestroy_v2": [ptr],
            "cuStreamGetCtx": [ptr, c.POINTER(ptr)],
            "cuEventCreate": [c.POINTER(ptr), uint],
            "cuEventRecord": [ptr, ptr],
            "cuEventSynchronize": [ptr],
            "cuEventElapsedTime": [c.POINTER(c.c_float), ptr, ptr],
            "cuEventDestroy_v2": [ptr],
            "cuStreamWaitEvent": [ptr, ptr, uint],
            "cuPointerGetAttribute": [ptr, integer, address],
            "cuMemAlloc_v2": [c.POINTER(address), size],
            "cuMemFree_v2": [address],
            "cuMemcpyHtoD_v2": [address, ptr, size],
            "cuMemcpyDtoH_v2": [ptr, address, size],
            "cuMemcpyDtoD_v2": [address, address, size],
            "cuMemcpyDtoDAsync_v2": [address, address, size, ptr],
            "cuMemcpy3DAsync_v2": [c.POINTER(Memcpy3D), ptr],
            "cuMemsetD8Async": [address, c.c_ubyte, size, ptr],
            "cuModuleLoadData": [c.POINTER(ptr), ptr],
            "cuModuleUnload": [ptr],
            "cuModuleGetFunction": [c.POINTER(ptr), ptr, c.c_char_p],
            "cuFuncSetAttribute": [ptr, integer, integer],
            "cuLaunchKernel": [
                ptr,
                uint,
                uint,
                uint,
                uint,
                uint,
                uint,
                uint,
                ptr,
                c.POINTER(ptr),
                c.POINTER(ptr),
            ],
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
            error = CudaError(
                f"{name}: {(label.value or b'CUDA_ERROR').decode()} ({code}): "
                f"{(detail.value or b'').decode()}"
            )
            error.code = code
            raise error

    def device_info(self, ordinal: int) -> tuple[int, dict]:
        count, device = c.c_int(), c.c_int()
        self.call("cuDeviceGetCount", c.byref(count))
        if not 0 <= ordinal < count.value:
            raise CudaUnavailable(
                f"CUDA device {ordinal} unavailable ({count.value} detected)"
            )
        self.call("cuDeviceGet", c.byref(device), ordinal)
        major, minor = c.c_int(), c.c_int()
        self.call("cuDeviceGetAttribute", c.byref(major), 75, device)
        self.call("cuDeviceGetAttribute", c.byref(minor), 76, device)
        name = c.create_string_buffer(256)
        self.call("cuDeviceGetName", name, len(name), device)
        return device.value, {
            "ordinal": ordinal,
            "name": name.value.decode(errors="replace"),
            "arch": f"sm_{major.value}{minor.value}",
        }
