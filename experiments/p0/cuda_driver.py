"""Minimal CUDA Driver API binding for the opaque-artifact experiment.

Only ctypes and the installed NVIDIA driver are needed. This is deliberately
not a public provider API. Calls use the documented pointer/size/versioned ABI:
https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/cuda_8h_source.html
"""

from __future__ import annotations

import ctypes as C
import os


class CudaError(RuntimeError):
    pass


class CudaUnavailable(CudaError):
    pass


class Driver:
    def __init__(self):
        if C.sizeof(C.c_void_p) != 8:
            raise CudaUnavailable("the experiment requires 64-bit Python")
        try:
            self.lib = C.WinDLL("nvcuda.dll") if os.name == "nt" else C.CDLL("libcuda.so.1")
        except OSError as exc:
            raise CudaUnavailable("NVIDIA CUDA driver unavailable; a real NVIDIA host is required") from exc
        p, i, u, z, d = C.c_void_p, C.c_int, C.c_uint, C.c_size_t, C.c_uint64
        signatures = {
            "cuInit": [u], "cuDriverGetVersion": [C.POINTER(i)],
            "cuDeviceGetCount": [C.POINTER(i)], "cuDeviceGet": [C.POINTER(i), i],
            "cuDeviceGetName": [p, i, i], "cuDeviceGetAttribute": [C.POINTER(i), i, i],
            "cuCtxGetCurrent": [C.POINTER(p)], "cuCtxSetCurrent": [p],
            "cuCtxCreate_v2": [C.POINTER(p), u, i], "cuCtxDestroy_v2": [p],
            "cuModuleLoadData": [C.POINTER(p), p], "cuModuleUnload": [p],
            "cuModuleGetFunction": [C.POINTER(p), p, C.c_char_p],
            "cuMemAlloc_v2": [C.POINTER(d), z], "cuMemFree_v2": [d],
            "cuMemcpyHtoD_v2": [d, p, z], "cuMemcpyDtoH_v2": [p, d, z],
            "cuStreamCreate": [C.POINTER(p), u], "cuStreamSynchronize": [p],
            "cuStreamDestroy_v2": [p],
            "cuLaunchKernel": [p, u, u, u, u, u, u, u, p, C.POINTER(p), C.POINTER(p)],
            "cuGetErrorName": [i, C.POINTER(C.c_char_p)],
            "cuGetErrorString": [i, C.POINTER(C.c_char_p)],
        }
        for name, args in signatures.items():
            try:
                fn = getattr(self.lib, name)
            except AttributeError as exc:
                raise CudaUnavailable(f"CUDA driver is missing required API {name}") from exc
            fn.argtypes, fn.restype = args, i
        try:
            self.call("cuInit", 0)
        except CudaError as exc:
            # 100 = no CUDA device, 35 = insufficient driver.
            if getattr(exc, "code", None) in (100, 35):
                raise CudaUnavailable(str(exc)) from exc
            raise

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            label, detail = C.c_char_p(), C.c_char_p()
            self.lib.cuGetErrorName(code, C.byref(label))
            self.lib.cuGetErrorString(code, C.byref(detail))
            error = CudaError(f"{name}: {(label.value or b'CUDA_ERROR').decode()} ({code}): "
                              f"{(detail.value or b'').decode()}")
            error.code = code
            raise error

    def device_info(self, ordinal=0):
        count, dev = C.c_int(), C.c_int()
        self.call("cuDeviceGetCount", C.byref(count))
        if not 0 <= ordinal < count.value:
            raise CudaUnavailable(f"CUDA device {ordinal} unavailable ({count.value} devices)")
        self.call("cuDeviceGet", C.byref(dev), ordinal)
        major, minor, version = C.c_int(), C.c_int(), C.c_int()
        # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR = 75/76.
        self.call("cuDeviceGetAttribute", C.byref(major), 75, dev)
        self.call("cuDeviceGetAttribute", C.byref(minor), 76, dev)
        self.call("cuDriverGetVersion", C.byref(version))
        name = C.create_string_buffer(256)
        self.call("cuDeviceGetName", name, len(name), dev)
        return dev.value, {"name": name.value.decode(), "arch": f"sm_{major.value}{minor.value}",
                           "driver_version": version.value, "ordinal": ordinal}


class Session:
    """Owns one context and stream; frees all resources on success or failure."""

    def __init__(self, driver: Driver, ordinal=0):
        self.driver = driver
        self.device, self.info = driver.device_info(ordinal)
        self.context, self.previous = C.c_void_p(), C.c_void_p()
        self.stream, self.module = C.c_void_p(), C.c_void_p()
        self.allocations = []

    def __enter__(self):
        try:
            self.driver.call("cuCtxGetCurrent", C.byref(self.previous))
            self.driver.call("cuCtxCreate_v2", C.byref(self.context), 0, self.device)
            self.driver.call("cuStreamCreate", C.byref(self.stream), 1)  # NON_BLOCKING
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc, tb):
        # Keep cleanup attempts independent; preserve the original failure.
        errors = []
        calls = []
        if self.stream.value:
            calls.append(("cuStreamSynchronize", self.stream))
        calls += [("cuMemFree_v2", ptr) for ptr in reversed(self.allocations)]
        if self.module.value:
            calls.append(("cuModuleUnload", self.module))
        if self.stream.value:
            calls.append(("cuStreamDestroy_v2", self.stream))
        if self.context.value:
            calls.append(("cuCtxDestroy_v2", self.context))
            calls.append(("cuCtxSetCurrent", self.previous))
        for name, arg in calls:
            try:
                self.driver.call(name, arg)
            except CudaError as error:
                errors.append(error)
        if errors and exc_type is None:
            raise errors[0]

    def load(self, binary: bytes, entrypoint: str):
        self.image = C.create_string_buffer(binary)
        self.driver.call("cuModuleLoadData", C.byref(self.module), self.image)
        function = C.c_void_p()
        self.driver.call("cuModuleGetFunction", C.byref(function), self.module, entrypoint.encode())
        return function

    def allocate(self, nbytes):
        pointer = C.c_uint64()
        self.driver.call("cuMemAlloc_v2", C.byref(pointer), nbytes)
        self.allocations.append(pointer)
        return pointer

    def launch(self, function, pointers, launch):
        # kernelParams is an array of addresses OF argument values, not device
        # addresses themselves. Keep the uint64 holders alive through the call.
        params = (C.c_void_p * len(pointers))(*(C.addressof(pointer) for pointer in pointers))
        self.driver.call("cuLaunchKernel", function, *launch["grid"], *launch["block"],
                         launch["shared_memory_bytes"], self.stream, params, None)

    def synchronize(self):
        self.driver.call("cuStreamSynchronize", self.stream)
