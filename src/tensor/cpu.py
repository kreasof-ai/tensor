"""Synchronous CPU runtime provider consuming the same ABI 1 descriptors as CUDA."""

from __future__ import annotations

import ctypes as c
from itertools import count
from pathlib import Path
import platform
import tempfile

from tensor.abi import CAPABILITIES, ErrorDescriptor, StreamDescriptor, check_requirement
from tensor.artifact import ArtifactError, read_artifact
from tensor.runtime import Buffer, Executable, Session, TensorRuntimeError


class Device(Session):
    device_type = 1
    error = TensorRuntimeError
    capabilities = CAPABILITIES | {"events"}

    def __init__(self, ordinal=0, *, stream=None):
        if type(ordinal) is not int or ordinal != 0 or stream not in (None, 0):
            raise ValueError("CPU provider supports ordinal 0 and its synchronous stream")
        self.ordinal = ordinal
        self.info = {"ordinal": 0, "name": "CPU", "arch": "cpu-linux-x86_64", "provider": "cpu"}
        self.limits = {"grid": [(1<<31)-1]*3, "block": [1024]*3}
        self._buffers = set()
        self._modules = []
        self._events = set()
        self._open = False
        self._generation = 0

    def __enter__(self):
        if self._open:
            raise self.error("device session is already open")
        self._directory = tempfile.TemporaryDirectory(prefix="tensor-cpu-runtime-")
        self._module_ids = count()
        self._generation += 1
        self._start_session()
        self._open = True
        return self

    def _check(self):
        if not self._open:
            raise self.error("CPU device session is closed")

    def stream_descriptor(self):
        return StreamDescriptor(1, 0, 0)

    def synchronize(self):
        self._check()

    def empty(self, shape, dtype="float32"):
        import numpy as np
        self._check()
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if not shape or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("buffer shape must contain positive integers")
        dtype = np.dtype(dtype)
        if dtype.hasobject or dtype.itemsize < 1:
            raise ValueError("buffer dtype must have a fixed byte size")
        size = dtype.itemsize
        for extent in shape:
            size *= extent
        if size >= (1<<63)-4096:
            raise ValueError("buffer allocation size exceeds CPU address space")
        # Satisfy any alignment accepted by the artifact profile, not NumPy's incidental alignment.
        storage = np.empty(size + 4095, dtype="uint8")
        offset = (-storage.ctypes.data) % 4096
        array = storage[offset:offset + size].view(dtype).reshape(shape)
        buffer = Buffer(self, array.ctypes.data, shape, dtype)
        buffer._storage = array
        self._buffers.add(buffer)
        return buffer

    def from_numpy(self, array):
        import numpy as np
        array = np.ascontiguousarray(array)
        buffer = self.empty(array.shape, array.dtype)
        buffer._storage[...] = array
        return buffer

    def from_dlpack(self, source):
        import numpy as np
        if source.__dlpack_device__() != (1, 0):
            raise BufferError("CPU provider imports CPU DLPack tensors only")
        return self.from_numpy(np.from_dlpack(source))

    def _download(self, buffer):
        return buffer._storage.copy()

    def _dispose_buffer(self, buffer):
        buffer._storage = None

    def full(self, shape, fill_value, dtype="float32"):
        import numpy as np
        return self.from_numpy(np.full(shape, fill_value, dtype=dtype))

    def zeros(self, shape, dtype="float32"):
        return self.full(shape, 0, dtype)

    def ones(self, shape, dtype="float32"):
        return self.full(shape, 1, dtype)

    def randn(self, shape, dtype="float32", *, seed=None):
        import numpy as np
        return self.from_numpy(np.random.default_rng(seed).standard_normal(shape).astype(dtype))

    def arange(self, stop, dtype="float32"):
        import numpy as np
        return self.from_numpy(np.arange(stop, dtype=dtype))

    def load(self, artifact):
        self._check()
        manifest, files = read_artifact(artifact)
        if manifest.get("provider", "cuda") != "cpu":
            raise ArtifactError("artifact requires a different runtime provider")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise ArtifactError("CPU native artifacts require Linux x86-64")
        check_requirement(manifest["runtime_abi"], self.capabilities)
        path = Path(self._directory.name) / f"kernel-{next(self._module_ids)}.so"
        path.write_bytes(files["kernel.so"])
        try:
            library = c.CDLL(str(path))
        except OSError as exc:
            path.unlink()
            raise ArtifactError(f"CPU executable image unavailable: {exc}") from exc
        try:
            function = getattr(library, manifest["entrypoint"])
        except AttributeError as exc:
            from _ctypes import dlclose
            dlclose(library._handle)
            path.unlink()
            raise ArtifactError(f"CPU executable entrypoint unavailable: {manifest['entrypoint']}") from exc
        from tensor.abi import CallDescriptor
        function.argtypes = [c.POINTER(CallDescriptor), c.POINTER(ErrorDescriptor)]
        function.restype = c.c_int32
        self._modules.append(library)
        return Executable(self, manifest, library, function, None)

    def _unload_executable(self, executable):
        from _ctypes import dlclose
        dlclose(executable.module._handle)
        self._modules.remove(executable.module)

    def _launch(self, executable, call):
        error = ErrorDescriptor()
        status = executable.function(c.byref(call.descriptor), c.byref(error))
        if status:
            raise self.error(f"CPU kernel: {error.message.decode(errors='replace')} ({status})")

    def record_event(self):
        from tensor.runtime import Event
        self._check()
        event = Event(self, None)
        self._events.add(event)
        return event

    def wait(self, event):
        self._check()
        event._check(self)

    def _dispose_event(self, event):
        pass

    def __exit__(self, exc_type, exc, tb):
        for executable in list(self._executables.values()):
            executable._dispose()
        for event in list(self._events):
            event.release()
        for buffer in list(self._buffers):
            buffer._dispose()
        self._modules.clear()
        self._directory.cleanup()
        self._open = False
