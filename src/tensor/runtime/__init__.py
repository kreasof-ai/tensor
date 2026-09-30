"""Shared buffers, signature binding, output allocation and launch calls.

Runtime providers own allocation, image loading, stream ordering and execution.
This layer imports no compiler or vendor library.
"""
from __future__ import annotations

import statistics
import time
import ctypes as c
from itertools import count

from tensor.runtime.abi import (ABI_MAJOR, BoundCall, BufferDescriptor, DTYPES,
                        EventDescriptor, ExecutableDescriptor, WorkspaceRequirements)
from tensor.runtime.signature import (bind_shapes, buffer_argument, resolve_launch, resolve_shape)


class TensorRuntimeError(RuntimeError):
    pass


_identities = count(1)


class Session:
    """Common executable ownership; providers still own vendor resources."""

    def _start_session(self):
        self._session_id = next(_identities)
        self._executables = {}

    def get_executable(self, descriptor):
        if (not isinstance(descriptor, ExecutableDescriptor)
                or descriptor.abi_version != ABI_MAJOR
                or descriptor.struct_size < c.sizeof(ExecutableDescriptor)):
            raise self.error("unsupported executable descriptor ABI")
        self._check()
        if (descriptor.device_type != self.device_type or descriptor.device_ordinal != self.ordinal
                or descriptor.session != self._session_id):
            raise self.error("executable descriptor belongs to a different device session")
        executable = self._executables.get(descriptor.handle)
        if executable is None:
            raise self.error("executable handle is released or unknown")
        expected = executable._descriptor
        if (descriptor.argument_count != expected.argument_count or descriptor.flags != expected.flags
                or bytes(descriptor.workspace) != bytes(expected.workspace)):
            raise self.error("executable descriptor metadata mismatch")
        return executable

    def get_event(self, descriptor):
        if (not isinstance(descriptor, EventDescriptor) or descriptor.abi_version != ABI_MAJOR
                or descriptor.struct_size < c.sizeof(EventDescriptor)):
            raise self.error("unsupported event descriptor ABI")
        self._check()
        if (descriptor.device_type != self.device_type or descriptor.device_ordinal != self.ordinal
                or descriptor.session != self._session_id):
            raise self.error("event descriptor belongs to a different device session")
        event = next((item for item in self._events if item._descriptor.handle == descriptor.handle), None)
        if event is None:
            raise self.error("event handle is released or unknown")
        if descriptor.flags != event._descriptor.flags or descriptor.reserved:
            raise self.error("event descriptor metadata mismatch")
        return event


class Event:
    """Provider-owned completion point. Buffers and external streams stay borrowed."""

    def __init__(self, device, handle):
        self.device, self.handle = device, handle
        self._generation = device._generation
        self._released = False
        self._descriptor = EventDescriptor(ABI_MAJOR, c.sizeof(EventDescriptor), device.device_type,
            device.ordinal, device._session_id, next(_identities), int("async_launch" not in device.capabilities), 0)

    @property
    def descriptor(self):
        self._check(self.device)
        return EventDescriptor.from_buffer_copy(self._descriptor)

    def _check(self, device):
        if (self._released or self._generation != self.device._generation
                or device.device_type != self.device.device_type or device.ordinal != self.device.ordinal):
            raise TensorRuntimeError("event is released or belongs to a different provider/device")
        self.device._check()

    def release(self):
        if not self._released:
            self._check(self.device)
            self.device._dispose_event(self)
            self.device._events.remove(self)
            self._released = True


class Buffer:
    """Contiguous owned or DLPack-borrowed allocation, valid within its session."""

    def __init__(self, device: Device, pointer: int, shape: tuple[int, ...], dtype: object, *, owner=None):
        self.device, self._pointer, self._shape, self._dtype = device, pointer, shape, dtype
        self._owner = owner
        self._generation = device._generation
        self.owned = owner is None
        self._nbytes = self.dtype.itemsize
        for extent in shape:
            self._nbytes *= extent
        strides = []
        stride = self.dtype.itemsize
        for extent in reversed(shape):
            strides.insert(0, stride)
            stride *= extent
        self._strides = tuple(strides)
        self._abi_shape = (c.c_int64 * len(shape))(*shape)
        self._abi_strides = (c.c_int64 * len(strides))(*strides)
        self._descriptor = BufferDescriptor(pointer, self.nbytes, self._abi_shape, self._abi_strides,
                                            len(shape), DTYPES.get(str(dtype), 0), device.device_type, device.ordinal)
        self._released = False

    pointer = property(lambda self: self._pointer)
    shape = property(lambda self: self._shape)
    dtype = property(lambda self: self._dtype)
    nbytes = property(lambda self: self._nbytes)
    strides = property(lambda self: self._strides)

    def _check(self) -> None:
        self.device._check()
        if self._generation != self.device._generation:
            raise self.device.error("buffer belongs to a previous device session")
        if self._released:
            raise self.device.error("buffer has been released")

    def to_numpy(self):
        self._check()
        self.device.synchronize()
        return self.device._download(self)

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
        self.device._dispose_buffer(self)
        self._released = True
        self.device._buffers.remove(self)


class Executable:
    def __init__(self, device: Device, manifest: dict, module, function, image):
        self.device, self.manifest, self.module, self.function, self._image = (
            device, manifest, module, function, image)
        self._shared_limit = 0
        self._generation = device._generation
        self._released = False
        self._handle = next(_identities)
        workspace = WorkspaceRequirements(0, 1, device.device_type, 0, 0)
        self._descriptor = ExecutableDescriptor(ABI_MAJOR, c.sizeof(ExecutableDescriptor), device.device_type,
            device.ordinal, device._session_id, self._handle,
            len(manifest.get("abi", manifest["arguments"])), int("async_launch" in device.capabilities), workspace)
        device._executables[self._handle] = self

    def _check(self):
        if not self.device._open:
            raise self.device.error("device session is closed")
        if self._generation != self.device._generation:
            raise self.device.error("executable belongs to a previous device session")
        if self._released:
            raise self.device.error("executable has been released")

    @property
    def descriptor(self):
        self._check()
        return ExecutableDescriptor.from_buffer_copy(self._descriptor)

    def workspace_requirements(self):
        self._check()
        return WorkspaceRequirements.from_buffer_copy(self._descriptor.workspace)

    def release(self):
        if not self._released:
            self._check()
            self.device.synchronize()
            self._dispose()

    def _dispose(self):
        if not self._released:
            self.device._unload_executable(self)
            self._released = True
            self.device._executables.pop(self._handle, None)
            self.module = self.function = self._image = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

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
                if value.device is not self.device:
                    raise ValueError(f"{descriptor['name']} belongs to a different device session")
                if value._released or value._generation != self._generation:
                    raise self.device.error("buffer has been released")
                # V1 omitted alignment metadata; retain CUDA allocation alignment there.
                alignment = descriptor.get("alignment", 256)
                if "opaque_buffer_handles" not in self.device.capabilities and value.pointer % alignment:
                    raise ValueError(f"{descriptor['name']} needs {alignment}-byte pointer alignment")
        symbols, values = bind_shapes(self.manifest, values, dimensions)
        launch = resolve_launch(self.manifest["launch"], symbols)
        for field in ("grid", "block"):
            if any(value > limit for value, limit in zip(launch[field], self.device.limits[field])):
                raise ValueError(f"launch {field} exceeds this device's limits")
        return values, symbols, launch

    def launch(self, *arguments, **bindings) -> None:
        """Launch with all frontend arguments; symbolic dimensions are inferred."""
        self._check()
        values, symbols, launch = self._bind(arguments, bindings, include_outputs=True)
        call = BoundCall(self.device, self.manifest, values, symbols, launch, validated=True)
        self.device._launch(self, call)

    def prepare(self, *inputs, **bindings):
        """Allocate declared outputs without launching, for repeated benchmarks."""
        self._check()
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
