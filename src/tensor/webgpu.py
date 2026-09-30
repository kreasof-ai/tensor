"""Native wgpu consumer. No TileLang, TVM, CUDA toolkit or PyTorch imports."""

from __future__ import annotations

import ctypes as c
from itertools import count
import math
import struct

from tensor.abi import CAPABILITIES, DTYPES, CallDescriptor, StreamDescriptor, check_requirement
from tensor.artifact import ArtifactError, read_artifact
from tensor.runtime import Buffer, Event, Executable, Session, TensorRuntimeError
from tensor.signature import bind_shapes, resolve_launch, scalar_value
from tensor.webgpu_contract import TARGET

_handles = count(1)


def _adapters():
    try:
        import wgpu
    except ImportError as exc:
        raise TensorRuntimeError("WebGPU needs the optional tensor-workspace[webgpu] extra") from exc
    try:
        adapters = wgpu.gpu.enumerate_adapters_sync()
    except Exception as exc:
        raise TensorRuntimeError(f"WebGPU adapter enumeration failed: {exc}") from exc
    # Software adapters are explicit in info; they are useful for contract CI,
    # but never count as a second physical GPU for Phase 5 acceptance.
    return sorted(adapters, key=lambda a: a.info["adapter_type"] == "CPU")


def probe(ordinal=0):
    adapters = _adapters()
    if type(ordinal) is not int or not 0 <= ordinal < len(adapters):
        raise TensorRuntimeError(f"WebGPU adapter ordinal {ordinal} unavailable ({len(adapters)} adapters)")
    adapter = adapters[ordinal]
    return {"status": "ok", "ordinal": ordinal, "provider": "webgpu", "arch": TARGET,
            "adapter": dict(adapter.info), "features": sorted(adapter.features), "limits": dict(adapter.limits)}


class OpaqueBuffer(Buffer):
    @property
    def pointer(self):
        raise BufferError("WebGPU buffers have opaque handles, not device pointers")


class Device(Session):
    device_type = 256
    error = TensorRuntimeError
    capabilities = CAPABILITIES | {"opaque_buffer_handles", "async_launch", "events"}

    def __init__(self, ordinal=0, *, stream=None):
        if stream is not None:
            raise ValueError("WebGPU uses its owned queue; external streams are unsupported")
        adapters = _adapters()
        if type(ordinal) is not int or not 0 <= ordinal < len(adapters):
            raise self.error(f"WebGPU adapter ordinal {ordinal} unavailable ({len(adapters)} adapters)")
        self.ordinal, self._adapter = ordinal, adapters[ordinal]
        self.info = {"ordinal": ordinal, "provider": "webgpu", "arch": TARGET,
                     "name": str(self._adapter.info["device"]), "adapter": dict(self._adapter.info)}
        self._open = False
        self._generation = 0

    def __enter__(self):
        if self._open:
            raise self.error("device session is already open")
        limits = self._adapter.limits
        requested = {"max-compute-workgroup-storage-size": min(32768, limits["max-compute-workgroup-storage-size"]),
                     "max-storage-buffer-binding-size": min(134217728, limits["max-storage-buffer-binding-size"]),
                     "max-buffer-size": min(134217728, limits["max-buffer-size"])}
        features = ["shader-f16"] if "shader-f16" in self._adapter.features else []
        try:
            self._gpu = self._adapter.request_device_sync(required_features=features, required_limits=requested)
        except Exception as exc:
            raise self.error(f"WebGPU device creation failed: {exc}") from exc
        self.info.update(features=sorted(self._gpu.features), limits=dict(self._gpu.limits))
        dimension = self._gpu.limits["max-compute-workgroups-per-dimension"]
        self.limits = {"grid": [min((1<<31)-1, dimension*dimension), dimension, 1],
                       "block": [self._gpu.limits[f"max-compute-workgroup-size-{axis}"] for axis in "xyz"]}
        self._generation += 1
        self._start_session()
        self._buffers, self._buffer_handles, self._events = set(), {}, set()
        self._open = True
        return self

    def _check(self):
        if not self._open:
            raise self.error("WebGPU device session is closed")

    def stream_descriptor(self):
        return StreamDescriptor(self.device_type, self.ordinal, self._session_id)

    def synchronize(self):
        self._check()
        try:
            self._gpu.queue.on_submitted_work_done_sync()
        except Exception as exc:
            raise self.error(f"WebGPU queue completion failed: {exc}") from exc

    def empty(self, shape, dtype="float32"):
        import numpy as np
        import wgpu
        self._check()
        shape = (shape,) if type(shape) is int else tuple(shape)
        if not shape or any(type(size) is not int or size < 1 for size in shape):
            raise ValueError("buffer shape must contain positive integers")
        dtype = np.dtype(dtype)
        if str(dtype) not in ("float16", "float32", "int32", "uint32"):
            raise ValueError("WebGPU buffers support float16, float32, int32, uint32")
        if dtype == np.dtype("float16") and "shader-f16" not in self._gpu.features:
            raise self.error("WebGPU adapter lacks required shader-f16 feature")
        size = math.prod(shape)*dtype.itemsize
        allocated = (size+3)//4*4
        limit = min(self._gpu.limits["max-buffer-size"], self._gpu.limits["max-storage-buffer-binding-size"])
        if allocated > limit:
            raise ValueError(f"WebGPU buffer exceeds adapter binding/allocation limit {limit}")
        buffer = OpaqueBuffer(self, 0, shape, dtype)
        buffer._handle = next(_handles)
        buffer._allocated = allocated
        buffer._storage = self._gpu.create_buffer(size=allocated,
            usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
        self._buffers.add(buffer)
        self._buffer_handles[buffer._handle] = buffer
        return buffer

    def from_numpy(self, array):
        import numpy as np
        array = np.ascontiguousarray(array)
        buffer = self.empty(array.shape, array.dtype)
        try:
            self._gpu.queue.write_buffer(buffer._storage, 0, array.tobytes()+b"\0"*(buffer._allocated-buffer.nbytes))
        except BaseException:
            buffer.release()
            raise
        return buffer

    def from_dlpack(self, source):
        raise BufferError("WebGPU has no raw-address DLPack import; upload a NumPy array")

    def _download(self, buffer):
        import numpy as np
        data = self._gpu.queue.read_buffer(buffer._storage, 0, buffer._allocated)
        return np.frombuffer(data, dtype=buffer.dtype, count=math.prod(buffer.shape)).reshape(buffer.shape).copy()

    def _dispose_buffer(self, buffer):
        buffer._storage.destroy()
        buffer._storage = None
        self._buffer_handles.pop(buffer._handle, None)

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
        import wgpu
        self._check()
        manifest, files = read_artifact(artifact)
        if manifest.get("provider") != "webgpu":
            raise ArtifactError("artifact requires a different runtime provider")
        check_requirement(manifest["runtime_abi"], self.capabilities)
        requirements = manifest["webgpu"]
        if missing := set(requirements["required_features"]) - self._gpu.features:
            raise ArtifactError(f"WebGPU adapter lacks required features: {sorted(missing)}")
        block = manifest["launch"]["block"]
        if (requirements["workgroup_storage_bytes"] > self._gpu.limits["max-compute-workgroup-storage-size"]
                or math.prod(block) > self._gpu.limits["max-compute-invocations-per-workgroup"]
                or any(size > limit for size, limit in zip(block, self.limits["block"]))):
            raise ArtifactError("WebGPU shader exceeds adapter workgroup limits")
        for arg in manifest["arguments"]:
            if arg["kind"] == "buffer" and all(type(x) is int for x in arg["shape"]):
                import numpy as np
                if math.prod(arg["shape"])*np.dtype(arg["dtype"]).itemsize > self._gpu.limits["max-storage-buffer-binding-size"]:
                    raise ArtifactError("WebGPU artifact buffer exceeds adapter binding limit")
        try:
            shader = self._gpu.create_shader_module(code=files["kernel.wgsl"].decode())
            pipeline = self._gpu.create_compute_pipeline(layout="auto",
                compute={"module": shader, "entry_point": manifest["entrypoint"]})
            pod_size = (sum(item["kind"] == "scalar" for item in manifest["abi"])+1)*4
            uniform = self._gpu.create_buffer(size=(pod_size+15)//16*16,
                usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
        except Exception as exc:
            raise ArtifactError(f"WebGPU shader/pipeline creation failed: {exc}") from exc
        return Executable(self, manifest, uniform, pipeline, shader)

    def _unload_executable(self, executable):
        executable.module.destroy()

    def _launch(self, executable, call):
        self._check()
        descriptor = call.descriptor
        if descriptor.abi_version != 1 or descriptor.struct_size < c.sizeof(CallDescriptor):
            raise self.error("unsupported WebGPU call ABI")
        if bytes(descriptor.stream) != bytes(self.stream_descriptor()) or descriptor.flags or descriptor.shared_memory_bytes:
            raise self.error("WebGPU call belongs to another session or has invalid flags/workspace")
        abi = executable.manifest["abi"]
        if descriptor.argument_count != len(abi) or not descriptor.arguments:
            raise self.error("WebGPU call argument count mismatch")
        bindings, pod, scalars, values = [], bytearray(), {}, {}
        for index, expected in enumerate(abi):
            argument = descriptor.arguments[index]
            if argument.dtype != DTYPES[expected["dtype"]]:
                raise self.error("WebGPU call dtype mismatch")
            if expected["kind"] == "buffer":
                buffer = self._buffer_handles.get(argument.scalar)
                if argument.kind != 3 or buffer is None:
                    raise self.error("WebGPU opaque buffer handle is released or unknown")
                buffer._check()
                metadata = argument.buffer
                if (metadata.address or metadata.byte_size != buffer.nbytes or metadata.dtype != argument.dtype
                        or metadata.device_type != self.device_type or metadata.device_ordinal != self.ordinal
                        or metadata.rank != len(buffer.shape) or not metadata.shape or not metadata.strides
                        or tuple(metadata.shape[:metadata.rank]) != buffer.shape
                        or tuple(metadata.strides[:metadata.rank]) != buffer.strides):
                    raise self.error("WebGPU opaque buffer metadata mismatch")
                values[expected["name"]] = buffer
                bindings.append({"binding": len(bindings), "resource": {"buffer": buffer._storage,
                                                                        "offset": 0, "size": buffer._allocated}})
            else:
                if argument.kind != 2 or argument.scalar >> 32 or any(bytes(argument.buffer)):
                    raise self.error("WebGPU scalar payload is noncanonical")
                payload = struct.pack("<I", argument.scalar)
                value = struct.unpack({"float32": "<f", "int32": "<i", "uint32": "<I"}[expected["dtype"]], payload)[0]
                scalars[expected["name"]] = scalar_value(value, expected["dtype"], expected["name"])
                values[expected["name"]] = scalars[expected["name"]]
                pod.extend(payload)
        dimensions = {name: scalars[name] for name in executable.manifest["symbols"]}
        symbols, _ = bind_shapes(executable.manifest, values, dimensions)
        launch = resolve_launch(executable.manifest["launch"], symbols)
        if list(descriptor.grid) != launch["grid"] or list(descriptor.block) != launch["block"]:
            raise self.error("WebGPU call launch metadata mismatch")
        dimension = self._gpu.limits["max-compute-workgroups-per-dimension"]
        grid = launch["grid"]
        if any(size > limit for size, limit in zip(grid, self.limits["grid"])):
            raise self.error("WebGPU launch exceeds adapter dispatch limits")
        pod.extend(struct.pack("<I", grid[0]-1))
        self._gpu.queue.write_buffer(executable.module, 0, pod)
        bindings.append({"binding": len(bindings), "resource": {"buffer": executable.module}})
        try:
            group = self._gpu.create_bind_group(layout=executable.function.get_bind_group_layout(0), entries=bindings)
            encoder = self._gpu.create_command_encoder()
            compute = encoder.begin_compute_pass()
            compute.set_pipeline(executable.function)
            compute.set_bind_group(0, group)
            width = min(grid[0], dimension)
            compute.dispatch_workgroups(width, grid[1], (grid[0]+width-1)//width)
            compute.end()
            self._gpu.queue.submit([encoder.finish()])
        except Exception as exc:
            raise self.error(f"WebGPU dispatch failed: {exc}") from exc

    def record_event(self):
        self._check()
        event = Event(self, None)
        self._events.add(event)
        return event

    def wait(self, event):
        self._check()
        event._check(self)
        event.device.synchronize()

    def _dispose_event(self, event):
        pass

    def __exit__(self, exc_type, exc, tb):
        try:
            self.synchronize()
        finally:
            for executable in list(self._executables.values()):
                executable._dispose()
            for event in list(self._events):
                event.release()
            for buffer in list(self._buffers):
                buffer._dispose()
            self._gpu.destroy()
            self._open = False
