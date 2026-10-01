"""Native wgpu consumer. No TileLang, TVM, CUDA toolkit or PyTorch imports."""

from __future__ import annotations

import ctypes as c
from itertools import count
import math
import struct

from tensor.runtime.abi import CAPABILITIES, DTYPES, CallDescriptor, StreamDescriptor, check_requirement
from tensor.artifacts.format import ArtifactError, read_artifact
from tensor.runtime import Buffer, Event, Executable, Session, TensorRuntimeError
from tensor.runtime.signature import bind_shapes, resolve_launch, scalar_value
from tensor.providers.webgpu_contract import TARGET

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

    def __init__(self, ordinal=0, *, stream=None, max_buffer_size=None):
        if stream is not None:
            raise ValueError("WebGPU uses its owned queue; external streams are unsupported")
        if max_buffer_size is not None and (type(max_buffer_size) is not int or max_buffer_size < 4):
            raise ValueError("max_buffer_size must be an integer of at least four bytes")
        adapters = _adapters()
        if type(ordinal) is not int or not 0 <= ordinal < len(adapters):
            raise self.error(f"WebGPU adapter ordinal {ordinal} unavailable ({len(adapters)} adapters)")
        self.ordinal, self._adapter = ordinal, adapters[ordinal]
        self._max_buffer_size = max_buffer_size
        self.info = {"ordinal": ordinal, "provider": "webgpu", "arch": TARGET,
                     "name": str(self._adapter.info["device"]), "adapter": dict(self._adapter.info)}
        self._open = False
        self._generation = 0

    def __enter__(self):
        if self._open:
            raise self.error("device session is already open")
        limits = self._adapter.limits
        buffer_limit = min(limits["max-storage-buffer-binding-size"], limits["max-buffer-size"])
        if self._max_buffer_size is not None and self._max_buffer_size > buffer_limit:
            raise self.error(f"requested WebGPU buffer size {self._max_buffer_size} exceeds adapter limit {buffer_limit}")
        buffer_size = min(self._max_buffer_size or 134217728, buffer_limit)
        requested = {"max-compute-workgroup-storage-size": min(32768, limits["max-compute-workgroup-storage-size"]),
                     "max-storage-buffer-binding-size": buffer_size,
                     "max-buffer-size": buffer_size}
        features = ["shader-f16"] if "shader-f16" in self._adapter.features else []
        if "subgroup" in self._adapter.features:
            features.append("subgroup")
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
        self._prepared_plans = set()
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
        buffer._readback = None
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
        import wgpu
        if buffer._readback is None:
            buffer._readback=self._gpu.create_buffer(size=buffer._allocated,
                usage=wgpu.BufferUsage.COPY_DST|wgpu.BufferUsage.MAP_READ)
        staging=buffer._readback
        encoder=self._gpu.create_command_encoder()
        encoder.copy_buffer_to_buffer(buffer._storage,0,staging,0,buffer._allocated)
        self._gpu.queue.submit([encoder.finish()])
        staging.map_sync(wgpu.MapMode.READ)
        try:
            data=staging.read_mapped(copy=False)
            return np.frombuffer(data,dtype=buffer.dtype,count=math.prod(buffer.shape)).reshape(buffer.shape).copy()
        finally:staging.unmap()

    def _dispose_buffer(self, buffer):
        if buffer._readback is not None:
            buffer._readback.destroy();buffer._readback=None
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
        bindings, pod, grid = self._prepare_dispatch(executable, call)
        dimension = self._gpu.limits["max-compute-workgroups-per-dimension"]
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

    def _prepare_dispatch(self, executable, call):
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
        return bindings, pod, grid

    def prepare_plan(self, calls):
        """Validate and cache bindings; encode ordered dispatches once per replay."""
        return PreparedPlan(self, calls)

    def write(self, buffer, array):
        import numpy as np
        self._check();buffer._check()
        if buffer.device is not self:raise self.error("WebGPU write belongs to another device")
        array = np.ascontiguousarray(array, dtype=buffer.dtype)
        if array.shape != buffer.shape:raise ValueError("WebGPU write shape mismatch")
        self._gpu.queue.write_buffer(buffer._storage, 0, array.tobytes()+b"\0"*(buffer._allocated-buffer.nbytes))

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
            for plan in list(self._prepared_plans):plan.close()
            for executable in list(self._executables.values()):
                executable._dispose()
            for event in list(self._events):
                event.release()
            for buffer in list(self._buffers):
                buffer._dispose()
            self._gpu.destroy()
            self._open = False


class PreparedPlan:
    """Owned POD buffers and cached bindings for a fixed device-resident plan.

    Command buffers are freshly encoded per launch, as WebGPU consumes them on
    submission. This is batching, not CUDA-style captured graph replay.
    """
    def __init__(self, device, calls):
        import wgpu
        self.device=device;self.calls=tuple(calls);self.nodes=[];self.uniforms=[];self.closed=False
        self.resources=tuple(dict.fromkeys(resource for executable,call in self.calls for resource in (executable,*call.storage)))
        self.generation=device._generation
        # wgpu 0.29 native's ordinary set_bind_group allocates an empty C array
        # at every dispatch. These plans have no dynamic offsets and own groups.
        from wgpu.backends.wgpu_native._ffi import ffi,lib
        from wgpu.backends.wgpu_native._api import libf
        self._native_bind=libf.wgpuComputePassEncoderSetBindGroup
        self._null_offsets=ffi.NULL
        self._encode=None;self._records=b''
        try:
            from tensor.providers import _webgpu_native
        except ImportError:
            pass
        else:
            if wgpu.__version__=='0.29.0' and ffi.sizeof('void *')==8:
                self._encode=libf._make_proxy_func('tensorEncodePreparedPlan',_webgpu_native.encode)
                self._function_pointers=tuple(int(ffi.cast('uintptr_t',ffi.addressof(lib,name))) for name in (
                    'wgpuComputePassEncoderSetPipeline','wgpuComputePassEncoderSetBindGroup','wgpuComputePassEncoderDispatchWorkgroups'))
        device.info['prepared_encoding']='native' if self._encode else 'python'
        try:
            for executable,call in self.calls:
                if executable.device is not device:raise device.error("WebGPU plan belongs to another device")
                executable._check()
                bindings,pod,grid=device._prepare_dispatch(executable,call)
                uniform=device._gpu.create_buffer(size=(len(pod)+15)//16*16,
                    usage=wgpu.BufferUsage.UNIFORM|wgpu.BufferUsage.COPY_DST)
                self.uniforms.append(uniform)
                device._gpu.queue.write_buffer(uniform,0,pod)
                bindings.append({"binding":len(bindings),"resource":{"buffer":uniform}})
                group=device._gpu.create_bind_group(layout=executable.function.get_bind_group_layout(0),entries=bindings)
                self.nodes.append((executable.function,group,grid))
            if self._encode:
                dimension=device._gpu.limits['max-compute-workgroups-per-dimension']
                self._records=b''.join(struct.pack('<QQIIII',int(ffi.cast('uintptr_t',pipeline._internal)),
                    int(ffi.cast('uintptr_t',group._internal)),min(grid[0],dimension),grid[1],
                    (grid[0]+dimension-1)//dimension,0) for pipeline,group,grid in self.nodes)
            device._prepared_plans.add(self)
        except BaseException:
            self.close();raise

    def launch(self):
        device=self.device;device._check()
        if self.closed or self.generation!=device._generation:raise device.error("WebGPU plan is closed or belongs to a prior session")
        # Resource lifetime remains checked even though descriptors are bound once.
        for resource in self.resources:resource._check()
        dimension=device._gpu.limits['max-compute-workgroups-per-dimension']
        encoder=device._gpu.create_command_encoder()
        compute=encoder.begin_compute_pass()
        if self._encode:
            from wgpu.backends.wgpu_native._ffi import ffi
            self._encode(int(ffi.cast('uintptr_t',compute._internal)),self._records,*self._function_pointers)
        else:
            previous=None
            for pipeline,group,grid in self.nodes:
                if pipeline is not previous:compute.set_pipeline(pipeline)
                previous=pipeline
                self._native_bind(compute._internal,0,group._internal,0,self._null_offsets)
                width=min(grid[0],dimension)
                compute.dispatch_workgroups(width,grid[1],(grid[0]+width-1)//width)
        compute.end()
        device._gpu.queue.submit([encoder.finish()])

    def close(self):
        if self.closed:return
        for uniform in self.uniforms:uniform.destroy()
        self.uniforms.clear();self.nodes.clear();self.resources=();self._records=b'';self.closed=True
        self.device._prepared_plans.discard(self)
