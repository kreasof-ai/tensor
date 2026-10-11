"""Prepared WebGPU compute encoding and submission."""

from __future__ import annotations

import struct


class PreparedPlan:
    """Owned POD buffers and cached bindings for a fixed device-resident plan.

    Command buffers are freshly encoded per launch, as WebGPU consumes them on
    submission. This is batching, not CUDA-style captured graph replay.
    """

    def __init__(self, device, calls):
        import wgpu

        self.device = device
        self.calls = tuple(calls)
        self.nodes = []
        self.uniforms = []
        self.closed = False
        self.resources = tuple(
            dict.fromkeys(
                resource
                for executable, call in self.calls
                for resource in (executable, *call.storage)
            )
        )
        self.generation = device._generation
        # wgpu 0.29 native's ordinary set_bind_group allocates an empty C array
        # at every dispatch. These plans have no dynamic offsets and own groups.
        from wgpu.backends.wgpu_native._ffi import ffi, lib
        from wgpu.backends.wgpu_native._api import libf

        self._native_bind = libf.wgpuComputePassEncoderSetBindGroup
        self._null_offsets = ffi.NULL
        self._encode = None
        self._records = b""
        self._submit_native = None
        try:
            from tensor.providers import _webgpu_native
        except ImportError:
            pass
        else:
            if wgpu.__version__ == "0.29.0" and ffi.sizeof("void *") == 8:
                self._encode = libf._make_proxy_func(
                    "tensorEncodePreparedPlan", _webgpu_native.encode
                )
                self._function_pointers = tuple(
                    int(ffi.cast("uintptr_t", ffi.addressof(lib, name)))
                    for name in (
                        "wgpuComputePassEncoderSetPipeline",
                        "wgpuComputePassEncoderSetBindGroup",
                        "wgpuComputePassEncoderDispatchWorkgroups",
                    )
                )
                if hasattr(_webgpu_native, "submit"):
                    self._submit_native = libf._make_proxy_func(
                        "tensorSubmitPreparedPlan", _webgpu_native.submit
                    )
                    self._submission_descriptors = tuple(
                        ffi.new(name + " *")
                        for name in (
                            "WGPUCommandEncoderDescriptor",
                            "WGPUComputePassDescriptor",
                            "WGPUCommandBufferDescriptor",
                        )
                    )
                    self._descriptor_records = struct.pack(
                        "<QQQ",
                        *(
                            int(ffi.cast("uintptr_t", v))
                            for v in self._submission_descriptors
                        ),
                    )
                    self._submission_functions = struct.pack(
                        "<" + "Q" * 12,
                        *(
                            int(ffi.cast("uintptr_t", ffi.addressof(lib, name)))
                            for name in (
                                "wgpuDeviceCreateCommandEncoder",
                                "wgpuCommandEncoderBeginComputePass",
                                "wgpuComputePassEncoderSetPipeline",
                                "wgpuComputePassEncoderSetBindGroup",
                                "wgpuComputePassEncoderDispatchWorkgroups",
                                "wgpuComputePassEncoderEnd",
                                "wgpuCommandEncoderCopyBufferToBuffer",
                                "wgpuCommandEncoderFinish",
                                "wgpuQueueSubmit",
                                "wgpuCommandBufferRelease",
                                "wgpuComputePassEncoderRelease",
                                "wgpuCommandEncoderRelease",
                            )
                        ),
                    )

                    def captured_error():
                        stack = libf._error_handler._get_proxy_stack()
                        return bool(stack and stack[-1].message is not None)

                    self._submission_error_probe = captured_error
        device.info["prepared_encoding"] = "native" if self._encode else "python"
        device.info["prepared_submission"] = (
            "native" if self._submit_native else "python"
        )
        try:
            for executable, call in self.calls:
                if executable.device is not device:
                    raise device.error("WebGPU plan belongs to another device")
                executable._check()
                bindings, pod, grid = device._prepare_dispatch(executable, call)
                uniform = device._gpu.create_buffer(
                    size=(len(pod) + 15) // 16 * 16,
                    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                )
                self.uniforms.append(uniform)
                device._gpu.queue.write_buffer(uniform, 0, pod)
                bindings.append(
                    {"binding": len(bindings), "resource": {"buffer": uniform}}
                )
                group = device._gpu.create_bind_group(
                    layout=executable.function.get_bind_group_layout(0),
                    entries=bindings,
                )
                self.nodes.append((executable.function, group, grid))
            if self._encode:
                dimension = device._gpu.limits["max-compute-workgroups-per-dimension"]
                self._records = b"".join(
                    struct.pack(
                        "<QQIIII",
                        int(ffi.cast("uintptr_t", pipeline._internal)),
                        int(ffi.cast("uintptr_t", group._internal)),
                        min(grid[0], dimension),
                        grid[1],
                        (grid[0] + dimension - 1) // dimension,
                        0,
                    )
                    for pipeline, group, grid in self.nodes
                )
            device._prepared_plans.add(self)
        except BaseException:
            self.close()
            raise

    def launch(self, *, readback=None):
        """Submit dispatches; optionally copy and return an owned host snapshot."""
        device = self.device
        device._check()
        if self.closed or self.generation != device._generation:
            raise device.error("WebGPU plan is closed or belongs to a prior session")
        # Resource lifetime remains checked even though descriptors are bound once.
        for resource in self.resources:
            resource._check()
        staging = device._readback_staging(readback) if readback is not None else None
        if self._submit_native and self._encode:
            from wgpu.backends.wgpu_native._ffi import ffi

            handle = lambda value: int(ffi.cast("uintptr_t", value))
            self._submit_native(
                handle(device._gpu._internal),
                handle(device._gpu.queue._internal),
                self._records,
                self._descriptor_records,
                self._submission_functions,
                handle(readback._storage._internal) if staging is not None else 0,
                handle(staging._internal) if staging is not None else 0,
                readback._allocated if staging is not None else 0,
                self._submission_error_probe,
            )
            return (
                device._map_readback(readback, staging) if staging is not None else None
            )
        dimension = device._gpu.limits["max-compute-workgroups-per-dimension"]
        encoder = device._gpu.create_command_encoder()
        compute = encoder.begin_compute_pass()
        if self._encode:
            from wgpu.backends.wgpu_native._ffi import ffi

            self._encode(
                int(ffi.cast("uintptr_t", compute._internal)),
                self._records,
                *self._function_pointers,
            )
        else:
            previous = None
            for pipeline, group, grid in self.nodes:
                if pipeline is not previous:
                    compute.set_pipeline(pipeline)
                previous = pipeline
                self._native_bind(
                    compute._internal, 0, group._internal, 0, self._null_offsets
                )
                width = min(grid[0], dimension)
                compute.dispatch_workgroups(
                    width, grid[1], (grid[0] + width - 1) // width
                )
        compute.end()
        if staging is not None:
            encoder.copy_buffer_to_buffer(
                readback._storage, 0, staging, 0, readback._allocated
            )
        device._gpu.queue.submit([encoder.finish()])
        if staging is not None:
            return device._map_readback(readback, staging)

    def close(self):
        if self.closed:
            return
        for uniform in self.uniforms:
            uniform.destroy()
        self.uniforms.clear()
        self.nodes.clear()
        self.resources = ()
        self._records = b""
        self.closed = True
        self.device._prepared_plans.discard(self)
