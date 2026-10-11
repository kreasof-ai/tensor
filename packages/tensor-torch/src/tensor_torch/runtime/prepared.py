"""Reusable calls with fixed storage or validated pointer rebinding."""

import ctypes as c
import threading
from contextlib import nullcontext
import torch
from tensor.runtime.signature import SCALAR_TYPES, buffer_argument
from .launch import _native_plan, _enqueue
from .streams import current_stream as _current_stream


class Prepared:
    def __init__(self, state, args, outputs):
        self.executable, call, self.launch, self.tensors, stream = state
        self._descriptor_storage = call.storage
        self.args, self.outputs = args, outputs
        self.device = self.executable.device
        self.arguments = type(call.arguments).from_buffer_copy(call.arguments)
        from tensor.runtime.abi import Argument, BufferDescriptor

        self.parameters = (c.c_void_p * len(self.arguments))(
            *(
                c.addressof(arg)
                + (
                    Argument.buffer.offset + BufferDescriptor.address.offset
                    if arg.kind == 1
                    else Argument.scalar.offset
                )
                for arg in self.arguments
            )
        )
        self.streams = {stream.cuda_stream: stream}
        self.lock = threading.RLock()
        self.metadata = [
            (tuple(t.shape), tuple(t.stride()), t.dtype, t.device, t.data_ptr())
            for t in self.tensors
        ]
        self._native = _native_plan(self, args, outputs, fixed=True)

    def __call__(self):
        if not self.device._open or self.executable._released:
            raise RuntimeError("prepared call belongs to a closed session")
        if self._native is not None:
            self._native()
            return
        for tensor, (shape, stride, dtype, device, pointer) in zip(
            self.tensors, self.metadata
        ):
            if (
                tensor.data_ptr() != pointer
                or tuple(tensor.shape) != shape
                or tuple(tensor.stride()) != stride
                or tensor.dtype != dtype
                or tensor.device != device
            ):
                raise ValueError("prepared tensor metadata or storage changed")
        with (
            self.lock,
            (
                torch.cuda.device(self.device.ordinal)
                if torch.cuda.current_device() != self.device.ordinal
                else nullcontext()
            ),
        ):
            stream = _current_stream(self.device.ordinal)
            if stream.cuda_stream not in self.streams:
                for tensor in self.tensors:
                    tensor.record_stream(stream)
                self.streams[stream.cuda_stream] = stream
            _enqueue(self.executable, self.launch, stream.cuda_stream, self.parameters)


class LaunchPlan:
    """Internal fast path, reached only after Region's concrete metadata guard."""

    def __init__(self, kernel, args, fallback):
        self.fallback = fallback
        self.prepared = kernel.prepare(*args)
        self.inputs = kernel.inputs
        self.outputs = kernel.outputs
        self.descriptors = kernel.manifest.get("abi", kernel.manifest["arguments"])
        alignments = {
            d["name"]: d.get("alignment", 256)
            for d in kernel.manifest["arguments"]
            if buffer_argument(d)
        }
        self.descriptors = [
            {**d, "alignment": alignments[d["name"]]} if buffer_argument(d) else d
            for d in self.descriptors
        ]
        self.output_specs = [
            (tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
            for t in self.prepared.outputs
        ]
        self.allocate = getattr(torch._C._dynamo.guards, "_empty_strided_cuda", None)
        self.positions = {d["name"]: i for i, d in enumerate(self.inputs)}
        self.output_positions = {d["name"]: i for i, d in enumerate(self.outputs)}
        self.kernel = kernel
        self._native = _native_plan(
            self.prepared, args, self.prepared.outputs, fixed=False
        )
        # Plans rebind all pointers before launch; retain descriptors, not the
        # first invocation's PyTorch input/output allocations.
        self.prepared.args = self.prepared.tensors = self.prepared.outputs = ()
        self.prepared.metadata = []
        # FX plans must not retain the initial tensors through the fixed executor.
        self.prepared._native = None

    def __call__(self, *args):
        prepared = self.prepared
        if not prepared.device._open or prepared.executable._released:
            raise RuntimeError("launch plan belongs to a closed session")
        if self._native is not None:
            output = self._native(*args)
            return self.fallback(*args) if output is None else output
        with (
            prepared.lock,
            (
                torch.cuda.device(prepared.device.ordinal)
                if torch.cuda.current_device() != prepared.device.ordinal
                else nullcontext()
            ),
        ):
            outputs = [
                (
                    self.allocate(shape, stride, dtype)
                    if self.allocate is not None
                    else torch.empty(shape, dtype=dtype, device=device)
                )
                for shape, stride, dtype, device in self.output_specs
            ]
            stream = _current_stream(prepared.device.ordinal)
            for argument, descriptor in zip(prepared.arguments, self.descriptors):
                name = descriptor["name"]
                value = (
                    outputs[self.output_positions[name]]
                    if name in self.output_positions
                    else args[self.positions[name]]
                )
                if buffer_argument(descriptor):
                    pointer = value.data_ptr()
                    if pointer % descriptor.get("alignment", 256):
                        return self.fallback(*args)
                    value.record_stream(stream)
                    argument.buffer.address = pointer
                else:
                    # FX emitters currently produce only buffer arguments.
                    raise ValueError("scalar launch plans are unsupported")
            _enqueue(
                prepared.executable,
                prepared.launch,
                stream.cuda_stream,
                prepared.parameters,
            )
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
