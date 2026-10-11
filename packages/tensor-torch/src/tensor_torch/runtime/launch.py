"""Bind the Tensor ABI, rebind pointers, and submit on PyTorch streams."""

import ctypes as c
import importlib
import os
import torch
from tensor.runtime.abi import BoundCall
from tensor.runtime.signature import SCALAR_TYPES, buffer_argument, scalar_value
from tensor_torch.runtime.session import session

try:
    from tensor_torch._launch import submit as _submit
except ImportError:
    _submit = None

_executor = None
if os.environ.get("TENSOR_TORCH_NATIVE", "1") != "0":
    # The C++ executor uses Torch's non-stable ABI: only load a module built for
    # this major/minor. Portable adapter wheels have no such module.
    version = "_".join(torch.__version__.split(".")[:2])
    try:
        _executor = importlib.import_module(
            "._executor_" + version, __package__.rsplit(".", 1)[0]
        )
    except ImportError:
        pass


def _native_plan(prepared, args, outputs, *, fixed):
    if _executor is None:
        return None
    manifest = prepared.executable.manifest
    names = manifest["outputs"]
    inputs = [d for d in manifest["arguments"] if d["name"] not in names]
    tensors = tuple(v for d, v in zip(inputs, args) if buffer_argument(d))
    positions = {
        d["name"]: i for i, d in enumerate(d for d in inputs if buffer_argument(d))
    }
    positions.update((name, len(tensors) + i) for i, name in enumerate(names))
    bindings = []
    alignments = {
        d["name"]: d.get("alignment", 256)
        for d in manifest["arguments"]
        if buffer_argument(d)
    }
    for argument, descriptor in zip(
        prepared.arguments, manifest.get("abi", manifest["arguments"])
    ):
        if buffer_argument(descriptor):
            bindings.append(
                (positions[descriptor["name"]], alignments[descriptor["name"]])
            )
        else:
            bindings.append(
                c.string_at(
                    c.addressof(argument) + type(argument).scalar.offset,
                    c.sizeof(SCALAR_TYPES[descriptor["dtype"]]),
                )
            )
    return _executor.Plan(
        prepared.executable.function.value,
        (
            *prepared.launch["grid"],
            *prepared.launch["block"],
            prepared.launch["shared_memory_bytes"],
        ),
        tensors,
        tuple(outputs),
        bindings,
        fixed,
    )


def _enqueue(executable, launch, stream, parameters):
    if _submit is None:
        executable.device.driver.call(
            "cuLaunchKernel",
            executable.function,
            *launch["grid"],
            *launch["block"],
            launch["shared_memory_bytes"],
            c.c_void_p(stream),
            parameters,
            None,
        )
    else:
        code = _submit(
            executable.function.value,
            *launch["grid"],
            *launch["block"],
            launch["shared_memory_bytes"],
            stream,
            c.addressof(parameters),
        )
        if code:
            from tensor.providers.cuda.driver import CudaError

            raise CudaError(f"cuLaunchKernel failed with CUDA error {code}")


def launch(
    kernel, args, outputs, *, contract=None, trusted_outputs=False, enqueue=True
):
    expected, placement = contract or kernel.metadata(args)
    if len(outputs) != len(expected):
        raise ValueError("wrong output count")
    tensors = [v for d, v in zip(kernel.inputs, args) if buffer_argument(d)]
    for output_index, (out, (shape, dtype)) in enumerate(zip(outputs, expected)):
        if (
            not isinstance(out, torch.Tensor)
            or tuple(out.shape) != shape
            or out.dtype != dtype
            or out.device != placement
            or not out.is_contiguous()
        ):
            raise ValueError("output shape, dtype, layout or device mismatch")
        if not trusted_outputs and any(
            out.untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
            for v in tensors + list(outputs[:output_index])
        ):
            raise ValueError("outputs must not alias inputs or each other")
    ordinal = placement.index
    with kernel._lock, torch.cuda.device(ordinal):
        device = session(ordinal)
        executable = kernel._executables.get(ordinal)
        if executable is None or not executable.device._open:
            executable = kernel._executables[ordinal] = device.load(kernel.path)
            kernel._prepared.clear()
        stream = torch.cuda.current_stream(ordinal)
        # record_stream protects allocator reuse after Python references disappear.
        # PyTorch owns producer ordering; current-stream DLPack import performs
        # the first binding handshake and validates pointers with the provider.
        all_values = dict(zip((d["name"] for d in kernel.inputs), args))
        all_values.update(zip((d["name"] for d in kernel.outputs), outputs))
        key = (
            ordinal,
            tuple(
                (
                    (d["name"], tuple(v.shape), v.dtype)
                    if buffer_argument(d)
                    else (
                        d["name"],
                        type(v),
                        v if d["name"] in kernel.manifest.get("symbols", {}) else None,
                    )
                )
                for d in kernel.manifest["arguments"]
                for v in [all_values[d["name"]]]
            ),
        )
        prepared = kernel._prepared.get(key)
        if prepared is None:
            borrowed = []
            try:
                bound = {}
                for d in kernel.manifest["arguments"]:
                    v = all_values[d["name"]]
                    if buffer_argument(d):
                        b = device.from_dlpack(v.detach())
                        borrowed.append(b)
                        bound[d["name"]] = b
                    else:
                        bound[d["name"]] = v
                bound, symbols, launch = executable._bind(
                    (), bound, include_outputs=True
                )
                call = BoundCall(
                    device, kernel.manifest, bound, symbols, launch, validated=True
                )
                parameters = call.cuda_parameters()
                # Keep descriptor storage, not DLPack owners, in the prepared cache.
                # Disposed Buffer objects retain ABI shape/stride backing only.
                prepared = kernel._prepared[key] = (call, parameters, launch)
                if len(kernel._prepared) > 64:
                    kernel._prepared.popitem(last=False)
                shared = launch["shared_memory_bytes"]
                if shared > executable._shared_limit:
                    device.driver.call(
                        "cuFuncSetAttribute", executable.function, 8, shared
                    )
                    executable._shared_limit = shared
            finally:
                # Imports queued producer->Tensor-stream waits. Hand back that
                # ordering before releasing their DLPack owners.
                device.handoff(stream.cuda_stream)
                for b in borrowed:
                    b._dispose()
        kernel._prepared.move_to_end(key)
        call, parameters, launch = prepared
        for argument, d in zip(
            call.arguments, kernel.manifest.get("abi", kernel.manifest["arguments"])
        ):
            if buffer_argument(d):
                value = all_values[d["name"]]
                pointer = value.data_ptr()
                alignment = next(
                    a.get("alignment", 256)
                    for a in kernel.manifest["arguments"]
                    if a["name"] == d["name"]
                )
                if pointer % alignment:
                    raise ValueError(f"{d['name']} has insufficient pointer alignment")
                value.record_stream(stream)
                argument.buffer.address = pointer
            elif d["name"] in all_values:
                scalar = scalar_value(all_values[d["name"]], d["dtype"], d["name"])
                scalar = SCALAR_TYPES[d["dtype"]](scalar)
                c.memmove(
                    c.addressof(argument) + type(argument).scalar.offset,
                    c.byref(scalar),
                    c.sizeof(scalar),
                )
        # The context is selected by torch.cuda.device. No default-stream launch
        # or stream synchronization occurs on the prepared path.
        if enqueue:
            device.driver.call(
                "cuLaunchKernel",
                executable.function,
                *launch["grid"],
                *launch["block"],
                launch["shared_memory_bytes"],
                c.c_void_p(stream.cuda_stream),
                parameters,
                None,
            )
        return executable, call, launch, tuple(tensors) + tuple(outputs), stream
