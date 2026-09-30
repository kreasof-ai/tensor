"""Independently versioned, provider-neutral runtime descriptors (C ABI 1.1)."""

from __future__ import annotations

import ctypes as c
import sys

from tensor.signature import SCALAR_TYPES, buffer_argument, evaluate, scalar_value

ABI_MAJOR = 1
ABI_MINOR = 2
DTYPES = {name: index for index, name in enumerate(("bool", "int8", "uint8", "int16", "uint16",
          "int32", "uint32", "int64", "uint64", "float16", "float32", "float64"), 1)}
CAPABILITIES = frozenset({"contiguous", "scalars", "symbolic_shapes", "executable_descriptors", "no_external_workspace"})
KNOWN_CAPABILITIES = CAPABILITIES | {"opaque_buffer_handles"}


class BufferDescriptor(c.Structure):
    _fields_ = [("address", c.c_uint64), ("byte_size", c.c_uint64),
                ("shape", c.POINTER(c.c_int64)), ("strides", c.POINTER(c.c_int64)),
                ("rank", c.c_uint32), ("dtype", c.c_uint32),
                ("device_type", c.c_uint32), ("device_ordinal", c.c_int32)]


class Argument(c.Structure):
    _fields_ = [("kind", c.c_uint32), ("dtype", c.c_uint32),
                ("buffer", BufferDescriptor), ("scalar", c.c_uint64)]


class StreamDescriptor(c.Structure):
    _fields_ = [("device_type", c.c_uint32), ("device_ordinal", c.c_int32), ("handle", c.c_uint64)]


class CallDescriptor(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32),
                ("arguments", c.POINTER(Argument)), ("argument_count", c.c_uint32), ("flags", c.c_uint32),
                ("grid", c.c_uint32 * 3), ("block", c.c_uint32 * 3),
                ("shared_memory_bytes", c.c_uint64), ("stream", StreamDescriptor)]


class ErrorDescriptor(c.Structure):
    _fields_ = [("code", c.c_int32), ("message", c.c_char * 508)]


class WorkspaceRequirements(c.Structure):
    _fields_ = [("byte_size", c.c_uint64), ("alignment", c.c_uint32),
                ("device_type", c.c_uint32), ("flags", c.c_uint32), ("reserved", c.c_uint32)]


class ExecutableDescriptor(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32),
                ("device_type", c.c_uint32), ("device_ordinal", c.c_int32),
                ("session", c.c_uint64), ("handle", c.c_uint64),
                ("argument_count", c.c_uint32), ("flags", c.c_uint32),
                ("workspace", WorkspaceRequirements)]


class EventDescriptor(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32),
                ("device_type", c.c_uint32), ("device_ordinal", c.c_int32),
                ("session", c.c_uint64), ("handle", c.c_uint64),
                ("flags", c.c_uint32), ("reserved", c.c_uint32)]


def workspace_requirement() -> dict:
    # This single-kernel profile has no caller-supplied global scratch buffer.
    # CUDA shared memory remains a per-launch resource, not external workspace.
    return {"bytes": 0, "alignment": 1}


def check_workspace(requirement) -> None:
    if (not isinstance(requirement, dict) or set(requirement) != {"bytes", "alignment"}
            or type(requirement["bytes"]) is not int or requirement["bytes"] != 0
            or type(requirement["alignment"]) is not int or requirement["alignment"] != 1):
        raise ValueError("this runtime profile requires zero external workspace with alignment 1")


def runtime_requirement(arguments: list[dict], symbols: dict) -> dict:
    required = ["contiguous", "executable_descriptors", "no_external_workspace"]
    if any(not buffer_argument(item) for item in arguments):
        required.append("scalars")
    if symbols:
        required.append("symbolic_shapes")
    return {"major": ABI_MAJOR, "minor": 1, "required_capabilities": required}


def check_requirement(requirement: dict, capabilities=KNOWN_CAPABILITIES) -> None:
    if (not isinstance(requirement, dict) or set(requirement) != {"major", "minor", "required_capabilities"}
            or type(requirement["major"]) is not int or requirement["major"] != ABI_MAJOR
            or type(requirement["minor"]) is not int or not 0 <= requirement["minor"] <= ABI_MINOR):
        raise ValueError("unsupported Tensor runtime ABI version")
    required = requirement["required_capabilities"]
    if (not isinstance(required, list) or any(not isinstance(name, str) for name in required)
            or len(set(required)) != len(required)):
        raise ValueError("invalid runtime capabilities")
    if missing := set(required) - capabilities:
        raise ValueError(f"missing runtime capabilities: {sorted(missing)}")
    if "opaque_buffer_handles" in required and requirement["minor"] < 2:
        raise ValueError("opaque buffer handles require runtime ABI 1.2")


class BoundCall:
    """Own descriptor backing storage while providers consume a borrowed call."""

    def __init__(self, device, manifest, values, symbols, launch, *, validated=False):
        if c.sizeof(c.c_void_p) != 8 or sys.byteorder != "little":
            raise ValueError("Tensor ABI 1 requires a 64-bit little-endian host")
        abi = manifest.get("abi", [dict(item, kind="buffer") for item in manifest["arguments"]])
        self.storage = []
        self.arguments = (Argument * len(abi))()
        for argument, descriptor in zip(self.arguments, abi):
            name, dtype = descriptor["name"], descriptor["dtype"]
            argument.dtype = DTYPES[dtype]
            if buffer_argument(descriptor):
                value = values[name]
                self.storage.append(value)
                argument.kind = 1
                argument.buffer = value._descriptor
                if "opaque_buffer_handles" in device.capabilities:
                    argument.kind = 3
                    argument.scalar = value._handle
            else:
                value = values[name] if name in values else evaluate({"var": name}, symbols)
                scalar = SCALAR_TYPES[dtype](value if validated else scalar_value(value, dtype, name))
                argument.kind = 2
                c.memmove(c.addressof(argument) + Argument.scalar.offset, c.byref(scalar), c.sizeof(scalar))
        self.descriptor = CallDescriptor(ABI_MAJOR, c.sizeof(CallDescriptor), self.arguments, len(abi), 0,
            (c.c_uint32 * 3)(*launch["grid"]), (c.c_uint32 * 3)(*launch["block"]),
            launch["shared_memory_bytes"], device.stream_descriptor())

    def cuda_parameters(self):
        return (c.c_void_p * len(self.arguments))(*(
            c.addressof(arg) + (Argument.buffer.offset + BufferDescriptor.address.offset if arg.kind == 1
                                else Argument.scalar.offset) for arg in self.arguments))
