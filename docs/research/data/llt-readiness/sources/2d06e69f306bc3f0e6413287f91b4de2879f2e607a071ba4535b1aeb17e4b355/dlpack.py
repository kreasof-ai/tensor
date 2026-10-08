"""Legacy and versioned DLPack GPU imports, with exactly-once ownership release.

Layouts and stream/capsule rules follow https://dmlc.github.io/dlpack/latest/.
This module imports no producer framework.
"""

from __future__ import annotations

import ctypes as c


class DLDevice(c.Structure):
    _fields_ = [("device_type", c.c_int32), ("device_id", c.c_int32)]


class DLDataType(c.Structure):
    _fields_ = [("code", c.c_uint8), ("bits", c.c_uint8), ("lanes", c.c_uint16)]


class DLTensor(c.Structure):
    _fields_ = [
        ("data", c.c_void_p),
        ("device", DLDevice),
        ("ndim", c.c_int32),
        ("dtype", DLDataType),
        ("shape", c.POINTER(c.c_int64)),
        ("strides", c.POINTER(c.c_int64)),
        ("byte_offset", c.c_uint64),
    ]


class DLManagedTensor(c.Structure):
    _fields_ = [
        ("dl_tensor", DLTensor),
        ("manager_ctx", c.c_void_p),
        ("deleter", c.c_void_p),
    ]


class DLPackVersion(c.Structure):
    _fields_ = [("major", c.c_uint32), ("minor", c.c_uint32)]


class DLManagedTensorVersioned(c.Structure):
    _fields_ = [
        ("version", DLPackVersion),
        ("manager_ctx", c.c_void_p),
        ("deleter", c.c_void_p),
        ("flags", c.c_uint64),
        ("dl_tensor", DLTensor),
    ]


_capsule_name = c.pythonapi.PyCapsule_GetName
_capsule_name.argtypes, _capsule_name.restype = [c.py_object], c.c_char_p
_capsule_pointer = c.pythonapi.PyCapsule_GetPointer
_capsule_pointer.argtypes, _capsule_pointer.restype = (
    [c.py_object, c.c_char_p],
    c.c_void_p,
)
_capsule_rename = c.pythonapi.PyCapsule_SetName
_capsule_rename.argtypes, _capsule_rename.restype = [c.py_object, c.c_char_p], c.c_int
# PyCapsule stores, rather than copies, these names; keep both strings alive.
_USED_NAMES = {
    b"dltensor": b"used_dltensor",
    b"dltensor_versioned": b"used_dltensor_versioned",
}


class ManagedTensor:
    def __init__(self, capsule, source):
        name = _capsule_name(capsule)
        if name not in _USED_NAMES:
            raise BufferError(
                "expected an unused dltensor or dltensor_versioned capsule"
            )
        self.address = _capsule_pointer(capsule, name)
        if not self.address:
            raise BufferError("DLPack capsule contains a null managed tensor")
        structure = (
            DLManagedTensorVersioned
            if name == b"dltensor_versioned"
            else DLManagedTensor
        )
        self.tensor = c.cast(self.address, c.POINTER(structure)).contents
        self.versioned = name == b"dltensor_versioned"
        self._capsule, self._source, self._released = capsule, source, False
        _capsule_rename(capsule, _USED_NAMES[name])

    def metadata(self, ordinal: int):
        import numpy as np

        if self.versioned:
            if self.tensor.version.major != 1:
                raise BufferError("unsupported DLPack major version")
            if self.tensor.flags & 1:
                raise BufferError(
                    "read-only GPU DLPack buffers cannot be passed to writable kernels"
                )
            if self.tensor.flags & ~3:
                raise BufferError("unsupported DLPack flags")
        tensor = self.tensor.dl_tensor
        if tensor.device.device_type != 2 or tensor.device.device_id != ordinal:
            raise BufferError("DLPack tensor must be on this CUDA device")
        if not 1 <= tensor.ndim <= 64 or not tensor.shape:
            raise BufferError("DLPack tensor needs 1 to 64 dimensions")
        shape = tuple(tensor.shape[index] for index in range(tensor.ndim))
        if any(size < 1 for size in shape):
            raise BufferError("DLPack tensor needs positive extents")
        dtype = tensor.dtype
        codes = {0: "int", 1: "uint", 2: "float", 4: "bfloat", 6: "bool"}
        if (
            dtype.lanes != 1
            or dtype.code not in codes
            or dtype.bits not in (8, 16, 32, 64)
            or (dtype.code == 2 and dtype.bits not in (16, 32, 64))
            or (dtype.code == 4 and dtype.bits != 16)
            or (dtype.code == 6 and dtype.bits != 8)
        ):
            raise BufferError("unsupported DLPack element dtype")
        from tensor.runtime.dtypes import BFLOAT16

        dtype = (
            BFLOAT16
            if dtype.code == 4
            else np.dtype(
                "bool" if dtype.code == 6 else f"{codes[dtype.code]}{dtype.bits}"
            )
        )
        if tensor.strides:
            expected = 1
            for index in reversed(range(tensor.ndim)):
                stride = tensor.strides[index]
                if stride < 0 or (shape[index] > 1 and stride != expected):
                    raise BufferError("GPU DLPack import requires contiguous storage")
                expected *= shape[index]
        pointer = (tensor.data or 0) + tensor.byte_offset
        if not tensor.data or pointer >= 1 << 64 or pointer % dtype.itemsize:
            raise BufferError("invalid or misaligned DLPack data pointer")
        return pointer, shape, dtype

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        deleter = self.tensor.deleter
        if deleter:
            # Keep the GIL for producers whose deleter references Python objects.
            c.PYFUNCTYPE(None, c.c_void_p)(deleter)(self.address)
        self._capsule, self._source = None, None


def borrow(source, *, stream: int, ordinal: int):
    try:
        capsule = source.__dlpack__(stream=stream, max_version=(1, 0))
    except TypeError as exc:
        if "max_version" not in str(exc) or "keyword" not in str(exc):
            raise
        capsule = source.__dlpack__(stream=stream)
    owner = ManagedTensor(capsule, source)
    try:
        pointer, shape, dtype = owner.metadata(ordinal)
        return owner, pointer, shape, dtype
    except BaseException:
        owner.release()
        raise


# Export descriptors retain the Buffer until the consumer invokes the deleter.
# Capsule destruction releases only unconsumed exports. A consumed capsule is
# renamed by its consumer and must not release ownership a second time.
_exports = {}
_raw_name = c.PYFUNCTYPE(c.c_char_p, c.c_void_p)(("PyCapsule_GetName", c.pythonapi))
_raw_pointer = c.PYFUNCTYPE(c.c_void_p, c.c_void_p, c.c_char_p)(
    ("PyCapsule_GetPointer", c.pythonapi)
)


@c.CFUNCTYPE(None, c.c_void_p)
def _delete_export(address):
    owner = _exports.pop(address, None)
    if owner is not None:
        owner[0]._dlpack_exports -= 1


@c.CFUNCTYPE(None, c.c_void_p)
def _delete_capsule(capsule):
    name = _raw_name(capsule)
    if name in _USED_NAMES:
        _delete_export(_raw_pointer(capsule, name))


def export(buffer, *, versioned):
    name = str(buffer.dtype)
    code = (
        4
        if name == "bfloat16"
        else 6
        if name == "bool"
        else 2
        if name.startswith("float")
        else 1
        if name.startswith("uint")
        else 0
    )
    shape = (c.c_int64 * len(buffer.shape))(*buffer.shape)
    tensor = DLTensor(
        buffer.pointer,
        DLDevice(2, buffer.device.ordinal),
        len(buffer.shape),
        DLDataType(code, buffer.dtype.itemsize * 8, 1),
        shape,
        None,
        0,
    )
    deleter = c.cast(_delete_export, c.c_void_p)
    managed = (
        DLManagedTensorVersioned(DLPackVersion(1, 0), None, deleter, 0, tensor)
        if versioned
        else DLManagedTensor(tensor, None, deleter)
    )
    address = c.addressof(managed)
    buffer._dlpack_exports = getattr(buffer, "_dlpack_exports", 0) + 1
    _exports[address] = (buffer, managed, shape)
    create = c.pythonapi.PyCapsule_New
    create.argtypes, create.restype = [c.c_void_p, c.c_char_p, c.c_void_p], c.py_object
    try:
        return create(
            address,
            b"dltensor_versioned" if versioned else b"dltensor",
            c.cast(_delete_capsule, c.c_void_p),
        )
    except BaseException:
        _delete_export(address)
        raise
