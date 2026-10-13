"""Owned and DLPack-borrowed buffers and their lifetimes."""

from __future__ import annotations

import ctypes as c

from tensor.runtime.abi import DTYPES, BufferDescriptor
from tensor.runtime.errors import TensorRuntimeError


class Buffer:
    """Contiguous owned or DLPack-borrowed allocation, valid within its session."""

    def __init__(
        self, device, pointer: int, shape: tuple[int, ...], dtype: object, *, owner=None
    ):
        self.device, self._pointer, self._shape, self._dtype = (
            device,
            pointer,
            shape,
            dtype,
        )
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
        self._descriptor = BufferDescriptor(
            pointer,
            self.nbytes,
            self._abi_shape,
            self._abi_strides,
            len(shape),
            DTYPES.get(str(dtype), 0),
            device.device_type,
            device.ordinal,
        )
        self._released = False
        self._dlpack_exports = 0

    pointer = property(lambda self: self._pointer)
    shape = property(lambda self: self._shape)
    dtype = property(lambda self: self._dtype)
    nbytes = property(lambda self: self._nbytes)
    strides = property(lambda self: self._strides)

    def view(self, shape):
        """Borrow this allocation with another contiguous shape; never copy it."""
        return BufferView(self, shape)

    def copy_from(self, source):
        """Copy equal-sized storage on the device's ordered submission stream."""
        self._check()
        source._check()
        if source.device is not self.device:
            raise ValueError("buffer copy requires the same device")
        if source.dtype != self.dtype or source.nbytes != self.nbytes:
            raise ValueError("buffer copy requires the same dtype and storage size")
        self.device._copy(self, source)
        return self

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
        if str(self.dtype) == "bfloat16":
            self._check()
            self.device.synchronize()
            return self.device._download_storage(self).tobytes()
        return self.to_numpy().tobytes()

    def release(self) -> None:
        if not self._released:
            self._check()
            self.device.synchronize()
            self._dispose()

    def _dispose(self) -> None:
        if self._released:
            return
        if getattr(self, "_dlpack_exports", 0):
            raise TensorRuntimeError(
                "release DLPack consumers before releasing this buffer"
            )
        self.device._dispose_buffer(self)
        self._released = True
        self.device._buffers.remove(self)

    def __dlpack_device__(self):
        self._check()
        return self.device.device_type, self.device.ordinal

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        self._check()
        if self.device.device_type != 2:
            raise BufferError("DLPack export currently supports CUDA buffers")
        if copy is True or dl_device not in (None, self.__dlpack_device__()):
            raise BufferError("DLPack export does not copy or transfer devices")
        if stream is None:
            self.device.synchronize()
        else:
            if type(stream) is not int or stream < 1:
                raise BufferError(
                    "CUDA DLPack export requires a positive consumer stream"
                )
            self.device.handoff(stream)
        from tensor.runtime.dlpack import export

        return export(self, versioned=max_version is not None and max_version >= (1, 0))


class BufferView(Buffer):
    """A reshape descriptor borrowing storage and lifetime from another buffer."""

    def __init__(self, base, shape):
        import math

        base._check()
        shape = (shape,) if type(shape) is int else tuple(shape)
        if not shape or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError("view shape must contain positive integers")
        if math.prod(shape) * base.dtype.itemsize != base.nbytes:
            raise ValueError("reshape view must preserve buffer storage size")
        super().__init__(base.device, base._pointer, shape, base.dtype, owner=base)
        self._base = base
        self.device._buffers.add(self)

    def _check(self):
        super()._check()
        self._base._check()

    def release(self):
        # Releasing a descriptor is safe even after its base was explicitly freed.
        if not self._released:
            self.device._check()
            self.device.synchronize()
            self._dispose()

    def _dispose(self):
        if self._released:
            return
        if self._dlpack_exports:
            raise TensorRuntimeError("release DLPack consumers before releasing this buffer")
        self._released = True
        self.device._buffers.discard(self)
