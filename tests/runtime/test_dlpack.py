"""DLPack layout and managed-tensor lifetime checks without framework imports."""

import ctypes as c

import pytest

from tensor.dlpack import (DLDataType, DLDevice, DLManagedTensor,
                           DLManagedTensorVersioned, DLPackVersion, DLTensor, borrow)


class Producer:
    def __init__(self, *, versioned=True, strides=(1,), flags=0, major=1):
        self.calls, self.deletes = [], 0
        self.shape = (c.c_int64*1)(8)
        self.strides = (c.c_int64*1)(*strides) if strides is not None else None
        self.deleter = c.CFUNCTYPE(None, c.c_void_p)(self._delete)
        tensor = DLTensor(c.c_void_p(4096), DLDevice(2, 0), 1, DLDataType(2, 32, 1),
                          self.shape, self.strides, 64)
        pointer = c.cast(self.deleter, c.c_void_p)
        self.versioned = versioned
        if versioned:
            self.tensor = DLManagedTensorVersioned(DLPackVersion(major, 0), None, pointer, flags, tensor)
        else:
            self.tensor = DLManagedTensor(tensor, None, pointer)

    def _delete(self, pointer):
        assert pointer == c.addressof(self.tensor)
        self.deletes += 1

    def __dlpack__(self, *, stream, max_version=None):
        self.calls.append((stream, max_version))
        create = c.pythonapi.PyCapsule_New
        create.argtypes, create.restype = [c.c_void_p, c.c_char_p, c.c_void_p], c.py_object
        return create(c.addressof(self.tensor), b"dltensor_versioned" if self.versioned else b"dltensor", None)


@pytest.mark.parametrize("versioned", [False, True])
def test_dlpack_offset_stream_and_exactly_once_deleter(versioned):
    source = Producer(versioned=versioned)
    owner, pointer, shape, dtype = borrow(source, stream=123, ordinal=0)
    assert pointer == 4160 and shape == (8,) and str(dtype) == "float32"
    assert source.calls == [(123, (1, 0))]
    assert source.deletes == 0
    owner.release()
    owner.release()
    assert source.deletes == 1


@pytest.mark.parametrize("options,match", [({"strides": (2,)}, "contiguous"),
    ({"flags": 1}, "read-only"), ({"major": 2}, "major version")])
def test_rejected_dlpack_metadata_also_releases_owner(options, match):
    source = Producer(**options)
    with pytest.raises(BufferError, match=match):
        borrow(source, stream=1, ordinal=0)
    assert source.deletes == 1


def test_legacy_python_protocol_negotiation():
    class LegacyProducer(Producer):
        def __dlpack__(self, *, stream):
            return super().__dlpack__(stream=stream)

    source = LegacyProducer(versioned=False, strides=None)
    owner, _, _, _ = borrow(source, stream=1, ordinal=0)
    assert source.calls == [(1, None)]
    owner.release()
    assert source.deletes == 1
