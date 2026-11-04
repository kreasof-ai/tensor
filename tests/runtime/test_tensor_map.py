"""Static TMA launch metadata keeps the public buffer binding API."""

import ctypes as c
from types import SimpleNamespace
import pytest
from tensor.runtime.tensor_map import validate_descriptor


def spec():
    return dict(
        data_type=6,
        global_dims=[256, 128],
        global_strides=[512],
        box_dims=[64, 128],
        element_strides=[1, 1],
        interleave=0,
        swizzle=3,
        l2_promotion=2,
        oob_fill=0,
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(global_strides=[256]),
        dict(box_dims=[64]),
        dict(global_dims=[0, 128]),
        dict(data_type=99),
        dict(element_strides=[0, 1]),
        dict(swizzle=7),
    ],
)
def test_malformed_tensor_maps_fail_before_driver_encoding(change):
    with pytest.raises(ValueError):
        validate_descriptor({**spec(), **change}, 65536)


def test_tensor_map_rejects_wrong_bound_buffer_size():
    with pytest.raises(ValueError, match="bound buffer"):
        validate_descriptor(spec(), 32768)


def test_encoding_uses_aligned_owned_storage_and_full_pointer_width():
    from tensor.providers.cuda.tensor_map import encode

    calls = []
    function = lambda *args: None

    def call(name, *args):
        calls.append((name, args))
        assert args[0] % 64 == 0 and args[3] == 2**40
        assert list(args[4]) == [256, 128]
        c.memmove(args[0], bytes(range(128)), 128)

    driver = SimpleNamespace(
        lib=SimpleNamespace(cuTensorMapEncodeTiled=function), call=call
    )
    backing, address = encode(
        driver, SimpleNamespace(pointer=2**40, nbytes=65536), spec()
    )
    assert c.string_at(address, 128) == bytes(range(128))
    assert c.addressof(backing) <= address < c.addressof(backing) + 64
    assert len(calls) == 1
