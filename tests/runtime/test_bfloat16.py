"""BF16 ABI compatibility, rounding, and explicit capability negotiation."""

import numpy as np
import pytest
from tensor.runtime.abi import (
    DTYPES,
    CAPABILITIES,
    check_requirement,
    runtime_requirement,
)
from tensor.runtime.dtypes import encode_bfloat16, decode_bfloat16
from tests.runtime.test_dlpack import Producer
from tensor.runtime.dlpack import borrow, DLDataType


def test_bfloat16_ids_and_capability_negotiation():
    assert (
        DTYPES["float16"],
        DTYPES["float32"],
        DTYPES["float64"],
        DTYPES["bfloat16"],
    ) == (10, 11, 12, 13)
    requirement = runtime_requirement([{"dtype": "bfloat16", "kind": "buffer"}], {})
    assert requirement["minor"] == 3
    check_requirement(requirement)
    with pytest.raises(ValueError, match="capabilities"):
        check_requirement(requirement, capabilities=CAPABILITIES)
    with pytest.raises(ValueError, match="1.3"):
        check_requirement({**requirement, "minor": 2})
    assert (
        runtime_requirement([{"dtype": "float16", "kind": "buffer"}], {})["minor"] == 1
    )


def test_bfloat16_rounding_ties_extremes_and_nan():
    values = np.array(
        [1.0, 1.00390625, 1.01171875, -1.00390625, 0.0, -0.0, np.inf, -np.inf, np.nan],
        dtype="float32",
    )
    result = encode_bfloat16(values)
    np.testing.assert_array_equal(
        result[:8],
        np.array(
            [0x3F80, 0x3F80, 0x3F82, 0xBF80, 0, 0x8000, 0x7F80, 0xFF80], dtype="uint16"
        ),
    )
    assert np.isnan(decode_bfloat16(result)[-1])
    rng = np.random.default_rng(9)
    bits = rng.integers(0, 65536, 10000, dtype="uint16")
    finite = (bits & 0x7F80) != 0x7F80
    np.testing.assert_array_equal(
        encode_bfloat16(decode_bfloat16(bits[finite])), bits[finite]
    )


@pytest.mark.parametrize("versioned", [False, True])
def test_bfloat16_dlpack_metadata(versioned):
    source = Producer(versioned=versioned)
    source.tensor.dl_tensor.dtype = DLDataType(4, 16, 1)
    owner, pointer, shape, dtype = borrow(source, stream=1, ordinal=0)
    assert (
        str(dtype) == "bfloat16"
        and dtype.itemsize == 2
        and shape == (8,)
        and pointer == 4160
    )
    owner.release()
    assert source.deletes == 1


@pytest.mark.parametrize("bits,lanes", [(32, 1), (16, 2)])
def test_rejected_bfloat16_dlpack_releases_owner(bits, lanes):
    source = Producer()
    source.tensor.dl_tensor.dtype = DLDataType(4, bits, lanes)
    with pytest.raises(BufferError, match="dtype"):
        borrow(source, stream=1, ordinal=0)
    assert source.deletes == 1
