"""Framework-independent storage types and BF16 host conversion.

NumPy has no builtin BF16. BF16 buffers download as FP32 values; raw storage is
available through Buffer.to_bytes(). No optional dtype package is required.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class BFloat16:
    itemsize: int = 2
    hasobject: bool = False

    def __str__(self):
        return "bfloat16"


BFLOAT16 = BFloat16()


def dtype(value):
    return BFLOAT16 if str(value) == "bfloat16" else np.dtype(value)


def storage_dtype(value):
    return np.dtype("uint16") if str(value) == "bfloat16" else np.dtype(value)


def encode_bfloat16(values):
    """Round FP32 to nearest even, retaining infinities and quieting NaNs."""
    values = np.ascontiguousarray(values, dtype="float32")
    bits = values.view("uint32")
    nan = ((bits & 0x7F800000) == 0x7F800000) & ((bits & 0x007FFFFF) != 0)
    rounded = ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype("uint16")
    return np.where(nan, ((bits >> 16) | 0x40).astype("uint16"), rounded)


def decode_bfloat16(storage):
    return (np.asarray(storage, dtype="uint16").astype("uint32") << 16).view("float32")
