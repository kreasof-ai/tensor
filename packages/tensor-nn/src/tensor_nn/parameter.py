"""Explicit training storage: FP32 master weight, FP16 compute copy and state."""

from dataclasses import dataclass
import numpy as np


@dataclass
class Parameter:
    weight: object
    half: object
    grad: object
    moment: object
    variance: object
    shape: tuple

    @classmethod
    def from_numpy(cls, device, values):
        values = np.ascontiguousarray(values, dtype=np.float32)
        flat = values.reshape(-1)
        return cls(
            device.from_numpy(flat),
            device.from_numpy(flat.astype(np.float16)),
            device.empty(flat.shape),
            device.zeros(flat.shape),
            device.zeros(flat.shape),
            values.shape,
        )
