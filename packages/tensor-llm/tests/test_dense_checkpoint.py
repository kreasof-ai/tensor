"""The native loader must copy BF16 storage bits without numeric conversion."""

import ctypes as ct
from types import SimpleNamespace

import numpy as np

from tensor_llm.qwen35.dense.checkpoint import DenseCheckpoint


def test_upload_preserves_bfloat16_bits_and_official_float32_values():
    arrays = {
        "model.language_model.embed_tokens.weight": np.array(
            [[0x3F80, 0x8000, 0x7FC0]], dtype="uint16"
        ),
        "model.language_model.norm.weight": np.array([1.25, -0.125], dtype="float32"),
    }
    checkpoint = object.__new__(DenseCheckpoint)
    checkpoint.tensors = {
        name: SimpleNamespace(dtype="BF16" if raw.dtype == np.uint16 else "F32")
        for name, raw in arrays.items()
    }
    checkpoint.config = SimpleNamespace(layers=())
    checkpoint.read = arrays.__getitem__
    allocations = []

    class Driver:
        def call(self, name, *args):
            if name == "cuMemcpyHtoD_v2":
                ct.memmove(*args)
            else:
                assert name == "cuStreamSynchronize"

    class Device:
        driver = Driver()

        def empty(self, shape, dtype):
            raw = ct.create_string_buffer(
                int(np.prod(shape)) * (2 if dtype == "bfloat16" else 4)
            )
            allocations.append(raw)
            return SimpleNamespace(pointer=ct.addressof(raw), shape=shape, dtype=dtype)

        def from_numpy(self, *args, **kwargs):
            raise AssertionError("BF16 bit patterns must not be numerically converted")

    weights = checkpoint.upload(Device())
    for name, raw in arrays.items():
        assert ct.string_at(weights[name].pointer, raw.nbytes) == raw.tobytes()
