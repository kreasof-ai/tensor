"""Shared safetensors storage contracts; model adapters supply shapes and dtypes."""

from dataclasses import dataclass
import ctypes as ct
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType

import numpy as np

STORAGE = {"F8_E4M3": ("uint8", 1), "BF16": ("uint16", 2), "F32": ("float32", 4)}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate checkpoint JSON key: {key}")
        result[key] = value
    return result


def json_document(value):
    return json.loads(value, object_pairs_hook=_object)


@dataclass(frozen=True)
class TensorInfo:
    shard: Path
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


def inspect_tensors(directory, index, shapes, dtypes):
    """Validate selected tensors and every extent before any device allocation."""
    directory = Path(directory).resolve()
    if shapes.keys() != dtypes.keys() or any(name not in index for name in shapes):
        raise ValueError("incomplete checkpoint tensor contract")
    tensors = {}
    for filename in sorted({index[name] for name in shapes}):
        shard = (directory / filename).resolve()
        if shard.parent != directory or not shard.is_file():
            raise ValueError(
                "checkpoint shard must be a local file within its directory"
            )
        size = shard.stat().st_size
        with shard.open("rb") as stream:
            encoded = stream.read(8)
            if len(encoded) != 8:
                raise ValueError("truncated safetensors header")
            length = struct.unpack("<Q", encoded)[0]
            if not 2 <= length <= min(size - 8, 100_000_000):
                raise ValueError("invalid safetensors header length")
            header = json_document(stream.read(length))
        if not isinstance(header, dict):
            raise ValueError("invalid safetensors header object")
        intervals = []
        for name, info in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(info, dict):
                raise ValueError(f"invalid safetensors tensor entry: {name}")
            shape, offsets = info.get("shape"), info.get("data_offsets")
            if (
                not isinstance(shape, list)
                or any(type(x) is not int or x < 0 for x in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(x) is not int for x in offsets)
                or not 0 <= offsets[0] <= offsets[1] <= size - length - 8
            ):
                raise ValueError(f"invalid safetensors extent: {name}")
            intervals.append(tuple(offsets))
            if name not in shapes:
                continue
            dtype = info.get("dtype")
            if (
                name in tensors
                or index[name] != filename
                or dtype not in STORAGE
                or tuple(shape) != tuple(shapes[name])
                or offsets[1] - offsets[0] != math.prod(shape) * STORAGE[dtype][1]
            ):
                raise ValueError(f"invalid checkpoint tensor layout: {name}")
            if dtype != dtypes[name]:
                raise ValueError(
                    f"unsupported checkpoint tensor dtype: {name}: {dtype}"
                )
            tensors[name] = TensorInfo(
                shard,
                dtype,
                tuple(shape),
                8 + length + offsets[0],
                offsets[1] - offsets[0],
            )
        ordered = sorted(intervals)
        if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
            raise ValueError("overlapping safetensors storage")
    if tensors.keys() != shapes.keys():
        raise ValueError("index references missing safetensors tensors")
    return MappingProxyType(tensors)


def read_tensor(info):
    """Read storage bits, without interpreting BF16/FP8 as host numeric types."""
    with info.shard.open("rb") as stream:
        stream.seek(info.offset)
        raw = stream.read(info.nbytes)
    if len(raw) != info.nbytes:
        raise ValueError("checkpoint changed after validation")
    return np.frombuffer(raw, dtype=STORAGE[info.dtype][0]).reshape(info.shape)


def upload_bits(device, raw, dtype):
    """Copy native storage bytes and release the allocation on transfer failure."""
    raw = np.ascontiguousarray(raw)
    expected = {
        "bfloat16": np.dtype("uint16"),
        "uint8": np.dtype("uint8"),
        "float32": np.dtype("float32"),
    }
    if dtype not in expected or raw.dtype != expected[dtype]:
        raise ValueError("raw checkpoint storage does not match device dtype")
    buffer = device.empty(raw.shape, dtype)
    try:
        device.driver.call(
            "cuMemcpyHtoD_v2", buffer.pointer, ct.c_void_p(raw.ctypes.data), raw.nbytes
        )
        device.driver.call("cuStreamSynchronize", None)
        return buffer
    except BaseException:
        buffer.release()
        raise
