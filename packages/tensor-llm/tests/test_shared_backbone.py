"""Shared loaders reject bad artifacts and release partially built resources."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from tensor_llm.common.artifacts import artifact_path, load_kernels
from tensor_llm.common.state import allocate_hybrid_states
from tensor_llm.qwen35.dense.checkpoint import DenseCheckpoint


class Resource:
    def __init__(self):
        self.released = False

    def release(self):
        assert not self.released
        self.released = True


def test_bundle_formats_share_integrity_and_partial_load_cleanup(tmp_path):
    image = tmp_path / "kernel.tbin"
    image.write_bytes(b"compiled kernel")
    descriptor = dict(
        path=image.name, sha256=hashlib.sha256(image.read_bytes()).hexdigest()
    )
    assert artifact_path(tmp_path, dict(logical=descriptor)) == image
    assert (
        artifact_path(tmp_path, dict(artifact=image.name, sha256=descriptor["sha256"]))
        == image
    )
    loaded = Resource()

    class Device:
        def load(self, path):
            return loaded

    with pytest.raises(ValueError, match="checksum"):
        load_kernels(
            Device(),
            tmp_path,
            {"first": descriptor, "second": dict(descriptor, sha256="bad")},
        )
    assert loaded.released


def test_bundle_symlink_cannot_escape(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside.tbin")
    outside.write_bytes(b"outside")
    try:
        (tmp_path / "escape.tbin").symlink_to(outside)
        with pytest.raises(ValueError, match="escapes"):
            artifact_path(
                tmp_path,
                dict(path="escape.tbin", sha256=hashlib.sha256(b"outside").hexdigest()),
            )
    finally:
        outside.unlink()


def test_dense_upload_releases_previous_weight_on_later_failure():
    checkpoint = object.__new__(DenseCheckpoint)
    checkpoint.config = SimpleNamespace(layers=())
    checkpoint.tensors = {
        name: SimpleNamespace(dtype="BF16") for name in ("first", "second")
    }
    checkpoint.read = lambda name: np.zeros(2, np.uint16)
    allocations = []

    class Driver:
        def call(self, name, *args):
            if name == "cuMemcpyHtoD_v2" and len(allocations) == 2:
                raise RuntimeError("transfer failed")

    class Device:
        driver = Driver()

        def empty(self, shape, dtype):
            resource = Resource()
            resource.pointer = 1
            allocations.append(resource)
            return resource

    with pytest.raises(RuntimeError, match="transfer failed"):
        checkpoint.upload(Device())
    assert len(allocations) == 2 and all(resource.released for resource in allocations)


@pytest.mark.parametrize("kv_dtype", ["bfloat16", "fp8"])
def test_state_failure_releases_recurrent_and_cache_allocations(kv_dtype):
    config = SimpleNamespace(
        layers=("linear_attention", "full_attention"),
        key_heads=1,
        value_heads=2,
        key_dim=128,
        value_dim=128,
        kv_heads=1,
        head_dim=256,
    )
    allocations = []

    class Device:
        def empty(self, shape, dtype):
            if len(allocations) == 3:
                raise MemoryError("capacity exhausted")
            resource = Resource()
            allocations.append(resource)
            return resource

    with pytest.raises(MemoryError, match="capacity exhausted"):
        allocate_hybrid_states(Device(), config, 2, 128, kv_dtype=kv_dtype)
    assert len(allocations) == 3 and all(resource.released for resource in allocations)
