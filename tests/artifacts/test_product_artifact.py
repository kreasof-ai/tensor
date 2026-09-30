"""Executable bundles fail before CUDA sees corrupt or ambiguous payloads."""

import hashlib
import json
import zipfile

import pytest

from tensor.artifact import ArtifactError, read_artifact


def _bundle(path, *, corrupt=False, duplicate=False):
    files = {"kernel.cubin": b"\x7fELFtest", "kernel.tirx.json": b'{"nodes": []}'}
    manifest = {
        "format": "tensor.cuda", "format_version": 1, "kind": "cubin",
        "target": "sm_86", "entrypoint": "test_kernel",
        "launch": {"grid": [1, 1, 1], "block": [32, 1, 1], "shared_memory_bytes": 0},
        "arguments": [{"name": "out", "dtype": "float32", "shape": [32]}],
        "outputs": ["out"], "source_sha256": "0" * 64,
        "tilelang_version": "0.1.14", "tvm_ffi_version": "0.1.12", "op_set": [],
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    if corrupt:
        files["kernel.cubin"] = b"\x7fELFaltered"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, data in files.items():
            archive.writestr(name, data)
        if duplicate:
            archive.writestr("kernel.cubin", files["kernel.cubin"])


def test_product_artifact_reads_valid_payload(tmp_path):
    path = tmp_path / "valid.tbin"
    _bundle(path)
    manifest, files = read_artifact(path)
    assert manifest["target"] == "sm_86"
    assert files["kernel.cubin"].startswith(b"\x7fELF")


def test_product_artifact_rejects_corrupted_payload(tmp_path):
    path = tmp_path / "corrupt.tbin"
    _bundle(path, corrupt=True)
    with pytest.raises(ArtifactError, match="hash mismatch"):
        read_artifact(path)


def test_product_artifact_rejects_duplicate_members(tmp_path):
    path = tmp_path / "duplicate.tbin"
    with pytest.warns(UserWarning, match="Duplicate name"):
        _bundle(path, duplicate=True)
    with pytest.raises(ArtifactError, match="duplicate"):
        read_artifact(path)
