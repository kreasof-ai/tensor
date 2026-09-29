"""Read and validate Tensor's executable CUDA bundle without compiler imports."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

FORMAT = "tensor.cuda"
FORMAT_VERSION = 1
MAX_UNCOMPRESSED = 128 * 1024 * 1024
MAX_MANIFEST = 2 * 1024 * 1024
TARGET = re.compile(r"sm_[0-9]{2,3}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z_]\w*\Z")


class ArtifactError(ValueError):
    pass


def _dimensions(value: object, *, label: str) -> list[int]:
    if (not isinstance(value, list) or len(value) != 3
            or any(type(item) is not int or item < 1 for item in value)):
        raise ArtifactError(f"{label} must be three positive integers")
    return value


def validate_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest must be an object")
    if manifest.get("format") != FORMAT or manifest.get("format_version") != FORMAT_VERSION:
        raise ArtifactError("unsupported Tensor artifact format/version")
    if manifest.get("kind") != "cubin":
        raise ArtifactError("the runtime requires a cubin executable")
    target = manifest.get("target")
    if not isinstance(target, str) or not TARGET.fullmatch(target):
        raise ArtifactError("invalid CUDA target")
    entrypoint = manifest.get("entrypoint")
    if not isinstance(entrypoint, str) or not NAME.fullmatch(entrypoint):
        raise ArtifactError("invalid CUDA entrypoint")
    launch = manifest.get("launch")
    if not isinstance(launch, dict) or set(launch) != {"grid", "block", "shared_memory_bytes"}:
        raise ArtifactError("invalid launch description")
    _dimensions(launch["grid"], label="grid")
    block = _dimensions(launch["block"], label="block")
    if block[0] * block[1] * block[2] > 1024:
        raise ArtifactError("launch block exceeds 1024 threads")
    if type(launch["shared_memory_bytes"]) is not int or launch["shared_memory_bytes"] < 0:
        raise ArtifactError("invalid shared memory size")
    arguments = manifest.get("arguments")
    if not isinstance(arguments, list) or not arguments:
        raise ArtifactError("artifact has no buffer arguments")
    names = []
    for argument in arguments:
        if not isinstance(argument, dict) or set(argument) != {"name", "dtype", "shape"}:
            raise ArtifactError("invalid buffer argument")
        name, dtype, shape = argument["name"], argument["dtype"], argument["shape"]
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ArtifactError("invalid buffer name")
        if not isinstance(dtype, str) or not dtype:
            raise ArtifactError("invalid buffer dtype")
        if not isinstance(shape, list) or not shape or any(type(x) is not int or x < 1 for x in shape):
            raise ArtifactError("invalid buffer shape")
        names.append(name)
    if len(names) != len(set(names)):
        raise ArtifactError("duplicate buffer argument")
    outputs = manifest.get("outputs", [])
    if not isinstance(outputs, list) or any(name not in names for name in outputs) or len(outputs) != len(set(outputs)):
        raise ArtifactError("invalid output names")
    for field in ("source_sha256",):
        value = manifest.get(field)
        if not isinstance(value, str) or not HASH.fullmatch(value):
            raise ArtifactError(f"invalid {field}")
    for field in ("tilelang_version", "tvm_ffi_version"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise ArtifactError(f"missing {field}")
    if not isinstance(manifest.get("op_set"), list) or any(
        not isinstance(name, str) for name in manifest["op_set"]
    ):
        raise ArtifactError("invalid frontend operator set")
    files = manifest.get("files")
    if not isinstance(files, dict) or {"kernel.cubin", "kernel.tirx.json"} - files.keys():
        raise ArtifactError("missing executable or frontend IR")
    for name, digest in files.items():
        if (not isinstance(name, str) or not (name in ("kernel.cubin", "kernel.tirx.json")
            or re.fullmatch(r"licenses/[A-Za-z0-9_.-]+", name))):
            raise ArtifactError(f"invalid payload name: {name}")
        if not isinstance(digest, str) or not HASH.fullmatch(digest):
            raise ArtifactError(f"invalid payload hash: {name}")
    return manifest


def read_artifact(path: str | Path) -> tuple[dict, dict[str, bytes]]:
    """Verify all members before any binary is handed to the CUDA driver."""
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            names = [item.filename for item in members]
            if len(names) != len(set(names)) or len(names) > 1024:
                raise ArtifactError("duplicate or excessive archive members")
            if sum(item.file_size for item in members) > MAX_UNCOMPRESSED:
                raise ArtifactError("artifact exceeds the 128 MiB limit")
            if "manifest.json" not in names or archive.getinfo("manifest.json").file_size > MAX_MANIFEST:
                raise ArtifactError("missing or oversized manifest")
            manifest = validate_manifest(json.loads(archive.read("manifest.json")))
            if set(names) != {"manifest.json", *manifest["files"]}:
                raise ArtifactError("archive members do not match the manifest")
            files = {name: archive.read(name) for name in manifest["files"]}
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid artifact: {exc}") from exc
    for name, content in files.items():
        if hashlib.sha256(content).hexdigest() != manifest["files"][name]:
            raise ArtifactError(f"payload hash mismatch: {name}")
    if not files["kernel.cubin"].startswith(b"\x7fELF"):
        raise ArtifactError("kernel.cubin is not an ELF CUDA binary")
    try:
        ir = json.loads(files["kernel.tirx.json"])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid frontend IR: {exc}") from exc
    if not isinstance(ir, dict) or "nodes" not in ir:
        raise ArtifactError("invalid frontend IR graph")
    return manifest, files
