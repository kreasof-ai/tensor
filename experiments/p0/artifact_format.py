"""Strict, temporary format for the P0 CUDA artifact experiment; stdlib only.

This is not Tensor's public ABI. Source bundles and executable bundles are
distinct, and the runtime accepts only the latter. Hashes detect corruption,
not authenticity: use artifacts from a trusted producer.
"""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path, PurePosixPath

MAX_BYTES = 128 * 1024 * 1024


class ArtifactError(ValueError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest must be an object")
    if manifest.get("format") != "tensor.p0.cuda" or manifest.get("format_version") != 1:
        raise ArtifactError("unsupported artifact format/version")
    if manifest.get("kind") not in ("source", "cubin"):
        raise ArtifactError("expected a source bundle or cubin executable")
    if manifest.get("operation") != "relu_2a_plus_b" or manifest.get("dtype") != "float32":
        raise ArtifactError("this experiment supports only relu(2*a+b) on float32")
    size = manifest.get("size")
    if type(size) is not int or not 1 <= size <= 2**31 - 1:
        raise ArtifactError("size must be a positive int32 extent")
    if not isinstance(manifest.get("arch"), str) or not re.fullmatch(r"sm_[0-9]{2,3}", manifest["arch"]):
        raise ArtifactError("arch must be an exact generic SM target, e.g. sm_80")
    if manifest.get("entrypoint") != "elementwise_kernel":
        raise ArtifactError("unexpected entrypoint")
    if manifest.get("arguments") != ["a", "b", "c"]:
        raise ArtifactError("expected exactly three pointer arguments: a, b, c")
    expected = {"grid": [(size + 127) // 128, 1, 1], "block": [128, 1, 1], "shared_memory_bytes": 0}
    if manifest.get("launch") != expected:
        raise ArtifactError("launch geometry does not match the kernel contract")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ArtifactError("missing payload hashes")
    payload = "kernel.cu" if manifest["kind"] == "source" else "kernel.cubin"
    if payload not in files:
        raise ArtifactError(f"missing {payload}")
    for name, digest in files.items():
        parts = PurePosixPath(name).parts if isinstance(name, str) else ()
        allowed = name == payload or (parts and parts[0] == "licenses") or (
            manifest["kind"] == "source" and parts and parts[0] == "include")
        if (not allowed or "\\" in name or ":" in name or ".." in parts
                or PurePosixPath(name).is_absolute() or PurePosixPath(name).as_posix() != name):
            raise ArtifactError(f"invalid payload path: {name}")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ArtifactError(f"invalid hash for {name}")


def write_bundle(path: Path, manifest: dict, files: dict[str, bytes]) -> None:
    manifest = {**manifest, "files": {name: sha256(data) for name, data in files.items()}}
    validate(manifest)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous experiment artifacts.
    with path.open("xb") as output, zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(name, data)
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))


def read_bundle(path: Path, *, kind: str | None = None) -> tuple[dict, dict[str, bytes]]:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or len(names) > 10000:
                raise ArtifactError("duplicate members or too many files")
            if sum(info.file_size for info in infos) > MAX_BYTES:
                raise ArtifactError("artifact exceeds the 128 MiB experiment limit")
            if "manifest.json" not in names or archive.getinfo("manifest.json").file_size > 2 * 1024 * 1024:
                raise ArtifactError("missing or oversized manifest")
            manifest = json.loads(archive.read("manifest.json"))
            validate(manifest)
            if kind is not None and manifest["kind"] != kind:
                raise ArtifactError(f"expected {kind}; {manifest['kind']} bundles cannot be used here")
            if set(names) != {"manifest.json", *manifest["files"]}:
                raise ArtifactError("archive members do not match the manifest")
            files = {name: archive.read(name) for name in manifest["files"]}
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid artifact: {exc}") from exc
    for name, data in files.items():
        if sha256(data) != manifest["files"][name]:
            raise ArtifactError(f"hash mismatch: {name}")
    payload = files["kernel.cu" if manifest["kind"] == "source" else "kernel.cubin"]
    if not payload:
        raise ArtifactError("empty kernel payload")
    if manifest["kind"] == "cubin" and not payload.startswith(b"\x7fELF"):
        raise ArtifactError("kernel.cubin is not an ELF CUDA binary")
    return manifest, files
