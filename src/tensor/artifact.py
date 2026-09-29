"""Read and validate Tensor executable bundles without compiler imports."""

from __future__ import annotations

import hashlib
import json
import re
import zipfile
from pathlib import Path

from tensor.signature import INTEGER_TYPES, SCALAR_TYPES, validate_expression

FORMAT = "tensor.module"
FORMAT_VERSION = 3
MAX_UNCOMPRESSED = 128 * 1024 * 1024
MAX_MANIFEST = 2 * 1024 * 1024
TARGET = re.compile(r"sm_[0-9]{2,3}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
NAME = re.compile(r"[A-Za-z_]\w*\Z")


class ArtifactError(ValueError):
    pass


def _expression(value, symbols: dict) -> None:
    try:
        validate_expression(value, symbols)
    except (ValueError, TypeError) as exc:
        raise ArtifactError(str(exc)) from exc


def validate_manifest(manifest: object) -> dict:
    if not isinstance(manifest, dict):
        raise ArtifactError("manifest must be an object")
    revision = manifest.get("format_version")
    if (type(revision) is not int or revision not in (1, 2, FORMAT_VERSION)
            or manifest.get("format") != (FORMAT if revision >= 3 else "tensor.cuda")):
        raise ArtifactError("unsupported Tensor artifact format/version")
    provider = manifest.get("provider", "cuda") if revision >= 3 else "cuda"
    if provider not in ("cuda", "cpu"):
        raise ArtifactError("unsupported runtime provider")
    if manifest.get("kind") != ("cubin" if provider == "cuda" else "native"):
        raise ArtifactError("invalid executable kind for provider")
    target = manifest.get("target")
    if (not isinstance(target, str) or
            (not TARGET.fullmatch(target) if provider == "cuda" else target != "cpu-linux-x86_64")):
        raise ArtifactError("invalid provider target")
    if revision >= 3:
        from tensor.abi import check_requirement
        try:
            check_requirement(manifest.get("runtime_abi"))
        except ValueError as exc:
            raise ArtifactError(str(exc)) from exc
        if "provider" not in manifest:
            raise ArtifactError("missing runtime provider")
        compiler = manifest.get("compiler")
        if not isinstance(compiler, dict) or not isinstance(compiler.get("name"), str) or not isinstance(compiler.get("version"), str):
            raise ArtifactError("missing compiler provenance")
    entrypoint = manifest.get("entrypoint")
    if not isinstance(entrypoint, str) or not NAME.fullmatch(entrypoint):
        raise ArtifactError("invalid executable entrypoint")
    symbols = manifest.get("symbols", {})
    if (not isinstance(symbols, dict) or len(symbols) > 128
            or any(not isinstance(name, str) or not NAME.fullmatch(name)
                   or not isinstance(dtype, str) or dtype not in INTEGER_TYPES
                   for name, dtype in symbols.items())):
        raise ArtifactError("invalid dimension symbols")
    if revision == 1 and symbols:
        raise ArtifactError("v1 artifacts cannot contain dimension symbols")
    launch = manifest.get("launch")
    if not isinstance(launch, dict) or set(launch) != {"grid", "block", "shared_memory_bytes"}:
        raise ArtifactError("invalid launch description")
    for field in ("grid", "block"):
        dimensions = launch[field]
        if not isinstance(dimensions, list) or len(dimensions) != 3:
            raise ArtifactError(f"{field} must contain three dimensions")
        for extent in dimensions:
            _expression(extent, symbols)
            if type(extent) is int and not 1 <= extent < (1 << 31):
                raise ArtifactError(f"{field} must have positive int32 dimensions")
            if revision == 1 and type(extent) is not int:
                raise ArtifactError("v1 launch dimensions must be static")
    block = launch["block"]
    if all(type(value) is int for value in block) and block[0] * block[1] * block[2] > 1024:
        raise ArtifactError("launch block exceeds 1024 threads")
    shared = launch["shared_memory_bytes"]
    _expression(shared, symbols)
    if (type(shared) is int and not 0 <= shared < (1 << 31)) or (revision == 1 and type(shared) is not int):
        raise ArtifactError("invalid shared memory size")
    arguments = manifest.get("arguments")
    if not isinstance(arguments, list) or not arguments or len(arguments) > 256:
        raise ArtifactError("invalid artifact argument count")
    names = []
    buffers = set()
    for argument in arguments:
        if not isinstance(argument, dict):
            raise ArtifactError("invalid argument")
        kind = argument.get("kind", "buffer")
        fields = {"name", "dtype", "shape"} if revision == 1 else (
            {"kind", "name", "dtype", "shape", "alignment"} if kind == "buffer" else {"kind", "name", "dtype"})
        if set(argument) != fields or kind not in ("buffer", "scalar") or (revision == 1 and kind != "buffer"):
            raise ArtifactError("invalid argument descriptor")
        name, dtype = argument["name"], argument["dtype"]
        if not isinstance(name, str) or not NAME.fullmatch(name):
            raise ArtifactError("invalid argument name")
        if not isinstance(dtype, str) or not dtype:
            raise ArtifactError("invalid argument dtype")
        if dtype not in {*SCALAR_TYPES, "float16"}:
            raise ArtifactError(f"unsupported argument dtype {dtype}")
        if kind == "scalar":
            if dtype not in SCALAR_TYPES or (name in symbols and symbols[name] != dtype):
                raise ArtifactError("invalid scalar argument dtype")
        else:
            if name in symbols:
                raise ArtifactError("buffer and dimension names must be distinct")
            buffers.add(name)
            shape = argument["shape"]
            if not isinstance(shape, list) or not shape or len(shape) > 64:
                raise ArtifactError("invalid buffer shape")
            for extent in shape:
                _expression(extent, symbols)
                if (type(extent) is int and extent < 1) or (revision == 1 and type(extent) is not int):
                    raise ArtifactError("invalid buffer shape")
            if revision >= 2:
                alignment = argument["alignment"]
                if type(alignment) is not int or not 1 <= alignment <= 4096 or alignment & (alignment-1):
                    raise ArtifactError("invalid buffer alignment")
        names.append(name)
    if len(names) != len(set(names)):
        raise ArtifactError("duplicate argument")
    if not buffers:
        raise ArtifactError("artifact has no buffer arguments")
    outputs = manifest.get("outputs", [])
    if (not isinstance(outputs, list) or any(not isinstance(name, str) or name not in buffers for name in outputs)
            or len(outputs) != len(set(outputs))):
        raise ArtifactError("invalid output names")
    if revision >= 2:
        abi = manifest.get("abi")
        if not isinstance(abi, list) or not abi or len(abi) > 256:
            raise ArtifactError("missing or invalid kernel ABI")
        frontend = {item["name"]: item for item in arguments}
        abi_names = []
        for item in abi:
            if not isinstance(item, dict) or set(item) != {"kind", "name", "dtype"}:
                raise ArtifactError("invalid kernel ABI argument")
            name, kind, dtype = item["name"], item["kind"], item["dtype"]
            if not isinstance(name, str) or kind not in ("scalar", "buffer") or not isinstance(dtype, str):
                raise ArtifactError("invalid kernel ABI argument")
            original = frontend.get(name)
            if original is not None:
                if kind != original["kind"] or dtype != original["dtype"]:
                    raise ArtifactError("kernel ABI does not match frontend argument")
            elif kind != "scalar" or symbols.get(name) != dtype:
                raise ArtifactError(f"unmapped kernel ABI argument {name}")
            abi_names.append(name)
        if len(abi_names) != len(set(abi_names)) or buffers - set(abi_names):
            raise ArtifactError("duplicate or missing kernel buffer ABI argument")
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
    image = "kernel.cubin" if provider == "cuda" else "kernel.so"
    if not isinstance(files, dict) or {image, "kernel.tirx.json"} - files.keys():
        raise ArtifactError("missing executable or frontend IR")
    for name, digest in files.items():
        if (not isinstance(name, str) or not (name in (image, "kernel.tirx.json")
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
    image = "kernel.cubin" if manifest.get("provider", "cuda") == "cuda" else "kernel.so"
    if not files[image].startswith(b"\x7fELF"):
        raise ArtifactError(f"{image} is not an ELF executable")
    try:
        ir = json.loads(files["kernel.tirx.json"])
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ArtifactError(f"invalid frontend IR: {exc}") from exc
    if not isinstance(ir, dict) or "nodes" not in ir:
        raise ArtifactError("invalid frontend IR graph")
    return manifest, files
