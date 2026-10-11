"""Atomic storage and verification of cached modules."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile

from tensor.modules.manifest import HASH
from tensor.modules.errors import ModuleError
from tensor.modules.content import _directory
from tensor.modules.content import _json
from tensor.modules.content import _canonical


def _cache_root(override=None):
    return Path(
        override
        or os.environ.get("TENSOR_MODULE_CACHE")
        or Path.home() / ".cache/tensor/modules"
    )


def _cached(cache, digest):

    if not HASH.fullmatch(digest):
        raise ModuleError("invalid cached package identity")
    root = cache / "packages" / digest
    record = _json((root / "content.json").read_bytes())
    data = _directory(root)
    if data.digest != digest or record != {"sha256": digest}:
        raise ModuleError("cached package hash mismatch; reinstall its verified source")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != {*data.files, "content.json"}:
        raise ModuleError("cached package contains undeclared files")
    data.origin = None
    return data


def _atomic(path, contents):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=".tensor-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(contents)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _store(cache, data):

    try:
        _cached(cache, data.digest)
        return
    except (OSError, ModuleError):
        pass
    root = cache / "packages" / data.digest
    # Every file is replaced atomically; content.json is the commit marker.
    # Readers verify all bytes and refuse incomplete or modified snapshots.
    root.mkdir(parents=True, exist_ok=True)
    for name, contents in data.files.items():
        destination = root / name
        if destination.is_symlink() or any(
            p.is_symlink()
            for p in destination.parents
            if p != cache and cache in p.parents
        ):
            raise ModuleError("module cache contains a symlink")
        _atomic(destination, contents)
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).as_posix() not in {
            *data.files,
            "content.json",
        }:
            path.unlink()
    _atomic(root / "content.json", _canonical({"sha256": data.digest}))
    _cached(cache, data.digest)


def cache_info(cache_dir=None):

    root = _cache_root(cache_dir)
    packages = [
        p
        for p in (root / "packages").glob("*")
        if p.is_dir() and HASH.fullmatch(p.name)
    ]
    artifacts = list((root / "artifacts").glob("*.tbin"))
    return {
        "path": str(root.resolve()),
        "packages": len(packages),
        "compiled_exports": len(artifacts),
        "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if root.exists()
        else 0,
    }
