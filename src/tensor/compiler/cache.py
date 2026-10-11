"""Compiled-image cache identities, reads and writes."""

from __future__ import annotations

import os
from pathlib import Path
import hashlib
import json
import tempfile


def _cache_root(override: Path | None = None) -> Path:
    return Path(
        override
        or os.environ.get("TENSOR_CACHE_DIR")
        or (Path.home() / ".cache" / "tensor")
    )


def cache_info(cache_dir: Path | None = None) -> dict:
    root = _cache_root(cache_dir)
    entries = (
        [*root.glob("*.cubin"), *root.glob("*.so"), *root.glob("webgpu/*.tbin")]
        if root.is_dir()
        else []
    )
    return {
        "path": str(root.resolve()),
        "entries": len(entries),
        "bytes": sum(path.stat().st_size for path in entries),
    }


def _header_hash(includes: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for root in includes:
        digest.update(root.name.encode() + b"\0")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _cached_image(root: Path, key: str, extension: str = "cubin") -> bytes | None:
    binary, metadata = root / f"{key}.{extension}", root / f"{key}.json"
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        data = binary.read_bytes()
    except (OSError, ValueError):
        return None
    if (
        isinstance(record, dict)
        and record.get("key") == key
        and record.get("sha256") == hashlib.sha256(data).hexdigest()
        and data.startswith(b"\x7fELF")
    ):
        return data
    return None


def _write_cache(root: Path, key: str, data: bytes, extension: str = "cubin") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for suffix, content in (
        (f".{extension}", data),
        (
            ".json",
            json.dumps(
                {"key": key, "sha256": hashlib.sha256(data).hexdigest()}
            ).encode(),
        ),
    ):
        with tempfile.NamedTemporaryFile(
            dir=root, prefix=f".{key}-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            os.replace(temporary, root / f"{key}{suffix}")
        finally:
            temporary.unlink(missing_ok=True)
