"""Verified module content and deterministic encoding."""

from __future__ import annotations

import json
from pathlib import PurePosixPath
from dataclasses import dataclass
from pathlib import Path
import hashlib
import io
from tensor.artifacts.format import read_artifact

from tensor.modules.errors import ModuleError
from tensor.modules.manifest import MAX_FILES
from tensor.modules.manifest import MAX_BYTES
from tensor.modules.manifest import _manifest


def _canonical(value):
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def _json(data):

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ModuleError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        return json.loads(data, object_pairs_hook=unique)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ModuleError(f"invalid module JSON: {exc}") from exc


def _members(manifest):

    paths = {"tensor.json", *manifest["files"]}
    for export in manifest["exports"].values():
        paths.update(export.get("artifacts", []))
        paths.update(export[k] for k in ("source", "portable") if k in export)
    prefixes = {}
    for path in paths:
        for index in range(1, len(PurePosixPath(path).parts) + 1):
            prefix = "/".join(PurePosixPath(path).parts[:index])
            if prefix.casefold() in prefixes and prefixes[prefix.casefold()] != prefix:
                raise ModuleError("case-colliding package paths")
            prefixes[prefix.casefold()] = prefix
    if len(paths) > MAX_FILES or "content.json" in paths:
        raise ModuleError("excessive, reserved or case-colliding package paths")
    return paths


@dataclass
class _Data:
    manifest: dict
    files: dict
    digest: str
    origin: Path | None = None


def _logical(manifest):
    return {
        "arguments": [
            {
                **{k: v for k, v in a.items() if k != "alignment"},
                "kind": a.get("kind", "buffer"),
            }
            for a in manifest["arguments"]
        ],
        "outputs": manifest.get("outputs", []),
        "symbols": manifest.get("symbols", {}),
    }


def _data(manifest, files, origin=None):

    manifest = _manifest(manifest)
    files = {**files, "tensor.json": _canonical(manifest)}
    if set(files) != _members(manifest) or sum(map(len, files.values())) > MAX_BYTES:
        raise ModuleError("module file set or size does not match its manifest")
    for spec in manifest["exports"].values():
        signatures, targets = set(), set()
        for name in dict.fromkeys(
            [
                *spec.get("artifacts", []),
                *([spec["portable"]] if "portable" in spec else []),
            ]
        ):
            artifact, _ = read_artifact(io.BytesIO(files[name]))
            logical = _logical(artifact)
            signatures.add(_canonical(logical))
            if (
                "source" in spec
                and artifact["source_sha256"]
                != hashlib.sha256(files[spec["source"]]).hexdigest()
            ):
                raise ModuleError("export artifact does not match its packaged source")
            if name in spec.get("artifacts", []):
                target = (artifact.get("provider", "cuda"), artifact["target"])
                if target in targets:
                    raise ModuleError("ambiguous exact-target export artifacts")
                targets.add(target)
        if len(signatures) > 1:
            raise ModuleError("export artifact signatures disagree")
    hashes = {
        name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())
    }
    digest = hashlib.sha256(_canonical(hashes)).hexdigest()
    return _Data(manifest, files, digest, origin)


def _directory(path):

    root = Path(path).resolve()
    if (root / "tensor.json").stat().st_size > 2 * 1024 * 1024:
        raise ModuleError("module manifest exceeds size limit")
    manifest = _manifest(_json((root / "tensor.json").read_bytes()))
    files = {}
    total = 0
    for name in _members(manifest):
        candidate = root / name
        if any(
            p.is_symlink()
            for p in (candidate, *candidate.parents)
            if p != root and root in p.parents
        ):
            raise ModuleError(f"symlink package member: {name}")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
            raise ModuleError(f"missing or external package member: {name}")
        total += candidate.stat().st_size
        if total > MAX_BYTES:
            raise ModuleError("package member exceeds size limit")
        files[name] = candidate.read_bytes()
    return _data(manifest, files, root)
