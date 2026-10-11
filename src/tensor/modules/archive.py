"""Deterministic module archives and verified unpacking."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from tensor.modules.manifest import MAX_BYTES
from tensor.modules.manifest import MAX_FILES
from tensor.modules.errors import ModuleError
from tensor.modules.manifest import NAME
from tensor.modules.content import _data
from tensor.modules.content import _json
from tensor.modules.manifest import _path
from tensor.modules.lock import _validate_graph
from tensor.modules.cache import _cache_root
from tensor.modules.content import _canonical


def _archive(path):

    try:
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            names = [i.filename for i in infos]
            if (
                len(names) != len(set(names))
                or len(names) > MAX_FILES
                or sum(i.file_size for i in infos) > MAX_BYTES
            ):
                raise ModuleError("duplicate, excessive or oversized package members")
            if (
                any(_path(n) != n for n in names)
                or "package.json" not in names
                or bundle.getinfo("package.json").file_size > 2 * 1024 * 1024
            ):
                raise ModuleError("invalid package archive paths/index")
            index = _json(bundle.read("package.json"))
            if (
                not isinstance(index, dict)
                or set(index) != {"format", "formatVersion", "graph", "files"}
                or index["format"] != "tensor.package"
                or type(index["formatVersion"]) is not int
                or index["formatVersion"] != 1
                or not isinstance(index["files"], dict)
                or set(names) != {"package.json", *index["files"]}
            ):
                raise ModuleError("unsupported package archive format or file index")
            files = {}
            for name, digest in index["files"].items():
                contents = bundle.read(name)
                if hashlib.sha256(contents).hexdigest() != digest:
                    raise ModuleError(f"package payload hash mismatch: {name}")
                files[name] = contents
            graph = index["graph"]
            if (
                not isinstance(graph, dict)
                or not isinstance(graph.get("packages"), dict)
                or len(graph["packages"]) > 128
            ):
                raise ModuleError("invalid package dependency graph")
            packages = {}
            consumed = set()
            for name in graph["packages"]:
                if not NAME.fullmatch(name):
                    raise ModuleError("invalid packaged module name")
                prefix = f"modules/{name}/"
                members = {
                    n[len(prefix) :]: value
                    for n, value in files.items()
                    if n.startswith(prefix)
                }
                data = _data(_json(members["tensor.json"]), members)
                if data.manifest["name"] != name:
                    raise ModuleError("package module name mismatch")
                packages[name] = data
                consumed.update(prefix + n for n in members)
            if consumed != files.keys():
                raise ModuleError("unindexed package members")
            _validate_graph(graph, packages)
            return graph, packages
    except (zipfile.BadZipFile, KeyError, TypeError) as exc:
        raise ModuleError(f"invalid package archive: {exc}") from exc


def _package(path):
    if Path(path).suffix == ".whl":
        from tensor.modules.registry import read_wheel

        return read_wheel(path)
    return _archive(path)


def pack(project, out, *, cache_dir=None):
    from tensor.modules.resolver import _resolve

    record, packages = _resolve(Path(project).resolve(), _cache_root(cache_dir))
    graph = {k: v for k, v in record.items() if k != "project_sha256"}
    files = {
        f"modules/{name}/{path}": contents
        for name, data in packages.items()
        for path, contents in data.files.items()
    }
    index = {
        "format": "tensor.package",
        "formatVersion": 1,
        "graph": graph,
        "files": {n: hashlib.sha256(v).hexdigest() for n, v in sorted(files.items())},
    }
    if (
        len(files) >= MAX_FILES
        or sum(map(len, files.values())) + len(_canonical(index)) > MAX_BYTES
    ):
        raise ModuleError("package exceeds archive limits")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with out.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, contents in sorted(
                    {**files, "package.json": _canonical(index)}.items()
                ):
                    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, contents)
        _archive(out)
    except BaseException:
        if created:
            out.unlink(missing_ok=True)
        raise
    return {
        "status": "packed",
        "path": str(out.resolve()),
        "bytes": out.stat().st_size,
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "modules": graph["packages"],
    }
