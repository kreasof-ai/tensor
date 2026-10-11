"""Dependency resolution and project installation."""

from __future__ import annotations

import copy
from pathlib import Path
import os

from tensor.modules.errors import ModuleError
from tensor.modules.cache import _cached
from tensor.modules.content import _data
from tensor.modules.content import _directory
from tensor.modules.lock import _graph_record
from tensor.modules.archive import _package
from tensor.modules.lock import _validate_graph
from tensor.modules.cache import _atomic
from tensor.modules.cache import _cache_root
from tensor.modules.content import _canonical
from tensor.modules.lock import _lock
from tensor.modules.cache import _store
from tensor.modules.registry import PYPI_INDEX, fetch, reference


def _resolve(project, cache, locked=None, root_data=None, *, offline=False):

    raw = root_data or _directory(project)
    packages, active = {}, []

    def visit(data):
        name = data.manifest["name"]
        if name in active:
            raise ModuleError("dependency cycle: " + " -> ".join([*active, name]))
        active.append(name)
        manifest = copy.deepcopy(data.manifest)
        for child, spec in manifest["dependencies"].items():
            candidate = (
                data.origin / spec["path"] if data.origin and "path" in spec else None
            )
            if "pypi" in spec:
                saved_active, saved_packages = len(active), dict(packages)
                try:
                    # Verify the whole cached closure before deciding it needs repair.
                    dependency = visit(_cached(cache, spec["sha256"]))
                except (OSError, ModuleError):
                    del active[saved_active:]
                    packages.clear()
                    packages.update(saved_packages)
                    if offline:
                        raise ModuleError(
                            f"missing or corrupt cached registry dependency: {child}; install without --offline to restore it"
                        )

                    remote = spec["pypi"]
                    graph, archived, _ = fetch(
                        remote["distribution"],
                        spec["version"],
                        index=remote["index"],
                        wheel_sha256=remote["sha256"],
                    )
                    dependency = archived[graph["root"]]
                    for item in archived.values():
                        merge(item)
            elif candidate is not None and candidate.exists():
                if candidate.is_dir():
                    dependency = visit(_directory(candidate))
                else:
                    graph, archived = _package(candidate)
                    dependency = archived[graph["root"]]
                    for item in archived.values():
                        merge(item)
            else:
                digest = spec.get("sha256")
                if digest is None and locked:
                    digest = locked.get("packages", {}).get(child, {}).get("sha256")
                if digest is None:
                    raise ModuleError(f"missing dependency source: {child}")
                dependency = visit(_cached(cache, digest))
            if (
                dependency.manifest["name"] != child
                or dependency.manifest["version"] != spec["version"]
            ):
                raise ModuleError(f"dependency name/version mismatch: {child}")
            if spec.get("sha256", dependency.digest) != dependency.digest:
                raise ModuleError(f"dependency hash mismatch: {child}")
            spec["sha256"] = dependency.digest
        active.pop()
        result = _data(manifest, data.files)
        merge(result)
        return result

    def merge(data):
        name = data.manifest["name"]
        if name in packages and packages[name].digest != data.digest:
            raise ModuleError(f"conflicting resolutions for module {name}")
        packages[name] = data

    root = visit(raw)
    record = _graph_record(root.manifest["name"], packages)
    _validate_graph(record, packages)
    record["project_sha256"] = raw.digest
    return record, packages


def install(project=".", *, cache_dir=None, frozen=False, offline=False):

    project, cache = Path(project).resolve(), _cache_root(cache_dir)
    path = project / "tensor.lock"
    previous = _lock(path) if path.exists() else None
    if frozen and previous is None:
        raise ModuleError("--frozen requires tensor.lock")
    raw = _directory(project)
    if frozen and raw.digest != previous["project_sha256"]:
        raise ModuleError("tensor.lock is stale; run tensor install to update it")
    record, packages = _resolve(project, cache, previous, raw, offline=offline)
    if frozen and record != previous:
        raise ModuleError("tensor.lock is stale; run tensor install to update it")
    for data in packages.values():
        _store(cache, data)
    if not frozen:
        _atomic(path, _canonical(record))
    return {
        "status": "installed",
        "project": str(project),
        "lock": str(path),
        "frozen": frozen,
        "modules": record["packages"],
        "cache": str(cache.resolve()),
    }


def add(source, project=".", *, cache_dir=None, index_url=None):

    project, cache = Path(project).resolve(), _cache_root(cache_dir)
    raw = _directory(project)
    remote = None
    if isinstance(source, str) and source.startswith("pypi:"):
        distribution, version = reference(source)
        graph, packages, remote = fetch(
            distribution, version, index=index_url or PYPI_INDEX
        )
    else:
        if index_url is not None:
            raise ModuleError("--index-url applies only to pypi: references")
        source = Path(source).resolve()
        if source.is_dir():
            graph, packages = _resolve(source, cache)
        else:
            graph, packages = _package(source)
    dependency = packages[graph["root"]]
    name = dependency.manifest["name"]
    if name == raw.manifest["name"]:
        raise ModuleError("a project cannot depend on itself")
    # Store the verified closure first; resolution can then use its exact hash.
    for data in packages.values():
        _store(cache, data)
    updated = copy.deepcopy(raw.manifest)
    updated["dependencies"][name] = {
        "version": dependency.manifest["version"],
        "sha256": dependency.digest,
    }
    if remote is not None:
        updated["dependencies"][name]["pypi"] = remote
    else:
        updated["dependencies"][name]["path"] = Path(
            os.path.relpath(source, project)
        ).as_posix()
    staged = _data(updated, raw.files, project)
    record, resolved = _resolve(project, cache, root_data=staged)
    for data in resolved.values():
        _store(cache, data)
    manifest_path, lock_path = project / "tensor.json", project / "tensor.lock"
    original, old_lock = (
        manifest_path.read_bytes(),
        lock_path.read_bytes() if lock_path.exists() else None,
    )
    try:
        _atomic(manifest_path, _canonical(updated))
        _atomic(lock_path, _canonical(record))
    except BaseException:
        _atomic(manifest_path, original)
        if old_lock is not None:
            _atomic(lock_path, old_lock)
        else:
            lock_path.unlink(missing_ok=True)
        raise
    return {
        "status": "added",
        "name": name,
        "version": dependency.manifest["version"],
        "sha256": dependency.digest,
        "lock": str(lock_path),
        **({"pypi": remote} if remote else {}),
    }
