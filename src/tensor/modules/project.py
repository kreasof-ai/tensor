"""Installed projects and named export selection."""

from __future__ import annotations

from pathlib import Path
from tensor.modules.cache import _cache_root
from tensor.modules.cache import _cached
from tensor.modules.content import _directory
from tensor.modules.errors import ModuleError
from tensor.modules.lock import _lock
from tensor.modules.lock import _validate_graph
from tensor.modules.module import Module

from tensor.modules.manifest import EXPORT
from tensor.modules.manifest import NAME


class Project:
    def __init__(self, path=".", *, cache_dir=None):
        self.path, self.cache = Path(path).resolve(), _cache_root(cache_dir)
        try:
            self._record = _lock(self.path / "tensor.lock")
        except FileNotFoundError as exc:
            raise ModuleError("project is not installed; run tensor install") from exc
        if _directory(self.path).digest != self._record["project_sha256"]:
            raise ModuleError("project changed since installation; run tensor install")
        packages = {
            n: _cached(self.cache, s["sha256"])
            for n, s in self._record["packages"].items()
        }
        graph = {k: v for k, v in self._record.items() if k != "project_sha256"}
        _validate_graph(graph, packages)

    def module(self, name=None):
        name = name or self._record["root"]
        if name not in self._record["packages"]:
            raise ModuleError(f"module is not installed: {name}")
        return Module(self.cache, self._record["packages"][name]["sha256"])


def resolve_reference(value, *, project=".", module_cache=None, **options):

    name, separator, export = value.partition("::")
    if not separator or not NAME.fullmatch(name) or not EXPORT.fullmatch(export):
        raise ModuleError("module references use module-name::export_name")
    return (
        Project(project, cache_dir=module_cache).module(name).resolve(export, **options)
    )
