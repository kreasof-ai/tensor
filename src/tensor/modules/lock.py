"""Pinned dependency graphs and lock validation."""

from __future__ import annotations

from pathlib import Path

from tensor.modules.errors import ModuleError
from tensor.modules.manifest import NAME
from tensor.modules.manifest import HASH
from tensor.modules.manifest import VERSION
from tensor.modules.content import _json


def _graph_record(root, packages):
    return {
        "format": "tensor.lock",
        "formatVersion": 1,
        "root": root,
        "packages": {
            name: {
                "version": d.manifest["version"],
                "sha256": d.digest,
                "dependencies": {
                    n: s["sha256"] for n, s in d.manifest["dependencies"].items()
                },
            }
            for name, d in sorted(packages.items())
        },
    }


def _validate_graph(record, packages):

    if (
        not isinstance(record, dict)
        or set(record) != {"format", "formatVersion", "root", "packages"}
        or record["format"] != "tensor.lock"
        or type(record["formatVersion"]) is not int
        or record["formatVersion"] != 1
        or not isinstance(record["root"], str)
        or not NAME.fullmatch(record["root"])
        or len(packages) > 128
    ):
        raise ModuleError("invalid or excessive package graph")
    if (
        record != _graph_record(record.get("root"), packages)
        or record["root"] not in packages
    ):
        raise ModuleError("package graph does not match its manifests")
    visited, active = set(), set()

    def visit(name):
        if name in active:
            raise ModuleError(f"dependency cycle at {name}")
        if name in visited:
            return
        active.add(name)
        for child, spec in packages[name].manifest["dependencies"].items():
            data = packages.get(child)
            if (
                data is None
                or data.digest != spec["sha256"]
                or data.manifest["version"] != spec["version"]
            ):
                raise ModuleError(f"unresolved or conflicting dependency: {child}")
            visit(child)
        active.remove(name)
        visited.add(name)

    visit(record["root"])
    if visited != packages.keys():
        raise ModuleError("package graph contains unreachable modules")


def _lock(path):

    value = _json(Path(path).read_bytes())
    if (
        not isinstance(value, dict)
        or set(value)
        != {"format", "formatVersion", "root", "packages", "project_sha256"}
        or value["format"] != "tensor.lock"
        or type(value["formatVersion"]) is not int
        or value["formatVersion"] != 1
        or not isinstance(value["root"], str)
        or not NAME.fullmatch(value["root"])
        or not isinstance(value["project_sha256"], str)
        or not HASH.fullmatch(value["project_sha256"])
        or not isinstance(value["packages"], dict)
        or not 1 <= len(value["packages"]) <= 128
    ):
        raise ModuleError("invalid tensor.lock format")
    for name, spec in value["packages"].items():
        if (
            not NAME.fullmatch(name)
            or not isinstance(spec, dict)
            or set(spec) != {"version", "sha256", "dependencies"}
            or not isinstance(spec["sha256"], str)
            or not HASH.fullmatch(spec["sha256"])
            or not isinstance(spec["version"], str)
            or not VERSION.fullmatch(spec["version"])
            or not isinstance(spec["dependencies"], dict)
        ):
            raise ModuleError("invalid locked module")
    return value
