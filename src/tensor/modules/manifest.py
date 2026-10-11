"""Module manifest rules and archive limits."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
import copy
from tensor.runtime.abi import ABI_MAJOR

from tensor.modules.errors import ModuleError


NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")


EXPORT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")


HASH = re.compile(r"[0-9a-f]{64}\Z")


MAX_BYTES = 512 * 1024 * 1024


MAX_FILES = 4096


def _path(value):

    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or any(c in value for c in ":<>|?*\0")
    ):
        raise ModuleError(f"invalid package path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ModuleError(f"package path must stay inside its module: {value!r}")
    for part in path.parts:
        if part.endswith((".", " ")) or part.split(".")[0].upper() in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *{f"COM{i}" for i in range(1, 10)},
            *{f"LPT{i}" for i in range(1, 10)},
        }:
            raise ModuleError(f"package path is not portable: {value!r}")
    return path.as_posix()


def _manifest(value):

    required = {"formatVersion", "name", "version", "tensorAbi", "exports"}
    if (
        not isinstance(value, dict)
        or required - value.keys()
        or value.keys() - (required | {"dependencies", "capabilities", "files"})
    ):
        raise ModuleError(
            "tensor.json needs formatVersion, name, version, tensorAbi and exports"
        )
    if type(value["formatVersion"]) is not int or value["formatVersion"] != 1:
        raise ModuleError("unsupported module manifest version")
    if not isinstance(value["name"], str) or not NAME.fullmatch(value["name"]):
        raise ModuleError("module name must be lowercase words separated by hyphens")
    _path(value["name"])
    if not isinstance(value["version"], str) or not VERSION.fullmatch(value["version"]):
        raise ModuleError("module version must be an exact major.minor.patch")
    if type(value["tensorAbi"]) is not int or value["tensorAbi"] != ABI_MAJOR:
        raise ModuleError("unsupported module Tensor ABI major")
    result = copy.deepcopy(value)
    exports = result["exports"]
    if not isinstance(exports, dict) or len(exports) > 256:
        raise ModuleError("exports must be an object with at most 256 names")
    for name, spec in list(exports.items()):
        if not EXPORT.fullmatch(name):
            raise ModuleError(f"invalid export name: {name}")
        if isinstance(spec, str):
            spec = {"source": spec}
        if (
            not isinstance(spec, dict)
            or not spec
            or spec.keys() - {"source", "portable", "artifacts"}
        ):
            raise ModuleError(f"{name}: export needs source, portable or artifacts")
        normalized = {}
        for field in ("source", "portable"):
            if field in spec:
                path = _path(spec[field])
                if not path.endswith(".py" if field == "source" else ".tbin"):
                    raise ModuleError(f"{name}: {field} has an invalid suffix")
                normalized[field] = path
        if "artifacts" in spec:
            paths = spec["artifacts"]
            if not isinstance(paths, list) or not paths or len(paths) > 64:
                raise ModuleError(f"{name}: artifacts must be a nonempty list")
            normalized["artifacts"] = [_path(p) for p in paths]
            if any(not p.endswith(".tbin") for p in normalized["artifacts"]) or len(
                set(normalized["artifacts"])
            ) != len(paths):
                raise ModuleError(f"{name}: invalid or duplicate artifact paths")
        exports[name] = normalized
    dependencies = result.setdefault("dependencies", {})
    if not isinstance(dependencies, dict) or len(dependencies) > 128:
        raise ModuleError("dependencies must be an object")
    for name, spec in dependencies.items():
        if (
            not NAME.fullmatch(name)
            or not isinstance(spec, dict)
            or spec.keys() - {"path", "version", "sha256", "pypi"}
            or not isinstance(spec.get("version"), str)
            or not VERSION.fullmatch(spec["version"])
            or not ({"path", "sha256", "pypi"} & spec.keys())
        ):
            raise ModuleError(
                f"{name}: dependency needs an exact version and local path, PyPI origin or SHA-256"
            )
        if "pypi" in spec:
            if "path" in spec or "sha256" not in spec:
                raise ModuleError(
                    "PyPI dependencies require a module hash and cannot also have a local path"
                )
            from tensor.modules.registry import origin

            spec["pypi"] = origin(spec["pypi"])
        if "path" in spec and (
            not isinstance(spec["path"], str)
            or not spec["path"]
            or "://" in spec["path"]
        ):
            raise ModuleError(
                "dependency paths must identify local directories or .tpack files"
            )
        if "sha256" in spec and (
            not isinstance(spec["sha256"], str) or not HASH.fullmatch(spec["sha256"])
        ):
            raise ModuleError("invalid dependency SHA-256")
    for field in ("capabilities", "files"):
        values = result.setdefault(field, [])
        if (
            not isinstance(values, list)
            or any(not isinstance(v, str) for v in values)
            or len(values) != len(set(values))
        ):
            raise ModuleError(f"{field} must contain distinct strings")
    result["files"] = [_path(p) for p in result["files"]]
    if len(result["files"]) != len(set(result["files"])):
        raise ModuleError("duplicate normalized package file paths")
    return result
