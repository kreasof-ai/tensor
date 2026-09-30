"""Tensor modules, pinned dependency graphs and verified export loading.

Package operations read bytes and metadata only. Source/TIRx execution requires
an explicit compile request; native export loading needs no frontend imports.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import importlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

from tensor.runtime.abi import ABI_MAJOR, ABI_MINOR
from tensor.artifacts.format import ArtifactError, read_artifact

NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
EXPORT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
VERSION = re.compile(r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
MAX_BYTES = 512 * 1024 * 1024
MAX_FILES = 4096


class ModuleError(ValueError):
    pass


def _canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


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


def _path(value):
    if not isinstance(value, str) or not value or "\\" in value or any(c in value for c in ':<>|?*\0'):
        raise ModuleError(f"invalid package path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ModuleError(f"package path must stay inside its module: {value!r}")
    for part in path.parts:
        if (part.endswith((".", " ")) or part.split(".")[0].upper() in
                {"CON", "PRN", "AUX", "NUL", *{f"COM{i}" for i in range(1,10)}, *{f"LPT{i}" for i in range(1,10)}}):
            raise ModuleError(f"package path is not portable: {value!r}")
    return path.as_posix()


def _manifest(value):
    required = {"formatVersion", "name", "version", "tensorAbi", "exports"}
    if not isinstance(value, dict) or required - value.keys() or value.keys() - (required | {"dependencies", "capabilities", "files"}):
        raise ModuleError("tensor.json needs formatVersion, name, version, tensorAbi and exports")
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
        if not isinstance(spec, dict) or not spec or spec.keys() - {"source", "portable", "artifacts"}:
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
            if any(not p.endswith(".tbin") for p in normalized["artifacts"]) or len(set(normalized["artifacts"])) != len(paths):
                raise ModuleError(f"{name}: invalid or duplicate artifact paths")
        exports[name] = normalized
    dependencies = result.setdefault("dependencies", {})
    if not isinstance(dependencies, dict) or len(dependencies) > 128:
        raise ModuleError("dependencies must be an object")
    for name, spec in dependencies.items():
        if (not NAME.fullmatch(name) or not isinstance(spec, dict) or spec.keys() - {"path", "version", "sha256", "pypi"}
                or not isinstance(spec.get("version"), str) or not VERSION.fullmatch(spec["version"])
                or not ({"path", "sha256", "pypi"} & spec.keys())):
            raise ModuleError(f"{name}: dependency needs an exact version and local path, PyPI origin or SHA-256")
        if "pypi" in spec:
            from tensor.artifacts.registry import origin
            if "path" in spec or "sha256" not in spec:
                raise ModuleError("PyPI dependencies require a module hash and cannot also have a local path")
            spec["pypi"] = origin(spec["pypi"])
        if "path" in spec and (not isinstance(spec["path"], str) or not spec["path"] or "://" in spec["path"]):
            raise ModuleError("dependency paths must identify local directories or .tpack files")
        if "sha256" in spec and (not isinstance(spec["sha256"], str) or not HASH.fullmatch(spec["sha256"])):
            raise ModuleError("invalid dependency SHA-256")
    for field in ("capabilities", "files"):
        values = result.setdefault(field, [])
        if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or len(values) != len(set(values)):
            raise ModuleError(f"{field} must contain distinct strings")
    result["files"] = [_path(p) for p in result["files"]]
    if len(result["files"]) != len(set(result["files"])):
        raise ModuleError("duplicate normalized package file paths")
    return result


def _members(manifest):
    paths = {"tensor.json", *manifest["files"]}
    for export in manifest["exports"].values():
        paths.update(export.get("artifacts", []))
        paths.update(export[k] for k in ("source", "portable") if k in export)
    prefixes = {}
    for path in paths:
        for index in range(1,len(PurePosixPath(path).parts)+1):
            prefix = '/'.join(PurePosixPath(path).parts[:index])
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
    return {"arguments": [{**{k:v for k,v in a.items() if k != "alignment"},
                           "kind": a.get("kind", "buffer")} for a in manifest["arguments"]],
            "outputs": manifest.get("outputs", []), "symbols": manifest.get("symbols", {})}


def _data(manifest, files, origin=None):
    manifest = _manifest(manifest)
    files = {**files, "tensor.json": _canonical(manifest)}
    if set(files) != _members(manifest) or sum(map(len, files.values())) > MAX_BYTES:
        raise ModuleError("module file set or size does not match its manifest")
    for spec in manifest["exports"].values():
        signatures, targets = set(), set()
        for name in dict.fromkeys([*spec.get("artifacts", []), *([spec["portable"]] if "portable" in spec else [])]):
            artifact, _ = read_artifact(io.BytesIO(files[name]))
            logical = _logical(artifact)
            signatures.add(_canonical(logical))
            if "source" in spec and artifact["source_sha256"] != hashlib.sha256(files[spec["source"]]).hexdigest():
                raise ModuleError("export artifact does not match its packaged source")
            if name in spec.get("artifacts", []):
                target = (artifact.get("provider", "cuda"), artifact["target"])
                if target in targets:
                    raise ModuleError("ambiguous exact-target export artifacts")
                targets.add(target)
        if len(signatures) > 1:
            raise ModuleError("export artifact signatures disagree")
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in sorted(files.items())}
    digest = hashlib.sha256(_canonical(hashes)).hexdigest()
    return _Data(manifest, files, digest, origin)


def _directory(path):
    root = Path(path).resolve()
    if (root / "tensor.json").stat().st_size > 2*1024*1024:
        raise ModuleError("module manifest exceeds size limit")
    manifest = _manifest(_json((root / "tensor.json").read_bytes()))
    files = {}
    total = 0
    for name in _members(manifest):
        candidate = root / name
        if any(p.is_symlink() for p in (candidate, *candidate.parents) if p != root and root in p.parents):
            raise ModuleError(f"symlink package member: {name}")
        if not candidate.is_file() or not candidate.resolve().is_relative_to(root):
            raise ModuleError(f"missing or external package member: {name}")
        total += candidate.stat().st_size
        if total > MAX_BYTES:
            raise ModuleError("package member exceeds size limit")
        files[name] = candidate.read_bytes()
    return _data(manifest, files, root)


def _cache_root(override=None):
    return Path(override or os.environ.get("TENSOR_MODULE_CACHE") or Path.home() / ".cache/tensor/modules")


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
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tensor-", delete=False) as stream:
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
        if destination.is_symlink() or any(p.is_symlink() for p in destination.parents if p != cache and cache in p.parents):
            raise ModuleError("module cache contains a symlink")
        _atomic(destination, contents)
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(root).as_posix() not in {*data.files, "content.json"}:
            path.unlink()
    _atomic(root / "content.json", _canonical({"sha256": data.digest}))
    _cached(cache, data.digest)


def _graph_record(root, packages):
    return {"format": "tensor.lock", "formatVersion": 1, "root": root,
            "packages": {name: {"version": d.manifest["version"], "sha256": d.digest,
                "dependencies": {n: s["sha256"] for n,s in d.manifest["dependencies"].items()}}
                for name,d in sorted(packages.items())}}


def _validate_graph(record, packages):
    if (not isinstance(record,dict) or set(record) != {"format","formatVersion","root","packages"}
            or record["format"] != "tensor.lock" or type(record["formatVersion"]) is not int or record["formatVersion"] != 1
            or not isinstance(record["root"],str) or not NAME.fullmatch(record["root"]) or len(packages) > 128):
        raise ModuleError("invalid or excessive package graph")
    if record != _graph_record(record.get("root"), packages) or record["root"] not in packages:
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
            if data is None or data.digest != spec["sha256"] or data.manifest["version"] != spec["version"]:
                raise ModuleError(f"unresolved or conflicting dependency: {child}")
            visit(child)
        active.remove(name)
        visited.add(name)
    visit(record["root"])
    if visited != packages.keys():
        raise ModuleError("package graph contains unreachable modules")


def _archive(path):
    try:
        with zipfile.ZipFile(path) as bundle:
            infos = bundle.infolist()
            names = [i.filename for i in infos]
            if len(names) != len(set(names)) or len(names) > MAX_FILES or sum(i.file_size for i in infos) > MAX_BYTES:
                raise ModuleError("duplicate, excessive or oversized package members")
            if any(_path(n) != n for n in names) or "package.json" not in names or bundle.getinfo("package.json").file_size > 2*1024*1024:
                raise ModuleError("invalid package archive paths/index")
            index = _json(bundle.read("package.json"))
            if (not isinstance(index, dict) or set(index) != {"format", "formatVersion", "graph", "files"}
                    or index["format"] != "tensor.package" or type(index["formatVersion"]) is not int or index["formatVersion"] != 1
                    or not isinstance(index["files"], dict) or set(names) != {"package.json", *index["files"]}):
                raise ModuleError("unsupported package archive format or file index")
            files = {}
            for name, digest in index["files"].items():
                contents = bundle.read(name)
                if hashlib.sha256(contents).hexdigest() != digest:
                    raise ModuleError(f"package payload hash mismatch: {name}")
                files[name] = contents
            graph = index["graph"]
            if not isinstance(graph, dict) or not isinstance(graph.get("packages"), dict) or len(graph["packages"]) > 128:
                raise ModuleError("invalid package dependency graph")
            packages = {}
            consumed = set()
            for name in graph["packages"]:
                if not NAME.fullmatch(name):
                    raise ModuleError("invalid packaged module name")
                prefix = f"modules/{name}/"
                members = {n[len(prefix):]: value for n,value in files.items() if n.startswith(prefix)}
                data = _data(_json(members["tensor.json"]), members)
                if data.manifest["name"] != name:
                    raise ModuleError("package module name mismatch")
                packages[name] = data
                consumed.update(prefix+n for n in members)
            if consumed != files.keys():
                raise ModuleError("unindexed package members")
            _validate_graph(graph, packages)
            return graph, packages
    except (zipfile.BadZipFile, KeyError, TypeError) as exc:
        raise ModuleError(f"invalid package archive: {exc}") from exc


def _package(path):
    if Path(path).suffix == ".whl":
        from tensor.artifacts.registry import read_wheel
        return read_wheel(path)
    return _archive(path)


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
            candidate = data.origin / spec["path"] if data.origin and "path" in spec else None
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
                        raise ModuleError(f"missing or corrupt cached registry dependency: {child}; install without --offline to restore it")
                    from tensor.artifacts.registry import fetch
                    remote = spec["pypi"]
                    graph, archived, _ = fetch(remote["distribution"], spec["version"],
                        index=remote["index"], wheel_sha256=remote["sha256"])
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
            if dependency.manifest["name"] != child or dependency.manifest["version"] != spec["version"]:
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


def _lock(path):
    value = _json(Path(path).read_bytes())
    if (not isinstance(value, dict) or set(value) != {"format", "formatVersion", "root", "packages", "project_sha256"}
            or value["format"] != "tensor.lock" or type(value["formatVersion"]) is not int or value["formatVersion"] != 1
            or not isinstance(value["root"], str) or not NAME.fullmatch(value["root"])
            or not isinstance(value["project_sha256"], str) or not HASH.fullmatch(value["project_sha256"])
            or not isinstance(value["packages"], dict) or not 1 <= len(value["packages"]) <= 128):
        raise ModuleError("invalid tensor.lock format")
    for name, spec in value["packages"].items():
        if (not NAME.fullmatch(name) or not isinstance(spec, dict) or set(spec) != {"version", "sha256", "dependencies"}
                or not isinstance(spec["sha256"], str) or not HASH.fullmatch(spec["sha256"])
                or not isinstance(spec["version"], str) or not VERSION.fullmatch(spec["version"])
                or not isinstance(spec["dependencies"], dict)):
            raise ModuleError("invalid locked module")
    return value


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
    return {"status": "installed", "project": str(project), "lock": str(path), "frozen": frozen,
            "modules": record["packages"], "cache": str(cache.resolve())}


def add(source, project=".", *, cache_dir=None, index_url=None):
    project, cache = Path(project).resolve(), _cache_root(cache_dir)
    raw = _directory(project)
    remote = None
    if isinstance(source, str) and source.startswith("pypi:"):
        from tensor.artifacts.registry import PYPI_INDEX, fetch, reference
        distribution, version = reference(source)
        graph, packages, remote = fetch(distribution, version, index=index_url or PYPI_INDEX)
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
    updated["dependencies"][name] = {"version": dependency.manifest["version"], "sha256": dependency.digest}
    if remote is not None:
        updated["dependencies"][name]["pypi"] = remote
    else:
        updated["dependencies"][name]["path"] = Path(os.path.relpath(source, project)).as_posix()
    staged = _data(updated, raw.files, project)
    record, resolved = _resolve(project, cache, root_data=staged)
    for data in resolved.values():
        _store(cache, data)
    manifest_path, lock_path = project / "tensor.json", project / "tensor.lock"
    original, old_lock = manifest_path.read_bytes(), lock_path.read_bytes() if lock_path.exists() else None
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
    return {"status": "added", "name": name, "version": dependency.manifest["version"],
            "sha256": dependency.digest, "lock": str(lock_path), **({"pypi": remote} if remote else {})}


def pack(project, out, *, cache_dir=None):
    record, packages = _resolve(Path(project).resolve(), _cache_root(cache_dir))
    graph = {k:v for k,v in record.items() if k != "project_sha256"}
    files = {f"modules/{name}/{path}": contents for name,data in packages.items() for path,contents in data.files.items()}
    index = {"format": "tensor.package", "formatVersion": 1, "graph": graph,
             "files": {n:hashlib.sha256(v).hexdigest() for n,v in sorted(files.items())}}
    if len(files) >= MAX_FILES or sum(map(len, files.values())) + len(_canonical(index)) > MAX_BYTES:
        raise ModuleError("package exceeds archive limits")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with out.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, contents in sorted({**files, "package.json": _canonical(index)}.items()):
                    info = zipfile.ZipInfo(name, (1980,1,1,0,0,0))
                    info.create_system = 3
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o100644 << 16
                    archive.writestr(info, contents)
        _archive(out)
    except BaseException:
        if created:
            out.unlink(missing_ok=True)
        raise
    return {"status": "packed", "path": str(out.resolve()), "bytes": out.stat().st_size,
            "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "modules": graph["packages"]}


class Project:
    def __init__(self, path=".", *, cache_dir=None):
        self.path, self.cache = Path(path).resolve(), _cache_root(cache_dir)
        try:
            self._record = _lock(self.path / "tensor.lock")
        except FileNotFoundError as exc:
            raise ModuleError("project is not installed; run tensor install") from exc
        if _directory(self.path).digest != self._record["project_sha256"]:
            raise ModuleError("project changed since installation; run tensor install")
        packages = {n:_cached(self.cache,s["sha256"]) for n,s in self._record["packages"].items()}
        graph = {k:v for k,v in self._record.items() if k != "project_sha256"}
        _validate_graph(graph, packages)

    def module(self, name=None):
        name = name or self._record["root"]
        if name not in self._record["packages"]:
            raise ModuleError(f"module is not installed: {name}")
        return Module(self.cache, self._record["packages"][name]["sha256"])


class Module:
    def __init__(self, cache, digest):
        self.cache, self.digest = Path(cache), digest

    @property
    def manifest(self):
        return _cached(self.cache, self.digest).manifest

    def resolve(self, export, *, provider="cuda", target=None, compile=False,
                compiler=None, nvcc=None, nvrtc_home=None, cache_dir=None):
        data = _cached(self.cache, self.digest)
        if export not in data.manifest["exports"]:
            raise ModuleError(f"unknown export: {export}")
        from tensor.providers import PROVIDERS
        if provider not in PROVIDERS:
            raise ModuleError(f"unknown runtime provider: {provider}")
        implementation = importlib.import_module(PROVIDERS[provider]).Device
        missing = set(data.manifest["capabilities"]) - implementation.capabilities
        if missing:
            raise ModuleError(f"provider {provider} lacks module capabilities: {sorted(missing)}")
        if target is None:
            if provider == "cpu":
                target = "cpu-linux-x86_64"
            elif provider == "webgpu":
                from tensor.providers.webgpu_contract import TARGET as WEBGPU_TARGET
                target = WEBGPU_TARGET
            else:
                from tensor.cli.doctor import check_device
                device = check_device(0)
                if device["status"] != "ok":
                    raise ModuleError("no CUDA device; supply an exact target")
                target = device["arch"]
        from tensor.artifacts.format import TARGET
        if ((provider == "cuda" and (not isinstance(target,str) or not TARGET.fullmatch(target)))
                or (provider == "cpu" and target != "cpu-linux-x86_64")
                or (provider == "webgpu" and target != "webgpu-portable-v1")):
            raise ModuleError("target does not match the requested provider")
        root = self.cache / "packages" / self.digest
        spec = data.manifest["exports"][export]
        binaries, candidates = {}, []
        signature = None
        for name in dict.fromkeys([*spec.get("artifacts", []), *([spec["portable"]] if "portable" in spec else [])]):
            manifest, files = read_artifact(root / name)
            logical = _logical(manifest)
            if signature is not None and signature != logical:
                raise ModuleError("export artifact signatures disagree")
            signature = logical
            if "source" in spec and manifest["source_sha256"] != hashlib.sha256(data.files[spec["source"]]).hexdigest():
                raise ModuleError("export artifact does not match its packaged source")
            if name in spec.get("artifacts", []):
                key = (manifest.get("provider", "cuda"), manifest["target"])
                if key in binaries:
                    raise ModuleError("ambiguous exact-target export artifacts")
                binaries[key] = name
            candidates.append({"path": name, "provider": manifest.get("provider", "cuda"), "target": manifest["target"]})
        if (provider,target) in binaries:
            return {"path": str(root / binaries[provider,target]), "selection": "packaged", "module_sha256": self.digest,
                    "provider": provider, "target": target, "export": export}
        selected_compiler = compiler or ("wgsl" if provider == "webgpu" else "native" if provider == "cpu" else "nvcc" if nvcc else "nvrtc")
        key = hashlib.sha256(_canonical({"module": self.digest,"export": export,"provider": provider,
                                        "target": target,"compiler": selected_compiler,"abi": [ABI_MAJOR,ABI_MINOR]})).hexdigest()
        output = self.cache / "artifacts" / f"{key}.tbin"
        metadata = output.with_suffix(".json")
        try:
            record = _json(metadata.read_bytes())
            if record != {"sha256": hashlib.sha256(output.read_bytes()).hexdigest()}:
                raise ModuleError("cached export hash mismatch")
            cached, _ = read_artifact(output)
            if cached.get("provider","cuda") != provider or cached["target"] != target:
                raise ModuleError("cached export target mismatch")
            if "source" in spec and cached["source_sha256"] != hashlib.sha256(data.files[spec["source"]]).hexdigest():
                raise ModuleError("cached export source mismatch")
            logical = _logical(cached)
            if signature is not None and signature != logical:
                raise ModuleError("cached export signature mismatch")
            return {"path": str(output), "selection": "cached", "module_sha256": self.digest,
                    "provider": provider, "target": target, "export": export}
        except (OSError, ValueError, ArtifactError):
            pass
        if not compile:
            raise ModuleError(f"no exact-target artifact for {export} on {provider}/{target}; use --compile with the pinned compiler environment")
        source = spec.get("portable", spec.get("source"))
        if source is None:
            raise ModuleError("export has no portable representation or source to compile")
        from tensor.compiler.build import build_artifact
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent,prefix=".compile-") as directory:
            temporary = Path(directory) / "kernel.tbin"
            result = build_artifact(root / source, temporary, provider=provider, target=target,
                                    compiler=compiler, nvcc=nvcc, nvrtc_home=nvrtc_home, cache_dir=cache_dir)
            built, _ = read_artifact(temporary)
            logical = _logical(built)
            if signature is not None and signature != logical:
                raise ModuleError("compiled export signature differs from its portable contract")
            _atomic(output, temporary.read_bytes())
        _atomic(metadata, _canonical({"sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
        return {"path": str(output), "selection": "compiled", "module_sha256": self.digest,
                "provider": provider, "target": target, "export": export, "build": result}

    def load(self, export, device, **options):
        resolved = self.resolve(export, provider=device.info["provider"],
                                target=device.info["arch"], **options)
        return device.load(resolved["path"])


def resolve_reference(value, *, project=".", module_cache=None, **options):
    name, separator, export = value.partition("::")
    if not separator or not NAME.fullmatch(name) or not EXPORT.fullmatch(export):
        raise ModuleError("module references use module-name::export_name")
    return Project(project,cache_dir=module_cache).module(name).resolve(export, **options)


def cache_info(cache_dir=None):
    root = _cache_root(cache_dir)
    packages = [p for p in (root / "packages").glob("*") if p.is_dir() and HASH.fullmatch(p.name)]
    artifacts = list((root / "artifacts").glob("*.tbin"))
    return {"path": str(root.resolve()), "packages": len(packages), "compiled_exports": len(artifacts),
            "bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0}
