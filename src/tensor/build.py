"""CUDA artifact producer for one TileLang kernel with typed runtime arguments."""

from __future__ import annotations

import hashlib
import json
import os
import re
import runpy
import subprocess
import tempfile
import time
import zipfile
from importlib.metadata import distribution, version
from pathlib import Path

from tensor.artifact import FORMAT_VERSION, validate_manifest
from tensor.doctor import TARGET, _resolve_nvcc, check_device, check_packages, check_provider


class BuildError(ValueError):
    pass


def _launch(value: object) -> dict:
    if not isinstance(value, dict) or set(value) != {"grid", "block", "shared_memory_bytes"}:
        raise BuildError("tensor_export()['launch'] needs grid, block, and shared_memory_bytes")
    result = {}
    for name in ("grid", "block"):
        dims = value[name]
        if not isinstance(dims, (tuple, list)) or len(dims) != 3 or any(
            type(x) is not int or x < 1 for x in dims
        ):
            raise BuildError(f"launch {name} must contain three positive integers")
        result[name] = list(dims)
    if result["block"][0] * result["block"][1] * result["block"][2] > 1024:
        raise BuildError("launch block exceeds 1024 threads")
    shared = value["shared_memory_bytes"]
    if type(shared) is not int or shared < 0:
        raise BuildError("shared_memory_bytes must be a non-negative integer")
    result["shared_memory_bytes"] = shared
    return result


def _op_set(serialized: str) -> list[str]:
    nodes = json.loads(serialized)["nodes"]

    def label(index: int) -> str:
        operation = nodes[index].get("data")
        if isinstance(operation, dict):
            for field in ("name", "global_name", "key", "op_name"):
                if isinstance(operation.get(field), str):
                    return operation[field]
            hint = operation.get("name_hint")
            if type(hint) is int and 0 <= hint < len(nodes):
                value = nodes[hint].get("data")
                if isinstance(value, str):
                    return value
        if isinstance(operation, str):
            return operation
        raise BuildError("cannot identify a frontend IR operator")

    ops = set()
    for node in nodes:
        if node.get("type") != "tirx.Call":
            continue
        index = node.get("data", {}).get("op")
        if type(index) is not int or not 0 <= index < len(nodes):
            raise BuildError("invalid operator reference in frontend IR")
        ops.add(label(index))
    return sorted(ops)


def _notices(tilelang_root: Path) -> dict[str, bytes]:
    result = {}
    header = tilelang_root / "3rdparty" / "cutlass" / "include" / "cutlass" / "cutlass.h"
    try:
        match = re.match(r"/\*.*?\*/", header.read_text(encoding="utf-8"), re.S)
    except OSError as exc:
        raise BuildError(f"CUTLASS redistribution notice unavailable: {exc}") from exc
    if not match or "SPDX-License-Identifier: BSD-3-Clause" not in match[0]:
        raise BuildError("CUTLASS redistribution notice missing from installed TileLang")
    result["licenses/cutlass-notice.txt"] = match[0].encode()
    dist = distribution("tilelang")
    for index, item in enumerate(dist.files or []):
        if "licenses" in item.parts:
            path = Path(dist.locate_file(item))
            if path.is_file():
                result[f"licenses/tilelang-{index}-{path.name}"] = path.read_bytes()
    return result


def _cache_root(override: Path | None = None) -> Path:
    return Path(override or os.environ.get("TENSOR_CACHE_DIR") or (Path.home() / ".cache" / "tensor"))


def cache_info(cache_dir: Path | None = None) -> dict:
    root = _cache_root(cache_dir)
    entries = list(root.glob("*.cubin")) if root.is_dir() else []
    return {"path": str(root.resolve()), "entries": len(entries),
            "bytes": sum(path.stat().st_size for path in entries)}


def _header_hash(includes: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for root in includes:
        digest.update(root.name.encode() + b"\0")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(path.relative_to(root).as_posix().encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


def _cached_cubin(root: Path, key: str) -> bytes | None:
    binary, metadata = root / f"{key}.cubin", root / f"{key}.json"
    try:
        record = json.loads(metadata.read_text(encoding="utf-8"))
        data = binary.read_bytes()
    except (OSError, ValueError):
        return None
    if (isinstance(record, dict) and record.get("key") == key
            and record.get("sha256") == hashlib.sha256(data).hexdigest()
            and data.startswith(b"\x7fELF")):
        return data
    return None


def _write_cache(root: Path, key: str, data: bytes) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for suffix, content in ((".cubin", data),
                            (".json", json.dumps({"key": key, "sha256": hashlib.sha256(data).hexdigest()}).encode())):
        with tempfile.NamedTemporaryFile(dir=root, prefix=f".{key}-", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        try:
            os.replace(temporary, root / f"{key}{suffix}")
        finally:
            temporary.unlink(missing_ok=True)


def build_artifact(source_path: Path, output_path: Path, *, target: str | None = None,
                   nvcc: str | None = None, cache_dir: Path | None = None) -> dict:
    source_path, output_path = Path(source_path), Path(output_path)
    if not source_path.is_file() or source_path.suffix != ".py":
        raise BuildError("source must be an existing Python file")
    if output_path.exists():
        raise BuildError(f"output already exists: {output_path}")
    if target is not None and not TARGET.fullmatch(target):
        raise BuildError("target must be an exact CUDA SM, e.g. sm_86")
    if target is None:
        device = check_device(0)
        if device["status"] != "ok":
            raise BuildError("no CUDA device found; pass --target sm_XX for a GPU-free build host")
        target = device["arch"]
    packages = check_packages()
    if packages["status"] != "ok":
        raise BuildError(packages["detail"])
    provider = check_provider()
    if provider["status"] != "ok":
        raise BuildError(provider["detail"])
    compiler = _resolve_nvcc(nvcc)
    started = time.perf_counter()
    # A source file is executable Python by design, exactly like a Python build script.
    try:
        namespace = runpy.run_path(str(source_path.resolve()))
    except Exception as exc:
        raise BuildError(f"source execution failed: {type(exc).__name__}: {exc}") from exc
    export = namespace.get("tensor_export")
    if not callable(export):
        raise BuildError("source must define tensor_export() returning {'kernel': ..., 'outputs': [...]}")
    try:
        spec = export()
    except Exception as exc:
        raise BuildError(f"tensor_export() failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(spec, dict) or "kernel" not in spec or set(spec) - {"kernel", "launch", "outputs"}:
        raise BuildError("tensor_export() needs 'kernel'; optional 'outputs' names the result buffers")

    # Compiler imports live in build only; `tensor doctor` and future consumers
    # can enter without importing TileLang or PyTorch.
    import tilelang
    import tvm
    from tvm.target import Target

    kernel = spec["kernel"]
    if not isinstance(kernel, tvm.tirx.PrimFunc):
        raise BuildError("tensor_export()['kernel'] must be a TileLang/TIRx PrimFunc")
    from tensor.lowering import device_signature, frontend_arguments

    symbols = {}
    try:
        arguments = frontend_arguments(kernel, symbols)
    except (ValueError, TypeError, AttributeError) as exc:
        raise BuildError(f"unsupported frontend signature: {exc}") from exc
    outputs = spec.get("outputs", [])
    argument_names = {argument["name"] for argument in arguments if argument["kind"] == "buffer"}
    if (not isinstance(outputs, list) or any(not isinstance(name, str) or name not in argument_names for name in outputs)
            or len(outputs) != len(set(outputs))):
        raise BuildError("outputs must be a list of distinct kernel buffer names")
    try:
        symbol = str(kernel.attrs["global_symbol"])
        ir = tvm.ir.save_json(tvm.IRModule({symbol: kernel}))
        resolved = Target({"kind": "cuda", "arch": target})
        with tilelang.transform.PassContext(opt_level=3), resolved:
            lowered = tilelang.lower(kernel, target=resolved, enable_device_compile=False)
        cuda = str(lowered.kernel_source)
    except Exception as exc:
        raise BuildError(f"TileLang lowering failed: {type(exc).__name__}: {exc}") from exc
    try:
        entrypoint, abi, launch = device_signature(kernel, lowered, cuda, symbols)
    except (ValueError, TypeError, AttributeError) as exc:
        raise BuildError(f"unsupported CUDA signature: {exc}") from exc
    if "launch" in spec and _launch(spec["launch"]) != launch:
        raise BuildError(f"declared launch {spec['launch']} disagrees with lowered kernel {launch}")
    tilelang_root = Path(tilelang.__file__).parent
    includes = (tilelang_root / "src", tilelang_root / "3rdparty" / "cutlass" / "include")
    for path in includes:
        if not path.is_dir():
            raise BuildError(f"required TileLang headers missing: {path}")
    try:
        nvcc_version = subprocess.run([compiler, "--version"], capture_output=True, text=True,
                                      check=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildError(f"CUDA compiler version check failed: {exc}") from exc
    cache_identity = {
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "frontend_sha256": hashlib.sha256(ir.encode()).hexdigest(),
        "headers_sha256": _header_hash(includes),
        "target": target, "compiler": compiler, "nvcc_version": nvcc_version,
        "tilelang_version": version("tilelang"), "tvm_ffi_version": version("apache-tvm-ffi"),
        "options": ["--cubin", "-std=c++20", "-O3", "-lineinfo"],
        "cuda_sha256": hashlib.sha256(cuda.encode()).hexdigest(),
    }
    key = hashlib.sha256(json.dumps(cache_identity, sort_keys=True).encode()).hexdigest()
    root = _cache_root(cache_dir)
    cubin = _cached_cubin(root, key)
    hit = cubin is not None
    nvcc_seconds = 0.0
    if cubin is None:
        compile_started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="tensor-build-") as directory:
            temporary_root = Path(directory)
            cuda_path, cubin_path = temporary_root / "kernel.cu", temporary_root / "kernel.cubin"
            cuda_path.write_text(cuda, encoding="utf-8")
            command = [compiler, "--cubin", "-std=c++20", "-O3", "-lineinfo", f"-arch={target}"]
            if os.name == "nt":
                command += ["-Xcompiler", "/Zc:preprocessor /Zc:__cplusplus"]
            command += ["-I", str(includes[0]), "-I", str(includes[1]),
                        str(cuda_path), "-o", str(cubin_path)]
            try:
                compiled = subprocess.run(command, capture_output=True, text=True, timeout=300)
            except (OSError, subprocess.SubprocessError) as exc:
                raise BuildError(f"nvcc failed to start or timed out: {exc}") from exc
            if compiled.returncode:
                raise BuildError(f"nvcc failed ({compiled.returncode}):\n{compiled.stderr or compiled.stdout}")
            cubin = cubin_path.read_bytes() if cubin_path.is_file() else b""
            if not cubin.startswith(b"\x7fELF"):
                raise BuildError("nvcc did not produce an ELF cubin")
        nvcc_seconds = time.perf_counter() - compile_started
        try:
            _write_cache(root, key, cubin)
        except OSError:
            pass  # An unwritable cache does not prevent a successful build.
    files = {"kernel.cubin": cubin, "kernel.tirx.json": ir.encode(), **_notices(tilelang_root)}
    manifest = {
        "format": "tensor.cuda", "format_version": FORMAT_VERSION, "kind": "cubin",
        "target": target, "entrypoint": entrypoint, "launch": launch,
        "arguments": arguments, "outputs": outputs, "symbols": symbols, "abi": abi,
        "source_sha256": cache_identity["source_sha256"],
        "tilelang_version": version("tilelang"),
        "tvm_ffi_version": version("apache-tvm-ffi"), "op_set": _op_set(ir),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    validate_manifest(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output_path.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, data in sorted(files.items()):
                    archive.writestr(name, data)
                archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
    except BaseException:
        if created:
            output_path.unlink(missing_ok=True)
        raise
    return {"status": "built", "path": str(output_path.resolve()), "bytes": output_path.stat().st_size,
            "target": target, "entrypoint": entrypoint, "arguments": arguments,
            "seconds": time.perf_counter() - started, "nvcc_seconds": nvcc_seconds,
            "cache_hit": hit, "cache_key": key, "gpu_execution": "unverified"}
