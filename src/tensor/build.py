"""First CUDA artifact producer: one static, pointer-only TileLang kernel."""

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


def _arguments(kernel: object) -> list[dict]:
    try:
        params, buffers = kernel.params, kernel.buffer_map
        result = []
        for parameter in params:
            buffer = buffers[parameter]
            dimensions = [int(extent) for extent in buffer.shape]
            if not dimensions or any(extent < 1 for extent in dimensions):
                raise BuildError("buffer shapes must have positive static extents")
            result.append({"name": str(buffer.name), "dtype": str(buffer.dtype), "shape": dimensions})
        if not result:
            raise BuildError("the kernel has no buffer arguments")
        return result
    except BuildError:
        raise
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise BuildError("this build profile accepts only static buffer arguments") from exc


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


def _entrypoint(source: str, argument_count: int) -> str:
    definitions = re.findall(
        r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\([^)]+\)\s+)?'
        r'([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{', source, flags=re.S,
    )
    if len(definitions) != 1:
        raise BuildError(f"expected one generated CUDA kernel, found {len(definitions)}")
    name, parameters = definitions[0]
    arguments = [part.strip() for part in parameters.split(",") if part.strip()]
    if len(arguments) != argument_count or any("*" not in part for part in arguments):
        raise BuildError("generated CUDA signature does not match static pointer-only buffer arguments")
    return name


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


def build_artifact(source_path: Path, output_path: Path, *, target: str | None = None,
                   nvcc: str | None = None) -> dict:
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
        raise BuildError("source must define tensor_export() returning {'kernel': ..., 'launch': ...}")
    try:
        spec = export()
    except Exception as exc:
        raise BuildError(f"tensor_export() failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(spec, dict) or set(spec) != {"kernel", "launch"}:
        raise BuildError("tensor_export() must return exactly 'kernel' and 'launch'")
    launch = _launch(spec["launch"])

    # Compiler imports live in build only; `tensor doctor` and future consumers
    # can enter without importing TileLang or PyTorch.
    import tilelang
    import tvm
    from tilelang.tools.compile_only import compile_kernel_source

    kernel = spec["kernel"]
    if not isinstance(kernel, tvm.tirx.PrimFunc):
        raise BuildError("tensor_export()['kernel'] must be a TileLang/TIRx PrimFunc")
    arguments = _arguments(kernel)
    try:
        symbol = str(kernel.attrs["global_symbol"])
        ir = tvm.ir.save_json(tvm.IRModule({symbol: kernel}))
        cuda = compile_kernel_source(kernel, {"kind": "cuda", "arch": target})
    except Exception as exc:
        raise BuildError(f"TileLang lowering failed: {type(exc).__name__}: {exc}") from exc
    entrypoint = _entrypoint(cuda, len(arguments))
    tilelang_root = Path(tilelang.__file__).parent
    includes = (tilelang_root / "src", tilelang_root / "3rdparty" / "cutlass" / "include")
    for path in includes:
        if not path.is_dir():
            raise BuildError(f"required TileLang headers missing: {path}")
    with tempfile.TemporaryDirectory(prefix="tensor-build-") as directory:
        root = Path(directory)
        cuda_path, cubin_path = root / "kernel.cu", root / "kernel.cubin"
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
    files = {"kernel.cubin": cubin, "kernel.tirx.json": ir.encode(), **_notices(tilelang_root)}
    manifest = {
        "format": "tensor.cuda", "format_version": 1, "kind": "cubin",
        "target": target, "entrypoint": entrypoint, "launch": launch,
        "arguments": arguments, "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "tilelang_version": version("tilelang"),
        "tvm_ffi_version": version("apache-tvm-ffi"), "op_set": _op_set(ir),
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
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
            "seconds": time.perf_counter() - started, "gpu_execution": "unverified"}
