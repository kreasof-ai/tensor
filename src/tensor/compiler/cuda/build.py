"""Lower TileLang kernels and build verified CUDA artifacts."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from importlib.metadata import version
from pathlib import Path
from tensor.artifacts.format import FORMAT_VERSION, validate_manifest
from tensor.diagnostics import TARGET, check_device, check_packages, check_provider
from tensor.compiler.cuda.toolchain import select_compiler
from tensor.runtime.abi import runtime_requirement, workspace_requirement
from tensor.compiler.cache import _cache_root
from tensor.compiler.cache import _cached_image
from tensor.compiler.cache import _header_hash
from tensor.compiler.cache import _write_cache
from tensor.compiler.errors import BuildError
from tensor.compiler.frontend.ir import _op_set
from tensor.compiler.frontend.signature import _launch
from tensor.compiler.licenses import _notices


def build_cuda(
    source_path: Path,
    output_path: Path,
    *,
    target: str | None = None,
    nvcc: str | None = None,
    cache_dir: Path | None = None,
    compiler: str | None = None,
    nvrtc_home: str | Path | None = None,
    provider: str = "cuda",
) -> dict:
    source_path, output_path = Path(source_path), Path(output_path)
    if not source_path.is_file() or source_path.suffix not in (".py", ".tbin"):
        raise BuildError("source must be an existing Python file or portable .tbin")
    from tensor.compiler.frontend.portable import export_spec
    from tensor.artifacts.portable import preflight

    checked = preflight(source_path) if source_path.suffix == ".tbin" else None
    if output_path.exists():
        raise BuildError(f"output already exists: {output_path}")
    if target is not None and not TARGET.fullmatch(target):
        raise BuildError("target must be an exact CUDA SM, e.g. sm_86")
    if target is None:
        device = check_device(0)
        if device["status"] != "ok":
            raise BuildError(
                "no CUDA device found; pass --target sm_XX for a GPU-free build host"
            )
        target = device["arch"]
    packages = check_packages()
    if packages["status"] != "ok":
        raise BuildError(packages["detail"])
    provider_check = check_provider()
    if provider_check["status"] != "ok":
        raise BuildError(provider_check["detail"])
    try:
        backend = select_compiler(compiler, nvcc=nvcc, nvrtc_home=nvrtc_home)
    except ValueError as exc:
        raise BuildError(str(exc)) from exc
    started = time.perf_counter()
    # A source file is executable Python by design, exactly like a Python build script.
    try:
        spec = export_spec(source_path, checked)
    except Exception as exc:
        raise BuildError(
            f"source execution failed: {type(exc).__name__}: {exc}"
        ) from exc
    if (
        not isinstance(spec, dict)
        or "kernel" not in spec
        or set(spec) - {"kernel", "launch", "outputs"}
    ):
        raise BuildError(
            "tensor_export() needs 'kernel'; optional 'outputs' names the result buffers"
        )

    # Compiler imports live in build only; `tensor doctor` and future consumers
    # can enter without importing TileLang or PyTorch.
    import tilelang
    import tvm
    from tvm.target import Target

    kernel = spec["kernel"]
    if not isinstance(kernel, tvm.tirx.PrimFunc):
        raise BuildError("tensor_export()['kernel'] must be a TileLang/TIRx PrimFunc")
    from tensor.compiler.frontend.signature import device_signature, frontend_arguments

    symbols = {}
    try:
        arguments = frontend_arguments(kernel, symbols)
    except (ValueError, TypeError, AttributeError) as exc:
        raise BuildError(f"unsupported frontend signature: {exc}") from exc
    outputs = spec.get("outputs", [])
    argument_names = {
        argument["name"] for argument in arguments if argument["kind"] == "buffer"
    }
    if (
        not isinstance(outputs, list)
        or any(
            not isinstance(name, str) or name not in argument_names for name in outputs
        )
        or len(outputs) != len(set(outputs))
    ):
        raise BuildError("outputs must be a list of distinct kernel buffer names")
    try:
        symbol = str(kernel.attrs["global_symbol"])
        ir = tvm.ir.save_json(tvm.IRModule({symbol: kernel}))
        resolved = Target({"kind": "cuda", "arch": target})
        # ABI 1.3 passes ordinary pointers/scalars and launches one grid. Hopper
        # auto-TMA introduces CUtensorMap parameters, while WGMMA requires an
        # architecture-specific sm_90a target. Both Hopper paths retain the
        # pointer/scalar ABI; only sm_90a enables warp-group matrix instructions.
        pass_config = (
            {
                "tl.disable_tma_lower": True,
                "tl.disable_wgmma": target == "sm_90",
                "tl.disable_warp_specialized": True,
            }
            if target in ("sm_90", "sm_90a")
            else {}
        )
        with tilelang.transform.PassContext(opt_level=3, config=pass_config), resolved:
            lowered = tilelang.lower(
                kernel, target=resolved, enable_device_compile=False
            )
        from tensor.compiler.cuda.lowering import lower_cuda_intrinsics

        cuda = lower_cuda_intrinsics(str(lowered.kernel_source))
    except Exception as exc:
        raise BuildError(
            f"TileLang lowering failed: {type(exc).__name__}: {exc}"
        ) from exc
    try:
        entrypoint, abi, launch = device_signature(kernel, lowered, cuda, symbols)
    except (ValueError, TypeError, AttributeError) as exc:
        raise BuildError(f"unsupported CUDA signature: {exc}") from exc
    if "launch" in spec and _launch(spec["launch"]) != launch:
        raise BuildError(
            f"declared launch {spec['launch']} disagrees with lowered kernel {launch}"
        )
    tilelang_root = Path(tilelang.__file__).parent
    includes = (
        tilelang_root / "src",
        tilelang_root / "3rdparty" / "cutlass" / "include",
    )
    for path in includes:
        if not path.is_dir():
            raise BuildError(f"required TileLang headers missing: {path}")
    compiler_identity = backend.identity(target, includes)
    compiler_identity["pass_config"] = pass_config
    compiler_identity["lowering_sha256"] = hashlib.sha256(
        Path(__file__).with_name("lowering.py").read_bytes()
    ).hexdigest()
    hashed_includes = (
        (*includes, backend.root / "include")
        if compiler_identity["name"] == "nvrtc"
        else includes
    )
    cache_identity = {
        "source_sha256": checked[0]["source_sha256"]
        if checked
        else hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "frontend_sha256": hashlib.sha256(ir.encode()).hexdigest(),
        "headers_sha256": _header_hash(hashed_includes),
        "target": target,
        "compiler": compiler_identity,
        "tilelang_version": version("tilelang"),
        "tvm_ffi_version": version("apache-tvm-ffi"),
        "cuda_sha256": hashlib.sha256(cuda.encode()).hexdigest(),
    }
    key = hashlib.sha256(
        json.dumps(cache_identity, sort_keys=True).encode()
    ).hexdigest()
    root = _cache_root(cache_dir)
    cubin = _cached_image(root, key)
    hit = cubin is not None
    compile_seconds = 0.0
    if cubin is None:
        compile_started = time.perf_counter()
        try:
            cubin = backend.compile(cuda, target, includes)
        except ValueError as exc:
            raise BuildError(str(exc)) from exc
        compile_seconds = time.perf_counter() - compile_started
        try:
            _write_cache(root, key, cubin)
        except OSError:
            pass  # An unwritable cache does not prevent a successful build.
    files = {
        "kernel.cubin": cubin,
        "kernel.tirx.json": ir.encode(),
        **_notices(tilelang_root),
    }
    manifest = {
        "format": "tensor.module",
        "format_version": FORMAT_VERSION,
        "kind": "cubin",
        "target": target,
        "entrypoint": entrypoint,
        "launch": launch,
        "compiler": compiler_identity,
        "provider": "cuda",
        "runtime_abi": runtime_requirement(arguments, symbols),
        "workspace": workspace_requirement(),
        "arguments": arguments,
        "outputs": outputs,
        "symbols": symbols,
        "abi": abi,
        "source_sha256": cache_identity["source_sha256"],
        "tilelang_version": version("tilelang"),
        "tvm_ffi_version": version("apache-tvm-ffi"),
        "op_set": _op_set(ir),
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
    }
    if checked:
        manifest["frontend_artifact_sha256"] = hashlib.sha256(
            source_path.read_bytes()
        ).hexdigest()
    validate_manifest(manifest)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output_path.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                for name, data in sorted(files.items()):
                    archive.writestr(name, data)
                archive.writestr(
                    "manifest.json", json.dumps(manifest, indent=2, sort_keys=True)
                )
    except BaseException:
        if created:
            output_path.unlink(missing_ok=True)
        raise
    return {
        "status": "built",
        "path": str(output_path.resolve()),
        "bytes": output_path.stat().st_size,
        "target": target,
        "entrypoint": entrypoint,
        "arguments": arguments,
        "seconds": time.perf_counter() - started,
        "compile_seconds": compile_seconds,
        "compiler": compiler_identity["name"],
        "nvcc_seconds": compile_seconds if compiler_identity["name"] == "nvcc" else 0.0,
        "cache_hit": hit,
        "cache_key": key,
        "gpu_execution": "unverified",
    }
