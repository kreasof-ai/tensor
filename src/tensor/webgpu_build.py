"""GPU-free producer: original TileLang/TIRx -> portable SIMT -> WGSL bundle."""

from __future__ import annotations

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time
import zipfile

from tensor.abi import runtime_requirement, workspace_requirement
from tensor.artifact import FORMAT, FORMAT_VERSION, read_artifact, validate_manifest
from tensor.build import BuildError, _cache_root, _notices, _op_set
from tensor.webgpu_contract import TARGET, reflect


def build_webgpu(source_path, output_path, *, target=None, cache_dir=None, compiler=None, nvcc=None, nvrtc_home=None):
    started = time.perf_counter()
    source_path, output_path = Path(source_path), Path(output_path)
    if target not in (None, TARGET) or compiler not in (None, "wgsl") or nvcc or nvrtc_home:
        raise BuildError(f"WebGPU uses compiler wgsl and target {TARGET}")
    if output_path.exists() or not source_path.is_file() or source_path.suffix not in (".py", ".tbin"):
        raise BuildError("source needs an existing .py or portable .tbin file and output must not exist")
    from tensor.portable import export_spec, preflight
    checked = preflight(source_path) if source_path.suffix == ".tbin" else None
    import tilelang
    import tvm
    from tensor.lowering import frontend_arguments, integer_expression
    from tensor.webgpu_lowering import lower_simt_gemm, verify_uniform_barriers
    try:
        spec = export_spec(source_path, checked)
        if not isinstance(spec, dict) or "kernel" not in spec or set(spec) - {"kernel", "outputs"}:
            raise BuildError("WebGPU exports need kernel and optional outputs")
        original = spec["kernel"]
        if not isinstance(original, tvm.tirx.PrimFunc):
            raise BuildError("WebGPU export needs a TIRx PrimFunc")
        symbols = {}
        arguments = frontend_arguments(original, symbols)
        if any(arg["dtype"] not in ("float16", "float32", "int32", "uint32")
               or (arg["kind"] == "scalar" and arg["dtype"] == "float16") for arg in arguments):
            raise BuildError("unsupported WebGPU dtype: profile accepts FP16/FP32/int32/uint32 buffers and 32-bit scalars")
        if any(dtype not in ("int32", "uint32") for dtype in symbols.values()):
            raise BuildError("unsupported WebGPU dimension dtype: use int32 or uint32")
        ir = tvm.ir.save_json(tvm.IRModule({str(original.attrs["global_symbol"]): original}))
        identity = {"name": "wgsl", "version": "tensor-simt-v1", "tilelang": version("tilelang"),
                    "lowering_sha256": hashlib.sha256(Path(__file__).with_name("webgpu_lowering.py").read_bytes()).hexdigest(),
                    "producer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        key = hashlib.sha256(json.dumps({"ir": ir, "identity": identity, "outputs": spec.get("outputs", [])}, sort_keys=True).encode()).hexdigest()
        cache = _cache_root(cache_dir) / "webgpu" / f"{key}.tbin"
        hit = False
        if cache.is_file():
            try:
                manifest, files = read_artifact(cache)
                hit = manifest["compiler"] == identity and files["kernel.tirx.json"] == ir.encode()
            except (OSError, ValueError):
                pass
        if not hit:
            kernel = lower_simt_gemm(original)
            target_object = tvm.target.Target("webgpu")
            with tilelang.transform.PassContext(opt_level=3, config={"tirx.disable_vectorize": True}), target_object:
                lowered = tilelang.lower(kernel, target=target_object,
                                         enable_device_compile=False, enable_host_codegen=False)
            functions = list(lowered.device_mod.functions.items())
            if len(functions) != 1:
                raise BuildError("WebGPU inference profile needs a single shader entrypoint")
            name, function = functions[0]
            verify_uniform_barriers(function)
            entrypoint = str(name.name_hint)
            buffers = {str(buffer.data.name): buffer for buffer in original.buffer_map.values()}
            scalars = {str(p.name): str(p.dtype) for p in original.params if p not in original.buffer_map}
            abi = []
            for parameter in function.params:
                name, dtype = str(parameter.name), str(parameter.dtype)
                if dtype == "handle" and name in buffers:
                    buffer = buffers[name]
                    abi.append({"kind": "buffer", "name": str(buffer.name), "dtype": str(buffer.dtype)})
                elif dtype in ("float32", "int32", "uint32") and scalars.get(name, symbols.get(name)) == dtype:
                    abi.append({"kind": "scalar", "name": name, "dtype": dtype})
                else:
                    raise BuildError(f"unsupported WebGPU argument {name}: {dtype}")
            extents = function.attrs["thread_extent"]
            launch = {"grid": [integer_expression(extents.get(f"blockIdx.{axis}", 1), symbols) for axis in "xyz"],
                      "block": [integer_expression(extents.get(f"threadIdx.{axis}", 1), symbols) for axis in "xyz"],
                      "shared_memory_bytes": 0}
            wgsl = str(lowered.kernel_source)
            metadata = reflect(wgsl, entrypoint, abi, launch)
            requirement = runtime_requirement(arguments, symbols)
            requirement["minor"] = 2
            requirement["required_capabilities"].append("opaque_buffer_handles")
            manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "provider": "webgpu", "kind": "wgsl",
                "target": TARGET, "entrypoint": entrypoint, "compiler": identity, "runtime_abi": requirement,
                "workspace": workspace_requirement(), "arguments": arguments, "abi": abi, "symbols": symbols,
                "outputs": spec.get("outputs", []), "launch": launch, "webgpu": metadata,
                "source_sha256": checked[0]["source_sha256"] if checked else hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "tilelang_version": version("tilelang"), "tvm_ffi_version": version("apache-tvm-ffi"), "op_set": _op_set(ir)}
            if checked:
                manifest["frontend_artifact_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            files = {"kernel.wgsl": wgsl.encode(), "kernel.tirx.json": ir.encode(), **_notices(Path(tilelang.__file__).parent)}
            manifest["files"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
        else:
            manifest["source_sha256"] = checked[0]["source_sha256"] if checked else hashlib.sha256(source_path.read_bytes()).hexdigest()
            manifest.pop("frontend_artifact_sha256", None)
            if checked:
                manifest["frontend_artifact_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        validate_manifest(manifest)
    except BuildError:
        raise
    except Exception as exc:
        raise BuildError(f"WebGPU TIRx lowering failed: {type(exc).__name__}: {str(exc)[-3000:]}") from exc
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output_path.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
                for name, content in files.items():
                    bundle.writestr(name, content)
                bundle.writestr("manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
        if not hit:
            from tensor.modules import _atomic
            try:
                _atomic(cache, output_path.read_bytes())
            except OSError:
                pass
    except BaseException:
        if created:
            output_path.unlink(missing_ok=True)
        raise
    return {"status": "built", "provider": "webgpu", "compiler": "wgsl", "target": TARGET,
            "path": str(output_path.resolve()), "bytes": output_path.stat().st_size, "arguments": arguments,
            "seconds": time.perf_counter()-started, "compile_seconds": 0, "nvcc_seconds": 0,
            "cache_hit": hit, "cache_key": key}
