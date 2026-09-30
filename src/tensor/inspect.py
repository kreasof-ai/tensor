"""Source inspection over TileLang's existing lowering and trace hooks."""

from __future__ import annotations

import contextlib
import io
import json
import re
import runpy
import tempfile
from pathlib import Path

from tensor.doctor import TARGET, check_device


def inspect_source(path: Path, *, stage: str, target: str | None = None,
                   trace_dir: Path | None = None, provider="cuda") -> str:
    if not path.is_file() or path.suffix != ".py":
        raise ValueError("inspect source must be an existing Python file")
    if stage == "manifest":
        raise ValueError("source has no artifact manifest; use tirx, target, or passes")
    if provider == "webgpu":
        if target not in (None, "webgpu-portable-v1"):
            raise ValueError("WebGPU inspection needs target webgpu-portable-v1")
        target = "webgpu-portable-v1"
    if provider == "cuda" and target is None and stage in ("target", "passes"):
        device = check_device(0)
        if device["status"] != "ok":
            raise ValueError("no CUDA target detected; pass --target sm_XX")
        target = device["arch"]
    if provider == "cuda" and target is not None and not TARGET.fullmatch(target):
        raise ValueError("target must be an exact CUDA SM, e.g. sm_86")
    namespace = runpy.run_path(str(path.resolve()))
    export = namespace.get("tensor_export")
    if not callable(export):
        raise ValueError("source must define tensor_export()")
    specification = export()
    if not isinstance(specification, dict) or "kernel" not in specification:
        raise ValueError("tensor_export() must return a kernel")
    kernel = specification["kernel"]
    import tilelang
    import tvm

    if not isinstance(kernel, tvm.tirx.PrimFunc):
        raise ValueError("exported kernel is not a TIRx PrimFunc")
    if stage == "tirx":
        symbol = str(kernel.attrs["global_symbol"])
        return str(tvm.IRModule({symbol: kernel}).script())
    def compile_source():
        if provider == "webgpu":
            from tensor.webgpu_lowering import lower_simt_gemm, verify_uniform_barriers
            device_target = tvm.target.Target("webgpu")
            with tilelang.transform.PassContext(opt_level=3, config={"tirx.disable_vectorize": True}), device_target:
                lowered = tilelang.lower(lower_simt_gemm(kernel), target=device_target,
                    enable_device_compile=False, enable_host_codegen=False)
                for function in lowered.device_mod.functions.values():
                    verify_uniform_barriers(function)
                return str(lowered.kernel_source)
        return compile_kernel_source(kernel, {"kind": "cuda", "arch": target})
    from tilelang.tools.compile_only import compile_kernel_source

    if stage == "target":
        return compile_source()
    if stage != "passes":
        raise ValueError("stage must be manifest, tirx, target, or passes")
    import tilelang.tools.lower_trace as trace

    temporary = None
    if trace_dir is None:
        temporary = tempfile.TemporaryDirectory(prefix="tensor-inspect-")
        root = Path(temporary.name)
    else:
        root = Path(trace_dir)
        root.mkdir(parents=True, exist_ok=False)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            trace.enable(mode="terminal", trace_dir=str(root))
            try:
                compile_source()
            finally:
                trace.reset()
        after = sorted(root.rglob("*_after.tir"), key=lambda file: file.name)
        passes = []
        for file in after:
            match = re.match(r"(\d+)_(.+)_after\.tir\Z", file.name)
            if match:
                passes.append({"order": int(match[1]), "name": match[2]})
        report = {"target": target, "passes": passes, "count": len(passes),
                  "trace_dir": str(root.resolve()) if temporary is None else None}
        return json.dumps(report, indent=2)
    finally:
        if temporary is not None:
            temporary.cleanup()
