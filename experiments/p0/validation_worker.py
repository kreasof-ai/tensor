"""Isolated workers for the remaining Phase 0 architecture experiments."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from experiments.p0.provenance import snapshot


def runtime():
    from experiments.p0.cuda_driver import Driver
    _, info = Driver().device_info()
    import tilelang
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot allocate CUDA buffers")
    return tilelang, torch, info


def check_elementwise(kernel, torch, size, device="cuda"):
    a = torch.arange(size, device=device, dtype=torch.float32) / 8 - 12
    b = torch.ones_like(a) / 4
    out = torch.full_like(a, float("nan"))
    kernel(a, b, out)
    torch.testing.assert_close(out, torch.relu(2*a+b), rtol=1e-6, atol=1e-6)
    return float((out - torch.relu(2*a+b)).abs().max())


def cache_probe(mode, root):
    tilelang, torch, device = runtime()
    from experiments.p0.kernels import artifact_elementwise
    from experiments.p0.upstream_probe import (cache_identity, cache_key_matrix,
                                               materialize, observe_compile)
    func = artifact_elementwise(129)
    target = {"kind": "cuda", "arch": device["arch"]}
    forbid = mode == "cache_disk"
    started = time.perf_counter()
    with observe_compile(forbid=forbid) as stages:
        kernel = tilelang.compile(func, out_idx=None, target=target, execution_backend="tvm_ffi")
        materialize(kernel)
    result = {"compile_seconds": time.perf_counter()-started, "stages": stages,
              "identity": cache_identity(kernel), "max_abs_error": check_elementwise(kernel, torch, 129)}
    if mode == "cache_cold":
        assert stages["lower_calls"] == 1
        started = time.perf_counter()
        with observe_compile(forbid=True) as warm_stages:
            warm = tilelang.compile(func, out_idx=None, target=target, execution_backend="tvm_ffi")
            materialize(warm)
        assert warm is kernel
        result["memory_hit"] = {"seconds": time.perf_counter()-started, "same_object": True,
                                "stages": warm_stages}
        matrix = cache_key_matrix(func, target)
        assert all(key != matrix["baseline"] for key in matrix["variants"].values())
        result["key_matrix"] = matrix
        with observe_compile() as changed_stages:
            changed = tilelang.compile(artifact_elementwise(130), out_idx=None,
                                       target=target, execution_backend="tvm_ffi")
            materialize(changed)
        assert changed_stages["lower_calls"] == 1
        assert cache_identity(changed)["key"] != result["identity"]["key"]
        result["shape_invalidation"] = {"stages": changed_stages,
                                        "max_abs_error": check_elementwise(changed, torch, 130)}
    elif mode == "cache_corrupt":
        assert stages["lower_calls"] == 1
    else:
        assert stages["lower_calls"] == stages["cuda_compile_calls"] == 0
    return result


def provider_probe(root):
    import tilelang
    import torch
    from experiments.p0.kernels import artifact_elementwise
    from experiments.p0.upstream_probe import cpu_library, cpu_registration_probe
    registration = cpu_registration_probe()
    try:
        tilelang.compile(artifact_elementwise(128), out_idx=None,
                         target="c", execution_backend="cython")
    except Exception as error:
        vectorization_error = f"{type(error).__name__}: {error}"
    else:
        vectorization_error = None
    rows = []
    for size in (1, 127, 128, 129, 1025):
        started = time.perf_counter()
        kernel = tilelang.compile(artifact_elementwise(size), out_idx=None,
                                  target="c", execution_backend="cython",
                                  pass_configs={"tirx.disable_vectorize": True})
        error = check_elementwise(kernel, torch, size, device="cpu")
        rows.append({"size": size, "seconds": time.perf_counter()-started, "max_abs_error": error})
        if size == 129:
            shutil.copy2(cpu_library(kernel), root / "libp0_cpu.so")
    try:
        tilelang.compile(artifact_elementwise(129), out_idx=None, target="c", execution_backend="tvm_ffi")
    except ValueError as error:
        unsupported = str(error)
        assert "does not support lowering with compilation" in unsupported
    else:
        raise AssertionError("pinned CPU TVM FFI behavior changed")
    return {"rows": rows, "registration": registration, "cpu_tvm_ffi_error": unsupported,
            "default_vectorization_error": vectorization_error,
            "pass_configs": {"tirx.disable_vectorize": True},
            "llvm_enabled": bool(tilelang.tvm.runtime.enabled("llvm")),
            "execution_backend": "cython", "native_library": str(root / "libp0_cpu.so")}


def symbolic_probe(root):
    tilelang, torch, device = runtime()
    from experiments.p0.validation_kernels import dynamic_elementwise, dynamic_gemm
    from experiments.p0.upstream_probe import materialize, observe_compile
    # Reload the artifact before compiling; one compiled object serves all extents.
    ir = tilelang.tvm.ir.save_json(tilelang.tvm.IRModule({"main": dynamic_elementwise()}))
    (root / "dynamic.json").write_text(ir)
    func = next(iter(tilelang.tvm.ir.load_json(ir).functions.values()))
    target = {"kind": "cuda", "arch": device["arch"]}
    with observe_compile() as stages:
        kernel = tilelang.compile(func, out_idx=None, target=target, execution_backend="tvm_ffi")
        materialize(kernel)
        rows = [{"size": size, "max_abs_error": check_elementwise(kernel, torch, size)}
                for size in (1, 127, 128, 129, 1025)]
    assert stages["lower_calls"] == 1
    with observe_compile() as gemm_stages:
        gemm = tilelang.compile(dynamic_gemm(), out_idx=None, target=target, execution_backend="tvm_ffi")
        materialize(gemm)
        gemm_rows = []
        for size in (1, 31, 32, 33, 65):
            torch.manual_seed(size)
            a = torch.randn((size, 32), dtype=torch.float16, device="cuda")
            b = torch.randn((32, 32), dtype=torch.float16, device="cuda")
            out = torch.full((size, 32), float("nan"), dtype=torch.float16, device="cuda")
            gemm(a, b, out)
            torch.testing.assert_close(out, a @ b, rtol=1e-2, atol=1e-2)
            gemm_rows.append({"rows": size, "max_abs_error": float((out-a@b).abs().max())})
    assert gemm_stages["lower_calls"] == 1
    try:
        tilelang.compile(dynamic_gemm(dynamic_tile=True), out_idx=None,
                         target=target, execution_backend="tvm_ffi")
    except Exception as error:
        dynamic_tile_error = f"{type(error).__name__}: {error}"
    else:
        raise AssertionError("dynamic GEMM tile unexpectedly compiled; execute it before accepting")
    return {"elementwise": rows, "elementwise_compile": stages,
            "static_tile_dynamic_gemm": gemm_rows, "gemm_compile": gemm_stages,
            "dynamic_tile_error": dynamic_tile_error}


def framework_probe(root):
    import operator
    import tilelang
    import torch
    from experiments.p0.kernels import artifact_elementwise
    from torch._dynamo.backends.common import aot_autograd
    from functorch.compile import make_boxed_func
    graphs = {"fx": [], "aot_forward": [], "aot_backward": [], "breaks": []}
    lowered = []

    def describe(gm, inputs):
        return {"code": gm.code, "operators": [str(n.target) for n in gm.graph.nodes if n.op.startswith("call")],
                "input_types": [type(x).__name__ for x in inputs]}

    def frontend(gm, inputs):
        graphs["fx"].append(describe(gm, inputs))
        nodes = [n for n in gm.graph.nodes if n.op not in ("placeholder", "output")]
        placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
        output = next(n for n in gm.graph.nodes if n.op == "output")
        supported = (len(placeholders) == 2 and len(nodes) == 3
                     and all(n.op == "call_function" for n in nodes)
                     and [n.target for n in nodes] == [operator.mul, operator.add, torch.relu]
                     and nodes[0].args == (placeholders[0], 2)
                     and nodes[1].args == (nodes[0], placeholders[1])
                     and nodes[2].args == (nodes[1],) and output.args == ((nodes[2],),)
                     and all(isinstance(x, torch.Tensor) and x.dtype == torch.float32
                             and x.device.type == "cpu" and x.ndim == 1 and x.is_contiguous() for x in inputs)
                     and inputs[0].shape == inputs[1].shape)
        if not supported:
            return gm.forward
        kernel = tilelang.compile(artifact_elementwise(inputs[0].numel()), out_idx=None,
                                  target="c", execution_backend="cython",
                                  pass_configs={"tirx.disable_vectorize": True})
        lowered.append(inputs[0].numel())
        def run(a, b):
            out = torch.empty_like(a)
            kernel(a, b, out)
            return (out,)
        return run

    def operation(a, b):
        return torch.relu(a * 2 + b)

    torch._dynamo.reset()
    compiled = torch.compile(operation, backend=frontend, fullgraph=True, dynamic=False)
    for size in (129, 129, 130):
        a, b = torch.randn(size), torch.randn(size)
        torch.testing.assert_close(compiled(a,b), operation(a,b))
    assert lowered == [129, 130], lowered

    def capture(kind):
        def compiler(gm, inputs):
            graphs[kind].append(describe(gm, inputs))
            return make_boxed_func(gm.forward)
        return compiler

    torch._dynamo.reset()
    aot = torch.compile(operation, backend=aot_autograd(fw_compiler=capture("aot_forward"),
                        bw_compiler=capture("aot_backward")), fullgraph=True, dynamic=True)
    for size in (129, 257):
        a = torch.randn(size, requires_grad=True)
        b = torch.randn(size, requires_grad=True)
        result = aot(a,b)
        expected = operation(a,b)
        torch.testing.assert_close(result, expected)
        result.sum().backward()
        torch.testing.assert_close(a.grad, 2*(2*a+b > 0).float())
        torch.testing.assert_close(b.grad, (2*a+b > 0).float())
    assert len(graphs["aot_forward"]) == len(graphs["aot_backward"]) == 1

    def broken(a,b):
        out = operation(a,b)
        torch._dynamo.graph_break()
        return out + 1
    def observe(gm, inputs):
        graphs["breaks"].append(describe(gm, inputs))
        return gm.forward
    torch._dynamo.reset()
    with_breaks = torch.compile(broken, backend=observe)
    a,b = torch.randn(129), torch.randn(129)
    torch.testing.assert_close(with_breaks(a,b), broken(a,b))
    assert len(graphs["breaks"]) == 2
    return {"graphs": graphs, "tilelang_lowered_extents": lowered,
            "aot_execution": "boxed FX eager reference; not TileLang training",
            "graph_break_fragments": len(graphs["breaks"])}


def abi_probe(root):
    import tilelang
    import tvm_ffi
    ffi = Path(tvm_ffi.libinfo.find_libtvm_ffi()).parent
    include = tvm_ffi.libinfo.include_paths()[0]
    provider = (root.parent / "provider").resolve()
    shutil.copy2(provider / "libp0_cpu.so", root / "libp0_cpu.so")
    root = root.resolve()
    source = Path(__file__).parent.resolve()
    common = ["g++", "-std=c++17", "-O2", "-I"+include]
    commands = [common + ["-fPIC", "-shared", str(source / "native_module.cpp"), "-L"+str(root),
                "-lp0_cpu", "-L"+str(ffi), "-ltvm_ffi", "-Wl,-rpath,$ORIGIN",
                "-Wl,-rpath,"+str(ffi), "-o", str(root / "libp0_ffi.so")],
                common + [str(source / "native_host.cpp"), "-L"+str(ffi), "-ltvm_ffi", "-ldl",
                "-Wl,-rpath,"+str(ffi), "-o", str(root / "native_host")]]
    for command in commands:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    compiler = Path(tilelang.__file__).parent / "lib"
    invocations = {"cpu_run": ["run", str(root / "libp0_ffi.so")],
                   "native_ir": ["ir", str(compiler / "libtvm_compiler.so"),
                                 str(compiler / "libtilelang.so"),
                                 str(root.parent / "symbolic" / "dynamic.json")]}
    results = {}
    for name, args in invocations.items():
        proc = subprocess.run([str(root / "native_host"), *args], check=True,
                              capture_output=True, text=True, timeout=60)
        results[name] = json.loads(proc.stdout)
    linked = subprocess.run(["ldd", str(root / "native_host")], check=True,
                            capture_output=True, text=True).stdout
    assert "libpython" not in linked and "libtorch" not in linked and "libc10" not in linked
    return {**results, "commands": commands, "host_dependencies": linked,
            "rust": "unverified: Rust toolchain not installed",
            "native_compilation_pipeline": "unverified; IR loading is not end-to-end native compilation",
            "abi_inventory": {
                "function_call": "TVM FFI Function + typed DLL exports; exercised",
                "buffers": "borrowed DLTensor/TensorView; caller owns storage; exercised on CPU",
                "errors": "wrong dtype rejected and propagated to C++; exercised",
                "gpu_load_launch_copy": "CUDA Driver API; exercised by opaque artifact consumer",
                "streams": "driver consumer owns its stream; foreign-stream interop unverified",
                "events": "PyTorch CUDA events measured; provider-neutral event ABI unverified",
                "async_signals_memory_ordering": "no stable Tensor contract yet; unverified"}}


def fusion_probe(root):
    import tilelang
    from experiments.p0.kernels import gemm_relu
    from tilelang.tools.compile_only import compile_kernel_source
    captured = []
    from experiments.p0.upstream_probe import capture_tile_op
    target = {"kind": "cuda", "arch": "sm_86"}
    with capture_tile_op(captured):
        original = compile_kernel_source(gemm_relu(64,64,32), target)
    assert len(captured) == 1, len(captured)
    (root / "post-lower-tile-op.json").write_text(captured[0])
    loaded = tilelang.tvm.ir.load_json(captured[0])
    try:
        reloaded = compile_kernel_source(loaded, target)
    except Exception as error:
        rerun = {"status": "rejected", "error": f"{type(error).__name__}: {error}"}
    else:
        rerun = {"status": "emitted", "same_source": original == reloaded}
    public_names = {name: [n for n in dir(module) if "fus" in n.lower()]
                    for name, module in (("tilelang",tilelang),("transform",tilelang.transform))}
    return {"post_lower_tile_op_bytes": len(captured[0]), "re_lowering": rerun,
            "public_names_containing_fus": public_names,
            "fusion": "unverified: this probe does not compose or reschedule two artifacts",
            "decision": "keep frontend IR as a pinned source tier; ship opaque executables first; defer fusion"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["cache_cold", "cache_disk", "cache_corrupt", "provider", "symbolic", "framework", "abi", "fusion", "compile", "perf", "cpu_provider", "composition", "rust_host", "foreign_stream", "static_symbolic"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--arch", default="sm_86")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        if args.mode.startswith("cache_"):
            result = cache_probe(args.mode, args.root)
        elif args.mode == "compile":
            from experiments.p0.measure import compile_matrix
            result = compile_matrix(args.root, args.arch)
        elif args.mode == "perf":
            from experiments.p0.measure import performance
            result = performance(args.root)
        elif args.mode == "cpu_provider":
            from experiments.p0.cpu_provider import probe
            result = probe(args.root)
        elif args.mode == "composition":
            from experiments.p0.composition import probe
            result = probe(args.root)
        elif args.mode == "rust_host":
            from experiments.p0.rust_host import probe
            result = probe(args.root)
        elif args.mode == "foreign_stream":
            from experiments.p0.foreign_stream import probe
            result = probe(args.root)
        elif args.mode == "static_symbolic":
            from experiments.p0.static_symbolic import probe
            result = probe(args.root)
        else:
            result = {"provider": provider_probe, "symbolic": symbolic_probe,
                      "framework": framework_probe, "abi": abi_probe,
                      "fusion": fusion_probe}[args.mode](args.root)
        report = {"status": "passed", "result": result}
        code = 0
    except Exception as error:
        from experiments.p0.cuda_driver import CudaUnavailable
        import traceback
        report = {"status": "skipped" if isinstance(error, CudaUnavailable) else "failed",
                  "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}
        code = 2 if report["status"] == "skipped" else 1
    report.update(mode=args.mode, seconds=time.perf_counter()-started, provenance=snapshot())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "mode": args.mode, "report": str(args.report)}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
