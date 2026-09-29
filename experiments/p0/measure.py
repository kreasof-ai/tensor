"""Full compile/cache timings and GPU baselines for the fixed P0 workloads."""

import statistics
import hashlib
import time


def compile_matrix(root, arch):
    import tilelang
    from experiments.p0.kernels import KERNELS
    from experiments.p0.upstream_probe import materialize, observe_compile
    rows = []
    for name, factory in KERNELS.items():
        started = time.perf_counter()
        func = factory()
        frontend = time.perf_counter() - started
        ir_before = hashlib.sha256(func.script(show_meta=True).encode()).hexdigest()
        started = time.perf_counter()
        try:
            with observe_compile() as stages:
                kernel = tilelang.compile(func, out_idx=None, target={"kind": "cuda", "arch": arch},
                                          execution_backend="tvm_ffi")
                materialize(kernel)
        except Exception as error:
            diagnostic = str(error)
            (root / f"{name}-compile-error.txt").write_text(diagnostic)
            rows.append({"name": name, "arch": arch, "status": "compiler_rejected",
                         "frontend_seconds": frontend, "full_compile_seconds": time.perf_counter()-started,
                         "stages": stages, "error": diagnostic[-6000:]})
            continue
        elapsed = time.perf_counter() - started
        ir_after = hashlib.sha256(func.script(show_meta=True).encode()).hexdigest()
        assert stages["lower_calls"] == stages["cuda_compile_calls"] == 1
        started = time.perf_counter()
        with observe_compile() as warm_stages:
            warm = tilelang.compile(func, out_idx=None, target={"kind": "cuda", "arch": arch},
                                    execution_backend="tvm_ffi")
            materialize(warm)
        warm_seconds = time.perf_counter() - started
        started = time.perf_counter()
        with observe_compile() as rebuilt_stages:
            rebuilt = tilelang.compile(factory(), out_idx=None, target={"kind": "cuda", "arch": arch},
                                       execution_backend="tvm_ffi")
            materialize(rebuilt)
        rebuilt_seconds = time.perf_counter() - started
        rows.append({"name": name, "arch": arch, "status": "compiled", "frontend_seconds": frontend,
                     "full_compile_seconds": elapsed, "stages": stages,
                     "warm_seconds": warm_seconds, "warm_stages": warm_stages,
                     "warm_same_object": warm is kernel, "ir_sha256_before": ir_before,
                     "ir_sha256_after": ir_after, "fresh_frontend_warm_seconds": rebuilt_seconds,
                     "fresh_frontend_warm_stages": rebuilt_stages, "fresh_frontend_same_object": rebuilt is kernel})
    return {"rows": rows, "timing_boundary": "imports excluded; lazy host FFI JIT included",
            "stage_relationship": "CUDA compilation is nested within lowering; do not add stage times",
            "gpu_execution": "not performed by this compilation probe"}


def performance(root):
    import tilelang
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from experiments.p0.cuda_driver import Driver
    from experiments.p0 import kernels as K
    _, device = Driver().device_info()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    rows = []
    target = {"kind": "cuda", "arch": device["arch"]}

    def random(shape, dtype=torch.float16):
        return torch.randn(shape, dtype=dtype, device="cuda")

    def timing(call):
        for _ in range(20):
            call()
        torch.cuda.synchronize()
        samples = []
        # CUDA graph replay removes Python submission starvation from tiny kernels.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(100):
                call()
        for _ in range(5):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / 100)
        return {"microseconds": statistics.median(samples), "samples_microseconds": samples}

    def check(name, func, inputs, reference, baseline, label, rtol=1e-2, atol=1e-2):
        kernel = tilelang.compile(func, out_idx=None, target=target, execution_backend="tvm_ffi")
        expected = reference()
        out = torch.full_like(expected, float("nan"))
        kernel(*inputs, out)
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
        torch.testing.assert_close(baseline(), expected, rtol=rtol, atol=atol)
        candidate = timing(lambda: kernel(*inputs, out))
        base = timing(baseline)
        rows.append({"name": name, "shape": list(out.shape), "dtype": str(out.dtype),
                     "max_abs_error": float((out-expected).abs().max()), "rtol": rtol, "atol": atol,
                     "candidate": candidate, "baseline": base, "baseline_implementation": label,
                     "speedup": base["microseconds"] / candidate["microseconds"]})

    # Float32 intermediates preserve the candidate's single-rounding semantics.
    a, b = random((1024, 1024)), random((1024, 1024))
    def elementwise(a, b):
        return torch.relu(2*a.float()+b.float()).to(a.dtype)
    fused = torch.compile(elementwise, backend="inductor", fullgraph=True)
    fused(a,b)  # compilation belongs outside the throughput measurement
    check("fused_elementwise", K.KERNELS["fused_elementwise"](), [a,b],
          lambda: elementwise(a,b), lambda: fused(a,b), "TorchInductor fused pointwise")
    a, b, bias = random((1024,1024)), random((1024,1024)), random((1024,))
    def gemm(a,b,bias):
        return torch.relu(a@b+bias)
    fused_gemm = torch.compile(gemm, backend="inductor", fullgraph=True)
    fused_gemm(a,b,bias)
    # Both use fp32 accumulation; reference avoids premature fp16 mm rounding.
    check("gemm_relu", K.KERNELS["gemm_relu"](), [a,b,bias],
          lambda: torch.relu(a.float()@b.float()+bias.float()).half(),
          lambda: fused_gemm(a,b,bias), "TorchInductor GEMM+epilogue (vendor mm selection)",
          rtol=2e-2, atol=5e-2)
    x = random((256,1024), torch.float32)
    check("row_sum", K.KERNELS["row_sum"](), [x], lambda: x.sum(1), lambda: x.sum(1),
          "PyTorch CUDA sum", rtol=1e-4, atol=1e-4)
    index, x = torch.randint(0,1024,(1024,),device="cuda",dtype=torch.int32), random((1024,128))
    check("gather_rows", K.KERNELS["gather_rows"](), [index,x],
          lambda: torch.index_select(x,0,index), lambda: torch.index_select(x,0,index),
          "PyTorch CUDA index_select; all indices valid", rtol=0, atol=0)
    q,k,v = [random((8,128,64)) for _ in range(3)]
    def attention():
        # Force the tuned FlashAttention backend and require it to work.
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0)).squeeze(0)
    check("flash_attention", K.KERNELS["flash_attention"](), [q,k,v],
          lambda: (torch.softmax(q.float()@k.float().transpose(-1,-2)/8,dim=-1)@v.float()).half(),
          attention, "PyTorch SDPA forced FLASH_ATTENTION", rtol=1e-2, atol=1e-2)
    return {"device": device, "rows": rows, "warmup": 20, "captured_iterations": 100,
            "repetitions": 5, "method": "CUDA events around CUDA graph replay; median per operation",
            "allocation_policy": "candidate output reused; baseline tensor lifetimes captured in graph",
            "other_gpu_architectures": "unverified: only sm_86 hardware is available"}
