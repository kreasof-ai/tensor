"""Matched A10G static versus runtime-symbolic throughput measurements."""

import statistics


def timing(torch, call):
    for _ in range(20):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(100):
            call()
    samples = []
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 10)  # 1000 us / 100 captured calls
    return {"microseconds": statistics.median(samples), "samples_microseconds": samples}


def probe(root):
    import tilelang
    import torch
    from experiments.p0.cuda_driver import Driver
    from experiments.p0.kernels import artifact_elementwise
    from experiments.p0.validation_kernels import dynamic_elementwise, dynamic_gemm
    from experiments.p0.upstream_probe import materialize, observe_compile

    _, device = Driver().device_info()
    assert device["arch"] == "sm_86"
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    target = {"kind": "cuda", "arch": device["arch"]}

    def compile_func(func):
        kernel = tilelang.compile(func, out_idx=None, target=target, execution_backend="tvm_ffi")
        materialize(kernel)
        return kernel

    def pair(name, sizes, factory, dtype, tensors, reference, tolerances):
        rows = []
        with observe_compile() as dynamic_stages:
            dynamic = compile_func(factory(None))
        assert dynamic_stages["lower_calls"] == 1
        for size in sizes:
            with observe_compile() as static_stages:
                static = compile_func(factory(size))
            assert static_stages["lower_calls"] == 1
            inputs = tensors(size, dtype)
            expected = reference(*inputs)
            out_static = torch.full_like(expected, float("nan"))
            out_dynamic = torch.full_like(expected, float("nan"))
            static(*inputs, out_static)
            dynamic(*inputs, out_dynamic)
            torch.testing.assert_close(out_static, expected, **tolerances)
            torch.testing.assert_close(out_dynamic, expected, **tolerances)
            torch.testing.assert_close(out_static, out_dynamic, **tolerances)
            static_time = timing(torch, lambda: static(*inputs, out_static))
            dynamic_time = timing(torch, lambda: dynamic(*inputs, out_dynamic))
            rows.append({"size": size, "static": static_time, "symbolic": dynamic_time,
                         "symbolic_over_static": dynamic_time["microseconds"] / static_time["microseconds"],
                         "static_compile": static_stages,
                         "static_max_abs_error": float((out_static-expected).abs().max()),
                         "symbolic_max_abs_error": float((out_dynamic-expected).abs().max())})
        return {"name": name, "dynamic_compile": dynamic_stages, "rows": rows}

    elementwise = pair("elementwise", (1,127,128,129,1025,1048576),
        lambda size: dynamic_elementwise() if size is None else artifact_elementwise(size),
        torch.float32,
        lambda size, dtype: (torch.randn(size, device="cuda", dtype=dtype),
                             torch.randn(size, device="cuda", dtype=dtype)),
        lambda a,b: torch.relu(2*a+b), {"rtol":1e-6,"atol":1e-6})
    gemm = pair("gemm_static_tile32", (1,31,32,33,65,1024),
        lambda size: dynamic_gemm(static_rows=size), torch.float16,
        lambda size, dtype: (torch.randn((size,32),device="cuda",dtype=dtype),
                             torch.randn((32,32),device="cuda",dtype=dtype)),
        lambda a,b: a@b, {"rtol":1e-2,"atol":1e-2})
    return {"device":device,"measurements":[elementwise,gemm],
            "warmup":20,"captured_iterations":100,"repetitions":5,
            "method":"CUDA events around graph replay; median microseconds per operation",
            "comparison":"same workload, dtype, inputs, output reuse and static tile; one dynamic binary per workload",
            "limit":"single A10G; no cross-GPU throughput claim"}
