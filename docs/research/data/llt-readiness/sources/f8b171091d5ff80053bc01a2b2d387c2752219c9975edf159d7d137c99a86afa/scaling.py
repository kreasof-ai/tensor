"""Kernel/full-model resource qualification on L40S, with all samples retained."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
import torch
from tensor_torch.llt import Operators, KVCache, AdamW
from .model import Config, Model
from .qualify import ROOT, OUT, setup, save, TOLERANCES, batch, train_step


def clear():
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()


def measure(fn, samples=9, graph=True):
    for _ in range(3):
        fn()
    clear()
    torch.cuda.reset_peak_memory_stats()
    result = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    del result
    gpu = []
    wall = []
    for _ in range(samples):
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start = time.perf_counter()
        a.record()
        fn()
        b.record()
        b.synchronize()
        gpu.append(a.elapsed_time(b))
        wall.append((time.perf_counter() - start) * 1000)
    report = {
        "eager_peak_allocated_bytes": peak,
        "gpu_samples_ms": gpu,
        "wall_samples_ms": wall,
        "gpu_median_ms": statistics.median(gpu),
        "wall_median_ms": statistics.median(wall),
    }
    if graph:
        clear()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        clear()
        g = torch.cuda.CUDAGraph()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        with torch.cuda.graph(g, stream=side):
            result = fn()
        g.replay()
        torch.cuda.synchronize()
        values = []
        for _ in range(samples):
            a, b = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            a.record()
            for _ in range(10):
                g.replay()
            b.record()
            b.synchronize()
            values.append(a.elapsed_time(b) / 10)
        report.update(
            graph_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            graph_baseline_allocated_bytes=baseline,
            graph_samples_ms=values,
            graph_median_ms=statistics.median(values),
        )
        del result, g
        clear()
    return report


def attention(ops):
    rows = []
    configs = [
        (1, 4, 129, d, dtype)
        for dtype in (torch.float16, torch.bfloat16)
        for d in (32, 64, 96, 128)
    ]
    configs += [
        (b, 4, n, d, torch.bfloat16)
        for b, n, d in ((4, 257, 64), (8, 129, 32), (1, 513, 64), (1, 1025, 128))
    ]
    for b, h, n, d, dtype in configs:
        setup()
        q = torch.randn(b, h, n, d, device="cuda", dtype=dtype, requires_grad=True)
        c = torch.randn(b, 1, n, d, device="cuda", dtype=dtype, requires_grad=True)
        dy = torch.randn_like(q)

        def tensor():
            out = ops.attention(q, c, c, causal=True, scale=0.125)
            return out, torch.autograd.grad(out, (q, c), dy)

        def reference():
            out = torch.nn.functional.scaled_dot_product_attention(
                q, c, c, is_causal=True, enable_gqa=True, scale=0.125
            )
            return out, torch.autograd.grad(out, (q, c), dy)

        saved = []

        def pack(x):
            saved.append(
                {
                    "shape": list(x.shape),
                    "bytes": x.numel() * x.element_size(),
                    "pointer": x.data_ptr(),
                }
            )
            return x

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda x: x):
            a, ag = tensor()
        unique = {x["pointer"]: x for x in saved}
        saved_bytes = sum(x["bytes"] for x in unique.values())
        for x in saved:
            x.pop("pointer")
        r, rg = reference()
        tol = TOLERANCES[
            "attention_bf16" if dtype == torch.bfloat16 else "attention_fp16"
        ]
        torch.testing.assert_close(a, r, **tol)
        errors = []
        for x, y in zip(ag, rg):
            # Shared dC combines both paths and all heads; absolute error scales
            # with that sum. The predeclared BF16 gradient tolerance applies.
            gt = TOLERANCES["gradient_bf16"] if dtype == torch.bfloat16 else tol
            torch.testing.assert_close(x, y, **gt)
            errors.append((x - y).abs().max().item())
        del a, ag, r, rg
        rows.append(
            {
                "batch": b,
                "heads": h,
                "sequence": n,
                "rank": d,
                "dtype": str(dtype),
                "gradient_max_errors": errors,
                "saved_tensors": saved,
                "saved_unique_bytes": saved_bytes,
                "split_head_backward": n >= 256,
                "split_head_workspace_bytes": 2 * b * h * n * d * 4 if n >= 256 else 0,
                "tensor": measure(tensor),
                "torch_sdpa": measure(reference),
            }
        )
        print("Attention qualified:", b, h, n, d, dtype, flush=True)
        del q, c, dy
        clear()
    finish("attention", {"status": "passed", "cases": rows, "coverage": ops.report})


def decode(ops):
    rows = []
    configs = [
        (1, 8, n, d, dtype)
        for dtype in (torch.float16, torch.bfloat16)
        for n in (4097, 65537)
        for d in (32, 64, 96, 128)
    ]
    configs += [(4, 8, 1025, 64, torch.bfloat16), (8, 8, 129, 32, torch.bfloat16)]
    with torch.inference_mode():
        for b, h, n, d, dtype in configs:
            setup()
            c = torch.randn(b, 1, n, d, device="cuda", dtype=dtype)
            q = torch.randn(b, h, 1, d, device="cuda", dtype=dtype)
            cache = KVCache(ops, b, 1, n, d, dtype=dtype, shared=True)
            cache.append(c)

            def tensor():
                return ops.decode(q, cache, scale=0.125)

            def reference():
                return torch.nn.functional.scaled_dot_product_attention(
                    q, c, c, enable_gqa=True, scale=0.125
                )

            a = tensor()
            r = reference()
            tol = TOLERANCES[
                "attention_bf16" if dtype == torch.bfloat16 else "attention_fp16"
            ]
            torch.testing.assert_close(a, r, **tol)
            error = (a - r).abs().max().item()
            del a, r
            rows.append(
                {
                    "batch": b,
                    "heads": h,
                    "history": n,
                    "rank": d,
                    "dtype": str(dtype),
                    "maximum_error": error,
                    "cache_bytes": cache.nbytes,
                    "tensor": measure(tensor),
                    "torch_sdpa": measure(reference),
                }
            )
            print("Decode qualified:", b, h, n, d, dtype, flush=True)
            del c, q, cache
            clear()
    finish("decode", {"status": "passed", "cases": rows, "coverage": ops.report})


def generation(ops):
    rows = []
    for architecture in ("llt", "naive"):
        for rotary in (0, 16):
            setup()
            c = Config(architecture=architecture, rotary=rotary, loops=3)
            model = Model(c, ops).cuda()
            with torch.inference_mode():
                tokens = torch.arange(21, device="cuda").reshape(1, -1)
                _, state = model.prefill(tokens[:, :17])
                errors = []
                for length in range(18, 22):
                    actual = model.decode_token(tokens[:, length - 1 : length], state)
                    full = model(tokens[:, :length])[:, -1:]
                    error = (actual - full).abs().max().item()
                    errors.append(error)
                    torch.testing.assert_close(actual, full, atol=0.025, rtol=0.025)

                def trajectory():
                    state["length"] = 17
                    for cache in state["caches"]:
                        cache.length = 17
                        cache.lengths.fill_(17)
                        cache.overflow.zero_()
                        cache.captured_mutation = False
                    for length in range(18, 22):
                        result = model.decode_token(
                            tokens[:, length - 1 : length], state
                        )
                    return result

                latency = measure(trajectory, graph=False)
                rows.append(
                    {
                        "complete_four_token_decode": latency,
                        "config": asdict(c),
                        "prefix": 17,
                        "generated_tokens": 4,
                        "maximum_errors": errors,
                        "cache_count": len(state["caches"]),
                        "cache_bytes": sum(x.nbytes for x in state["caches"]),
                    }
                )
                print(
                    "Generation qualified:", architecture, "rotary", rotary, flush=True
                )
                del model, state
                clear()
    finish("generation", {"status": "passed", "cases": rows, "coverage": ops.report})


def systems(ops):
    rows = []
    configs = [(64, 33, 256, t, r) for t in (1, 2, 10, 20) for r in (32, 64)]
    configs += [(512, 257, 4096, 10, 64), (512, 257, 50257, 2, 64)]
    for w, s, v, t, r in configs:
        for architecture in ("llt", "naive"):
            for backend in ("tensor", "torch"):
                setup()
                c = Config(
                    width=w,
                    heads=w // 32,
                    rank=r,
                    loops=t,
                    vocab=v,
                    capacity=max(s, 128),
                    architecture=architecture,
                    checkpoint=True,
                )
                model = Model(c, ops if backend == "tensor" else None).cuda()
                gen = torch.Generator().manual_seed(51)
                tokens, target = batch(gen, c, s=s)
                opt = (
                    AdamW(model.parameters(), ops, lr=1e-3)
                    if backend == "tensor"
                    else torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
                )

                def training():
                    return train_step(model, opt, tokens, target, backend == "torch")

                trained = measure(training, graph=False)
                opt.zero_grad(set_to_none=True)
                del opt
                clear()

                def prefill():
                    with (
                        torch.inference_mode(),
                        torch.autocast("cuda", dtype=torch.bfloat16),
                    ):
                        return model(tokens)

                inferred = measure(prefill)
                rows.append(
                    {
                        "config": asdict(c),
                        "backend": backend,
                        "training": trained,
                        "prefill": inferred,
                    }
                )
                print(
                    "Systems measured:",
                    w,
                    s,
                    v,
                    t,
                    r,
                    architecture,
                    backend,
                    flush=True,
                )
                del model
                clear()
                torch.cuda.empty_cache()
    finish("systems", {"status": "passed", "cases": rows, "coverage": ops.report})


def loss_memory(ops):
    rows = []
    for chunk in (0, 32):
        setup()
        x = torch.randn(129, 512, device="cuda", requires_grad=True)
        w = torch.randn(50257, 512, device="cuda", requires_grad=True) * 0.02
        w = w.detach().requires_grad_()
        target = torch.arange(129, device="cuda")

        def step():
            value = (
                ops.linear_cross_entropy(x, w, target, chunk_size=chunk)
                if chunk
                else ops.cross_entropy(ops.linear(x, w), target)
            )
            grads = torch.autograd.grad(value, (x, w))
            return value, grads

        result = measure(step, graph=False)
        rows.append(
            {
                "rows": 129,
                "width": 512,
                "vocabulary": 50257,
                "chunk_size": chunk,
                **result,
            }
        )
        print("Classifier memory measured:", chunk, flush=True)
        del x, w, target
        clear()
    finish("loss-memory", {"status": "passed", "cases": rows, "coverage": ops.report})


def finish(name, result):
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["scaling_source_sha256"] = sha
    path = OUT / "sources" / sha / Path(__file__).name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(Path(__file__).read_bytes())
    save(name, result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase", choices=("attention", "decode", "generation", "systems", "loss-memory")
    )
    args = parser.parse_args()
    setup()
    ops = Operators(ROOT / "build/llt-qualification/artifacts")
    globals()[args.phase.replace("-", "_")](ops)


if __name__ == "__main__":
    main()
