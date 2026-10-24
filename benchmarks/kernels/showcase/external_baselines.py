"""Isolated upstream baselines; exchange identical inputs using NumPy files."""

import argparse
import ctypes as c
import json
import statistics
from importlib import metadata
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kind", choices=["fa2", "ggml"], required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--library", type=Path)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--repetitions", type=int, default=9)
    args = p.parse_args()
    data = np.load(args.inputs)
    case = json.loads(str(data["case"]))
    expected = data["expected"].astype("float32")
    if args.kind == "ggml":
        lib = c.CDLL(str(args.library.resolve()))
        func = lib.showcase_ggml
        func.restype = c.c_int
        func.argtypes = (
            [c.c_int] * 4 + [c.c_void_p] * 3 + [c.c_int] * 2 + [c.c_void_p] * 2
        )
        actual = np.empty(expected.shape, dtype="float32")
        samples = np.empty(args.repetitions, dtype="float64")
        error = c.create_string_buffer(2048)
        inputs = [np.ascontiguousarray(data[f"input{i}"]) for i in range(2)]
        status = func(
            *case["shape"],
            case["encoding"],
            inputs[1].ctypes.data,
            inputs[0].ctypes.data,
            actual.ctypes.data,
            args.iterations,
            args.repetitions,
            samples.ctypes.data,
            error,
        )
        if status:
            raise RuntimeError(error.value.decode())
        # GGML may quantize F32 activations to Q8_1. Qualify this approximation
        # separately; never silently apply the fused FP16 kernel's tolerance.
        relative_l2 = float(
            np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-12)
        )
        max_abs = float(np.max(np.abs(actual - expected)))
        assert np.isfinite(actual).all() and relative_l2 <= 0.02, (relative_l2, max_abs)
        result = {
            "status": "passed",
            "provider": "ggml_cuda",
            "correctness": {
                "max_abs_error": max_abs,
                "relative_l2_error": relative_l2,
                "relative_l2_limit": 0.02,
            },
            "activation_precision": "GGML default dispatch, potentially Q8_1 quantized activations",
            "timing": "CUDA events on GGML backend stream; repeated full ggml_mul_mat in captured CUDA graph",
        }
        samples = samples.tolist()
    else:
        import torch
        from flash_attn import flash_attn_func

        torch.set_grad_enabled(False)
        inputs = [torch.from_numpy(data[f"input{i}"]).cuda() for i in range(3)]
        # Free stride-only view, preserving the harness's original BHSD inputs.
        q, k, v = [t.transpose(1, 2) for t in inputs]

        def call():
            return flash_attn_func(
                q, k, v, dropout_p=0.0, causal=case["causal"]
            ).transpose(1, 2)

        actual = call().float().cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=2e-2, atol=3e-3)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                call()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(args.iterations):
                    call()
            for _ in range(3):
                graph.replay()
            stream.synchronize()
            samples = []
            for _ in range(args.repetitions):
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000 / args.iterations)
        result = {
            "status": "passed",
            "provider": "upstream_fa2",
            "correctness": {
                "max_abs_error": float(np.max(np.abs(actual - expected))),
                "rtol": 0.02,
                "atol": 0.003,
            },
            "versions": {
                name: metadata.version(name) for name in ("torch", "flash-attn")
            },
            "cuda": torch.version.cuda,
            "layout": "BHSD inputs via stride-only BSHD views; output in native FA BSHD storage, viewed as BHSD; no output contiguous copy",
            "timing": "CUDA events over repeated calls in a captured CUDA graph",
        }
    result.update(
        median_us=statistics.median(samples),
        min_us=min(samples),
        max_us=max(samples),
        samples_us=samples,
    )
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
