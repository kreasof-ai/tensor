"""NVIDIA-only NumPy-reference validation of the corrected P0 workloads.

PyTorch is used as the existing JIT adapter's device-buffer client here. The
separate opaque-artifact consumer does not use it. Reports label no-device
runs as skipped; they never substitute source emission for numerics.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from pathlib import Path

from experiments.p0.cuda_driver import CudaUnavailable, Driver
from experiments.p0.provenance import snapshot


def run() -> dict:
    _, device = Driver().device_info()
    if not shutil.which("nvcc"):
        raise RuntimeError("numerics requires nvcc on the NVIDIA host")
    import numpy as np
    import torch
    import tilelang
    from experiments.p0 import kernels as K

    if not torch.cuda.is_available():
        raise RuntimeError("the installed PyTorch wheel cannot allocate CUDA buffers")
    rng = np.random.default_rng(0)
    rows = []

    def random(shape, dtype="float16"):
        return rng.standard_normal(shape).astype(dtype)

    def check(name, func, inputs, expected, rtol=1e-2, atol=1e-2):
        started = time.perf_counter()
        compiled = tilelang.compile(func, out_idx=None, execution_backend="tvm_ffi",
                                   target={"kind": "cuda", "arch": device["arch"]})
        buffers = [torch.from_numpy(value).cuda() for value in inputs]
        output = torch.full(expected.shape, float("nan"), dtype=buffers[-1].dtype, device="cuda")
        compiled(*buffers, output)
        torch.cuda.synchronize()
        actual = output.cpu().numpy()
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
        rows.append({"name": name, "status": "passed", "shape": list(expected.shape),
                     "seconds": time.perf_counter() - started,
                     "max_abs_error": float(np.max(np.abs(actual.astype(np.float32) - expected.astype(np.float32)))),
                     "rtol": rtol, "atol": atol})

    for m, n in ((16, 128), (17, 129)):
        a, b = random((m, n)), random((m, n))
        expected = np.maximum(2 * a.astype(np.float32) + b.astype(np.float32), 0).astype(np.float16)
        check(f"elementwise_{m}x{n}", K.fused_elementwise(m, n), [a, b], expected)
    for m, n, k in ((64, 64, 32), (65, 67, 33)):
        a, b, bias = random((m, k)), random((k, n)), random((n,))
        expected = np.maximum(a.astype(np.float32) @ b.astype(np.float32) + bias.astype(np.float32), 0).astype(np.float16)
        check(f"gemm_{m}x{n}x{k}", K.gemm_relu(m, n, k), [a, b, bias], expected)
    for n in (1, 127, 128, 129, 1024):
        x = random((3, n), "float32")
        check(f"row_sum_N{n}", K.row_sum(3, n), [x], x.sum(axis=1), rtol=1e-5, atol=1e-5)
    index = np.array([-1, 0, 18, 19] + rng.integers(0, 19, 13).tolist(), dtype=np.int32)
    x = random((19, 129))
    expected = x[np.clip(index, 0, 18)].copy()
    expected[(index < 0) | (index >= 19)] = 0
    check("gather_partial_and_invalid_indices", K.gather_rows(17, 19, 129), [index, x], expected, rtol=0, atol=0)
    for n in (64, 65):
        q, k, v = [random((2, n, 64)) for _ in range(3)]
        scores = q.astype(np.float32) @ k.astype(np.float32).swapaxes(-1, -2) / 8
        weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights /= weights.sum(axis=-1, keepdims=True)
        expected = (weights @ v.astype(np.float32)).astype(np.float16)
        check(f"attention_N{n}", K.flash_attention(n, 2, 64), [q, k, v], expected)
    return {"status": "passed", "device": device, "provenance": snapshot(),
            "host": platform.platform(), "seed": 0, "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("experiments/p0/out/numerics.json"))
    args = parser.parse_args()
    try:
        result, code = run(), 0
    except CudaUnavailable as exc:
        result, code = {"status": "skipped", "reason": str(exc), "numerics": "unverified", "provenance": snapshot()}, 2
    except Exception as exc:
        result, code = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "provenance": snapshot()}, 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
