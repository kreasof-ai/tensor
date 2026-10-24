"""Opt-in semantic operation selection before native CUDA/TileLang lowering."""

from __future__ import annotations

import hashlib
from pathlib import Path


def lowering_identity():
    root = Path(__file__).parent
    return hashlib.sha256(
        b"".join(
            (root / name).read_bytes()
            for name in ("prelower.py", "matmul.py", "schedules.py")
        )
    ).hexdigest()


def lower_cuda_operations(kernel, target):
    """Return the original explicit kernel or a checked matmul specialization."""
    import tvm

    from tensor.compiler.cuda.matmul import matmul, matmul_partials, tiled_matmul

    operation = str(kernel.attrs.get("tensor.cuda.operation", ""))
    if not operation or kernel.attrs.get("tensor.cuda.prelowered", 0):
        return kernel, None
    if operation not in ("matmul", "matmul_partials"):
        raise ValueError(f"unsupported CUDA operation {operation}")
    if len(kernel.params) != 3 or len(kernel.buffer_map) != 3:
        raise ValueError("CUDA matmul needs exactly x, w and out buffers")
    x, w, out = (kernel.buffer_map[p] for p in kernel.params)
    if [str(b.name) for b in (x, w, out)] != ["x", "w", "out"]:
        raise ValueError("CUDA matmul buffer names must be x, w and out")
    if any(
        len(b.shape) != 2 or any(not isinstance(v, tvm.tirx.IntImm) for v in b.shape)
        for b in (x, w)
    ):
        raise ValueError("CUDA matmul requires static rank-two buffers")
    m, k = map(int, x.shape)
    n, wk = map(int, w.shape)
    splits = int(kernel.attrs.get("tensor.cuda.splits", 1))
    if operation == "matmul_partials" and (splits < 2 or k % splits):
        raise ValueError("invalid CUDA matmul partitions")
    shape = (m, n) if operation == "matmul" else (splits, m, n)
    if wk != k or tuple(map(int, out.shape)) != shape or str(x.dtype) != str(w.dtype):
        raise ValueError("inconsistent CUDA matmul shapes or dtypes")
    if operation == "matmul_partials":
        if str(out.dtype) != "float32":
            raise ValueError("CUDA matmul partitions require FP32 output")
        expected = matmul_partials(m, n, k, splits, str(x.dtype))
    else:
        if splits != 1:
            raise ValueError("ordinary matmul cannot return K partitions")
        expected = matmul(m, n, k, str(x.dtype), str(out.dtype))

    # An annotation cannot authorize replacing arbitrary computation. Match
    # the complete canonical operation, including stores and accumulation.
    def computation(f):
        return tvm.tirx.PrimFunc(f.params, f.body, f.ret_type, f.buffer_map)

    if not tvm.ir.structural_equal(
        computation(kernel), computation(expected), map_free_vars=True
    ):
        raise ValueError("CUDA matmul annotation does not match its computation")
    hopper = target in ("sm_90", "sm_90a")
    defaults = {
        "block_m": 128 if m >= 128 else 32,
        "block_n": 32,
        "block_k": 64,
        "stages": 3,
        "threads": 128,
        "orientation": "weight",
    }
    schedule = {}
    for key, default in defaults.items():
        value = kernel.attrs.get(f"tensor.cuda.{key}", default)
        schedule[key] = str(value) if key == "orientation" else int(value)
    bm, bn, bk, st, threads = (
        schedule[key] for key in ("block_m", "block_n", "block_k", "stages", "threads")
    )
    if (
        min(bm, bn, bk, st, threads) <= 0
        or threads not in (128, 256, 512)
        or schedule["orientation"] not in ("weight", "activation")
    ):
        raise ValueError("invalid CUDA matmul geometry")
    if bm % 16 or bn % 8 or bk % 16 or st > 4:
        raise ValueError("unsupported CUDA Tensor Core tile dimensions")
    if splits > 1 and (k // splits) % bk:
        raise ValueError("matmul K partitions must align to the K tile")
    shared = (bm + bn) * bk * 2 * st
    limit = 227328 if hopper else 99 * 1024
    if shared > limit:
        raise ValueError("CUDA matmul schedule exceeds shared-memory limit")
    from tensor.compiler.cuda.schedules import matmul_legal

    if not matmul_legal(schedule, target=target):
        raise ValueError(
            "CUDA matmul tile cannot be distributed across the requested warps"
        )
    lowered = tiled_matmul(
        m, n, k, str(x.dtype), str(out.dtype), **schedule, splits=splits
    )
    lowered = lowered.with_attr("global_symbol", str(kernel.attrs["global_symbol"]))
    lowered = lowered.with_attr("tensor.cuda.prelowered", 1)
    return lowered, {
        "operation": operation,
        "parameters": {
            "m": m,
            "n": n,
            "k": k,
            "dtype": str(x.dtype),
            "out_dtype": str(out.dtype),
            "splits": splits,
        },
        "schedule": schedule,
        "estimated_shared_memory_bytes": shared,
    }
