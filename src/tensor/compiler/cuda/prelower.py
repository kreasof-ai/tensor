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
    if (any(len(b.shape) != 2 for b in (x, w))
            or not isinstance(x.shape[0], (tvm.tirx.IntImm, tvm.tirx.Var))
            or any(not isinstance(v, tvm.tirx.IntImm) for v in (*w.shape, *x.shape[1:]))):
        raise ValueError("CUDA matmul requires rank-two buffers and static N/K")
    m = int(x.shape[0]) if isinstance(x.shape[0], tvm.tirx.IntImm) else x.shape[0]
    k = int(x.shape[1])
    n, wk = map(int, w.shape)
    splits = int(kernel.attrs.get("tensor.cuda.splits", 1))
    if operation == "matmul_partials" and (splits < 2 or k % splits):
        raise ValueError("invalid CUDA matmul partitions")
    shape = (m, n) if operation == "matmul" else (splits, m, n)
    analyzer = tvm.arith.Analyzer()
    if (wk != k or len(out.shape) != len(shape)
            or any(not analyzer.can_prove_equal(a, b) for a, b in zip(out.shape, shape))
            or str(x.dtype) != str(w.dtype)):
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
    simt = str(x.dtype) == "float32"
    lanes = int(kernel.attrs.get("tensor.cuda.reduction_lanes", 1))
    defaults = {
        "block_m": (8 if lanes > 1 else 16) if simt else 128 if type(m) is int and m >= 128 else 32,
        "block_n": 8 if simt and lanes > 1 else 32,
        "block_k": (128 if lanes > 1 else 32) if simt else 64,
        "stages": 2 if simt else 3,
        "threads": 128,
        "orientation": "activation" if simt else "weight",
        "reduction_lanes": 1,
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
    if (not simt and (bm % 16 or bn % 8 or bk % 16)) or st > 4:
        raise ValueError("unsupported CUDA Tensor Core tile dimensions")
    if lanes not in (1, 2, 4, 8, 16, 32) or bk % lanes or (not simt and lanes != 1):
        raise ValueError("invalid CUDA matmul reduction lanes")
    if splits > 1 and (k // splits) % bk:
        raise ValueError("matmul K partitions must align to the K tile")
    from tensor.compiler.cuda.schedules import matmul_legal, matmul_shared

    shared = matmul_shared(schedule, dtype=str(x.dtype))
    limit = 227328 if hopper else 99 * 1024
    if shared > limit:
        raise ValueError("CUDA matmul schedule exceeds shared-memory limit")
    if not matmul_legal(schedule, target=target, dtype=str(x.dtype)):
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
        "algorithm": "simt_fma" if simt else "tensor_core",
        "parameters": {
            "m": m if type(m) is int else {"var": str(m.name)},
            "n": n,
            "k": k,
            "dtype": str(x.dtype),
            "out_dtype": str(out.dtype),
            "splits": splits,
        },
        "schedule": schedule,
        "estimated_shared_memory_bytes": shared,
    }
