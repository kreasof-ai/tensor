"""Logical CUDA matmul and its native Tensor Core schedule implementations.

The public operation computes x @ w.T with FP32 accumulation. Scheduling does
not change operand precision. Explicitly scheduled third-party kernels remain
outside this operation contract.
"""


def matmul(m, n, k, dtype="bfloat16", out_dtype=None, schedule=None, **options):
    """Describe a matrix product; CUDA prelowering chooses its implementation."""
    import tilelang.language as T

    from tvm import tirx

    if (any(type(x) is not int or x <= 0 for x in (n, k))
            or not ((type(m) is int and m > 0)
                    or (isinstance(m, tirx.Var) and str(m.dtype) in ("int32", "int64")))):
        raise ValueError("matmul needs static positive N/K and positive or symbolic M")
    out_dtype = dtype if out_dtype is None else out_dtype
    if dtype not in ("float16", "bfloat16", "float32") or out_dtype not in (dtype, "float32"):
        raise ValueError(
            "CUDA matmul needs FP16/BF16/FP32 inputs and matching or FP32 output"
        )
    allowed = {"block_m", "block_n", "block_k", "stages", "threads", "orientation", "reduction_lanes"}
    schedule = dict(schedule or {})
    if schedule.keys() & options.keys():
        raise ValueError("duplicate CUDA matmul schedule option")
    schedule.update(options)
    if set(schedule) - allowed:
        raise ValueError("unknown CUDA matmul schedule option")
    attrs = {"tensor.cuda.operation": "matmul"}
    attrs.update({f"tensor.cuda.{name}": value for name, value in schedule.items()})

    @T.prim_func
    def product(
        x: T.Tensor((m, k), dtype),
        w: T.Tensor((n, k), dtype),
        out: T.Tensor((m, n), out_dtype),
    ):
        T.func_attr(attrs)
        for i, j in T.Parallel(m, n):
            acc = T.alloc_var("float32")
            acc = 0
            for r in T.serial(k):
                acc += x[i, r].astype("float32") * w[j, r].astype("float32")
            out[i, j] = acc

    return product


def tiled_matmul(
    m,
    n,
    k,
    dtype,
    out_dtype,
    block_m,
    block_n,
    block_k,
    stages,
    threads,
    orientation,
    splits=1,
    reduction_lanes=1,
):
    """Emit either operand orientation while keeping the public x/w/out ABI."""
    import tilelang.language as T

    swapped = orientation == "weight"
    simt = dtype == "float32"
    left_rows, right_rows = (block_n, block_m) if swapped else (block_m, block_n)
    shape = (m, n) if splits == 1 else (splits, m, n)
    depth = k if splits == 1 else k // splits

    @T.prim_func
    def product(
        x: T.Tensor((m, k), dtype),
        w: T.Tensor((n, k), dtype),
        out: T.Tensor(shape, out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(n, block_n), T.ceildiv(m, block_m), splits, threads=threads
        ) as (bx, by, part):
            a = T.alloc_shared((left_rows, block_k), dtype)
            # Padding distributes the SIMT column reads across shared banks.
            b = T.alloc_shared((right_rows, block_k + (1 if simt else 0)), dtype)
            acc = T.alloc_fragment((left_rows, right_rows), "float32")
            T.clear(acc)
            if simt and reduction_lanes > 1:
                partial = T.alloc_fragment((left_rows, right_rows, reduction_lanes), "float32")
                T.clear(partial)
            for local in T.Pipelined(T.ceildiv(depth, block_k), num_stages=stages):
                r = part * T.ceildiv(depth, block_k) + local
                if swapped:
                    T.copy(w[bx * block_n, r * block_k], a)
                    T.copy(x[by * block_m, r * block_k], b[0:right_rows, 0:block_k])
                else:
                    T.copy(x[by * block_m, r * block_k], a)
                    T.copy(w[bx * block_n, r * block_k], b[0:right_rows, 0:block_k])
                if simt:
                    if reduction_lanes == 1:
                        for q in T.serial(block_k):
                            for i, j in T.Parallel(left_rows, right_rows):
                                acc[i, j] = T.ieee_fmaf(a[i, q], b[j, q], acc[i, j])
                    else:
                        for q in T.serial(block_k // reduction_lanes):
                            for i, j, lane in T.Parallel(left_rows, right_rows, reduction_lanes):
                                partial[i, j, lane] = T.ieee_fmaf(
                                    a[i, q * reduction_lanes + lane],
                                    b[j, q * reduction_lanes + lane], partial[i, j, lane])
                else:
                    T.gemm(a, b, acc, transpose_B=True)
            if simt and reduction_lanes > 1:
                T.reduce_sum(partial, acc, dim=2)
            for i, j in T.Parallel(block_m, block_n):
                if (by * block_m + i < m) & (bx * block_n + j < n):
                    if swapped:
                        if splits == 1:
                            out[by * block_m + i, bx * block_n + j] = acc[j, i]
                        else:
                            out[part, by * block_m + i, bx * block_n + j] = acc[j, i]
                    else:
                        if splits == 1:
                            out[by * block_m + i, bx * block_n + j] = acc[i, j]
                        else:
                            out[part, by * block_m + i, bx * block_n + j] = acc[i, j]

    return product


def matmul_partials(m, n, k, splits, dtype="bfloat16", schedule=None, **options):
    """Explicit K partitions, returning FP32 [splits,M,N] for reduction."""
    import tilelang.language as T

    if type(splits) is not int or splits < 2 or k % splits:
        raise ValueError("matmul partitions require at least two equal K ranges")
    logical = matmul(m, n, k, dtype, "float32", schedule, **options)
    attrs = {
        str(key): value
        for key, value in logical.attrs.items()
        if str(key).startswith("tensor.cuda.")
    }
    attrs.update(
        {"tensor.cuda.operation": "matmul_partials", "tensor.cuda.splits": splits}
    )
    depth = k // splits

    @T.prim_func
    def product(
        x: T.Tensor((m, k), dtype),
        w: T.Tensor((n, k), dtype),
        out: T.Tensor((splits, m, n), "float32"),
    ):
        T.func_attr(attrs)
        for part, i, j in T.Parallel(splits, m, n):
            acc = T.alloc_var("float32")
            acc = 0
            for local in T.serial(depth):
                r = part * depth + local
                acc += x[i, r].astype("float32") * w[j, r].astype("float32")
            out[part, i, j] = acc

    return product


def matmul_reduce(m, n, splits, out_dtype="bfloat16"):
    """Reduce explicit FP32 partitions; inputs are overwritten by their producer."""
    import tilelang.language as T

    if any(type(x) is not int or x <= 0 for x in (m, n, splits)) or out_dtype not in (
        "float16",
        "bfloat16",
        "float32",
    ):
        raise ValueError("invalid matmul reduction shape or dtype")

    @T.prim_func
    def reduce(
        parts: T.Tensor((splits, m, n), "float32"), out: T.Tensor((m, n), out_dtype)
    ):
        with T.Kernel(T.ceildiv(m * n, 256), threads=128) as bx:
            for lane in T.Parallel(256):
                idx = bx * 256 + lane
                if idx < m * n:
                    acc = T.alloc_var("float32")
                    acc = 0
                    for part in T.serial(splits):
                        acc += parts[part, idx // n, idx % n]
                    out[idx // n, idx % n] = acc

    return reduce
