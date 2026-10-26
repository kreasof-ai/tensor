"""Persistent routed GEMM over NVFP4 E2M1 weights with fused BF16 unpacking.

Weights remain packed. FP8 block scales cover sixteen K values and the FP32
expert scale is applied before the transient Tensor Core operand is rounded.
"""


def grouped_matmul(
    m,
    n,
    k,
    total_experts,
    local_experts,
    *,
    top=8,
    block_m=16,
    block_n=64,
    block_k=128,
    stages=2,
    workers=264,
    routed_input=False,
):
    import tilelang.language as T

    if k % block_k or block_k % 16 or block_m % 8 or block_n % 16:
        raise ValueError("aligned NVFP4/Tensor Core tiles required")
    max_tiles = local_experts * ((m + block_m - 1) // block_m)
    columns = (n + block_n - 1) // block_n
    source_rows = m * top if routed_input else m

    @T.prim_func
    def compute(
        x: T.Tensor((source_rows, k), "bfloat16"),
        weights: T.Tensor((total_experts, n, k // 2), "uint8"),
        scales: T.Tensor((total_experts, n, k // 16), "float8_e4m3"),
        global_scale: T.Tensor((total_experts,), "float32"),
        routes: T.Tensor((local_experts, m), "int32"),
        counts: T.Tensor((local_experts,), "int32"),
        tile_experts: T.Tensor((max_tiles,), "int32"),
        tile_rows: T.Tensor((max_tiles,), "int32"),
        tile_count: T.Tensor((1,), "int32"),
        out: T.Tensor((m * top, n), "bfloat16"),
    ):
        with T.Kernel(workers, threads=128) as worker:
            a = T.alloc_shared((block_n, block_k), "bfloat16")
            b = T.alloc_shared((block_m, block_k), "bfloat16")
            acc = T.alloc_fragment((block_n, block_m), "float32")
            task = T.alloc_var("int32")
            task = worker
            while task < tile_count[0] * columns:
                tile = task // columns
                expert = tile_experts[tile]
                first = tile_rows[tile]
                column = task % columns
                T.clear(acc)
                for step in T.Pipelined(k // block_k, num_stages=stages):
                    for i, j in T.Parallel(block_m, block_k):
                        route = routes[expert, T.min(first + i, m - 1)]
                        b[i, j] = T.if_then_else(
                            first + i < counts[expert],
                            x[
                                T.max(route, 0) if routed_input else T.max(route, 0) // top,
                                step * block_k + j,
                            ],
                            0,
                        )
                    for i, j in T.Parallel(block_n, block_k):
                        row = column * block_n + i
                        channel = step * block_k + j
                        packed = weights[expert, T.min(row, n - 1), channel // 2]
                        code = (T.cast(packed, "int32") >> ((channel % 2) * 4)) & 15
                        magnitude = code & 7
                        value = T.if_then_else(
                            magnitude < 2,
                            T.cast(magnitude, "float32") * 0.5,
                            T.if_then_else(
                                magnitude < 4,
                                1.0 + T.cast(magnitude - 2, "float32") * 0.5,
                                T.if_then_else(
                                    magnitude < 6,
                                    2.0 + T.cast(magnitude - 4, "float32"),
                                    4.0 + T.cast(magnitude - 6, "float32") * 2.0,
                                ),
                            ),
                        )
                        coefficient = T.if_then_else(code & 8 != 0, -value, value)
                        a[i, j] = T.if_then_else(
                            row < n,
                            coefficient
                            * T.cast(scales[expert, T.min(row, n - 1), channel // 16], "float32")
                            * global_scale[expert],
                            0,
                        )
                    T.gemm(a, b, acc, transpose_B=True)
                for i, j in T.Parallel(block_m, block_n):
                    if first + i < counts[expert] and column * block_n + j < n:
                        route = routes[expert, first + i]
                        out[route, column * block_n + j] = acc[j, i]
                task += workers

    return compute
