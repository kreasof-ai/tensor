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

    if k % block_k or block_k % 128 or block_m % 8 or block_n % 16:
        raise ValueError("aligned NVFP4/Tensor Core tiles required")
    max_tiles = local_experts * ((m + block_m - 1) // block_m)
    columns = (n + block_n - 1) // block_n
    source_rows = m * top if routed_input else m

    @T.prim_func
    def nvfp4_grouped(
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
            packed_tile = T.alloc_shared((block_n, block_k // 2), "uint8")
            decoded = T.alloc_fragment((4, block_k), "bfloat16")
            scale_tile = T.alloc_shared((block_n, block_k // 16), "float8_e4m3")
            # A warp owns one weight row. Each lane loads two packed bytes and
            # expands four adjacent coefficients. This prevents layout inference
            # from assigning all 128 lanes to K and serializing 64 weight rows.
            T.annotate_layout(
                {
                    decoded: T.Fragment(
                        (4, block_k),
                        forward_thread_fn=lambda i, j: i * 32 + j // 4 % 32,
                        forward_index_fn=lambda i, j: j // 128 * 4 + j % 4,
                    ),
                }
            )
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
                    T.copy(weights[expert, column * block_n, step * block_k // 2], packed_tile)
                    T.copy(scales[expert, column * block_n, step * block_k // 16], scale_tile)
                    for group in T.serial(block_n // 4):
                        for i, j in T.Parallel(4, block_k):
                            row = group * 4 + i
                            code = (T.cast(packed_tile[row, j // 2], "int32") >> (j % 2 * 4)) & 15
                            magnitude = code & 7
                            bits = (
                                T.cast(magnitude > 0, "int32")
                                * (0x3F000000 + T.cast(magnitude >= 2, "int32") * (magnitude << 22))
                            ) | ((code & 8) << 28)
                            coefficient = T.reinterpret(bits, "float32")
                            decoded[i, j] = (
                                coefficient
                                * T.cast(scale_tile[row, j // 16], "float32")
                                * global_scale[expert]
                            )
                        T.copy(decoded, a[group * 4, 0])
                    T.gemm(a, b, acc, transpose_B=True)
                for i, j in T.Parallel(block_m, block_n):
                    if first + i < counts[expert] and column * block_n + j < n:
                        route = routes[expert, first + i]
                        out[route, column * block_n + j] = acc[j, i]
                task += workers

    return nvfp4_grouped
