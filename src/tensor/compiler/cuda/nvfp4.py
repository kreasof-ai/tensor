"""Persistent routed GEMM over NVFP4 E2M1 weights with fused BF16 unpacking.

Weights remain packed. FP8 block scales cover sixteen K values and the FP32
expert scale enters the FP32 epilogue before the BF16 output boundary.
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
    decode_rows=16,
):
    import tilelang.language as T

    if k % block_k or block_k % 128 or block_m % 8 or block_n % 16:
        raise ValueError("aligned NVFP4/Tensor Core tiles required")
    if decode_rows not in (4, 8, 16, 32, 64) or block_n % decode_rows:
        raise ValueError("decode rows must divide the output tile")
    lanes = 128 // decode_rows
    max_tiles = local_experts * ((m + block_m - 1) // block_m)
    columns = (n + block_n - 1) // block_n
    source_rows = m * top if routed_input else m

    unpack_source = r"""
__device__ __forceinline__ void tensor_nvfp4_store2(void* dst, unsigned packed, unsigned scale) {
    unsigned lo = packed & 7, hi = (packed >> 4) & 7;
    lo = lo ? 0x3f00 + (lo >= 2 ? (lo << 6) : 0) : 0;
    hi = hi ? 0x3f00 + (hi >= 2 ? (hi << 6) : 0) : 0;
    unsigned coefficients = lo | (hi << 16) | ((packed & 8) << 12) | ((packed & 128) << 24);
    unsigned scales = scale | (scale << 16), result;
#if __CUDA_ARCH__ >= 900
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(coefficients), "r"(scales));
#else
    // E2M1 times E4M3 fits BF16 exactly, including subnormal E4M3 scales.
    float sf = __uint_as_float(scale << 16);
    unsigned low = __float_as_uint(__uint_as_float(coefficients << 16) * sf) >> 16;
    unsigned high = __float_as_uint(__uint_as_float(coefficients & 0xffff0000u) * sf) & 0xffff0000u;
    result = low | high;
#endif
    *reinterpret_cast<unsigned*>(dst) = result;
}
"""

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
            scale_tile = T.alloc_shared((block_n, block_k // 16), "float8_e4m3")
            T.import_source(unpack_source)
            task = T.alloc_var("int32")
            task = worker
            while task < tile_count[0] * columns:
                tile = task // columns
                expert = tile_experts[tile]
                first = tile_rows[tile]
                column = task % columns
                scale = T.alloc_var("float32")
                scale = global_scale[expert]
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
                    for group in T.serial(block_n // decode_rows):
                        for i, pair in T.Parallel(
                            decode_rows, block_k // 2,
                            loop_layout=T.Fragment((decode_rows, block_k // 2),
                                forward_thread_fn=lambda rr, cc: rr * lanes + cc % lanes,
                                forward_index_fn=lambda rr, cc: cc // lanes),
                        ):
                            row = group * decode_rows + i
                            T.call_extern("handle", "tensor_nvfp4_store2",
                                          T.access_ptr(a[group * decode_rows + i, pair * 2], "w", 2),
                                          T.cast(packed_tile[row, pair], "uint32"),
                                          T.cast(T.reinterpret(T.cast(scale_tile[row, pair // 8], "bfloat16"), "uint16"), "uint32"))
                    T.gemm(a, b, acc, transpose_B=True)
                for i, j in T.Parallel(block_m, block_n):
                    if first + i < counts[expert] and column * block_n + j < n:
                        route = routes[expert, first + i]
                        out[route, column * block_n + j] = acc[j, i] * scale
                task += workers

    return nvfp4_grouped


def pipelined_grouped_matmul(
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
    decode_rows=16,
    native_async=True,
):
    import tilelang.language as T

    if k % block_k or block_k % 128 or block_m % 8 or block_n % 16:
        raise ValueError("aligned NVFP4/Tensor Core tiles required")
    if decode_rows not in (4, 8, 16, 32, 64) or block_n % decode_rows:
        raise ValueError("decode rows must divide the output tile")
    lanes = 128 // decode_rows
    if stages != 2 or k % (4 * block_k):
        raise ValueError("two-stage ring requires a whole even number of barrier phases")
    max_tiles = local_experts * ((m + block_m - 1) // block_m)
    columns = (n + block_n - 1) // block_n
    source_rows = m * top if routed_input else m

    unpack_source = r"""
__device__ __forceinline__ void tensor_nvfp4_store2(void* dst, unsigned packed, unsigned scale) {
    unsigned lo = packed & 7, hi = (packed >> 4) & 7;
    lo = lo ? 0x3f00 + (lo >= 2 ? (lo << 6) : 0) : 0;
    hi = hi ? 0x3f00 + (hi >= 2 ? (hi << 6) : 0) : 0;
    unsigned coefficients = lo | (hi << 16) | ((packed & 8) << 12) | ((packed & 128) << 24);
    unsigned scales = scale | (scale << 16), result;
#if __CUDA_ARCH__ >= 900
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(result) : "r"(coefficients), "r"(scales));
#else
    // E2M1 times E4M3 fits BF16 exactly, including subnormal E4M3 scales.
    float sf = __uint_as_float(scale << 16);
    unsigned low = __float_as_uint(__uint_as_float(coefficients << 16) * sf) >> 16;
    unsigned high = __float_as_uint(__uint_as_float(coefficients & 0xffff0000u) * sf) & 0xffff0000u;
    result = low | high;
#endif
    *reinterpret_cast<unsigned*>(dst) = result;
}
"""

    @T.prim_func
    def nvfp4_pipelined(
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
        with T.Kernel(workers, threads=256) as worker:
            a = T.alloc_shared((2, block_n, block_k), "bfloat16")
            b = T.alloc_shared((2, block_m, block_k), "bfloat16")
            acc = T.alloc_fragment((block_n, block_m), "float32")
            packed_tile = T.alloc_shared((block_n, block_k // 2), "uint8")
            scale_tile = T.alloc_shared((block_n, block_k // 16), "float8_e4m3")
            T.import_source(unpack_source)
            if native_async:
                ready = T.alloc_barrier([128, 128])
                free = T.alloc_barrier([128, 128])
            T.sync_threads()
            tx = T.get_thread_binding()
            if not native_async:
                if tx >= 128:
                    for warm in T.unroll(2):
                        T.named_barrier_arrive(4 + warm, 256)
            task = T.alloc_var("int32")
            task = worker
            while task < tile_count[0] * columns:
                tile = task // columns
                expert = tile_experts[tile]
                first = tile_rows[tile]
                column = task % columns
                if tx < 128:
                    for step in T.serial(k // block_k):
                        if native_async:
                            T.mbarrier_wait_parity(free[step % 2], ((step // 2) % 2) ^ 1)
                        else:
                            T.sync_threads(4 + step % 2, 256)
                        for i, j in T.Parallel(block_m, block_k):
                            route = routes[expert, T.min(first + i, m - 1)]
                            b[step % 2, i, j] = T.if_then_else(
                                first + i < counts[expert],
                                x[
                                    T.max(route, 0) if routed_input else T.max(route, 0) // top,
                                    step * block_k + j,
                                ],
                                0,
                            )
                        T.async_copy(weights[expert, column * block_n, step * block_k // 2], packed_tile)
                        T.async_copy(scales[expert, column * block_n, step * block_k // 16], scale_tile)
                        T.ptx_wait_group(0)
                        T.sync_threads(1, 128)
                        for group in T.serial(block_n // decode_rows):
                            for i, pair in T.Parallel(
                                decode_rows, block_k // 2,
                                loop_layout=T.Fragment((decode_rows, block_k // 2),
                                    forward_thread_fn=lambda rr, cc: rr * lanes + cc % lanes,
                                    forward_index_fn=lambda rr, cc: cc // lanes),
                            ):
                                row = group * decode_rows + i
                                T.call_extern("handle", "tensor_nvfp4_store2",
                                              T.access_ptr(a[step % 2, group * decode_rows + i, pair * 2], "w", 2),
                                              T.cast(packed_tile[row, pair], "uint32"),
                                              T.cast(T.reinterpret(T.cast(scale_tile[row, pair // 8], "bfloat16"), "uint16"), "uint32"))
                        if native_async:
                            T.fence_proxy_async()
                            T.mbarrier_arrive(ready[step % 2])
                        else:
                            T.named_barrier_arrive(2 + step % 2, 256)
                else:
                    T.clear(acc)
                    for step in T.serial(k // block_k):
                        if native_async:
                            T.mbarrier_wait_parity(ready[step % 2], (step // 2) % 2)
                            T.wgmma_gemm(a[step % 2, :, :], b[step % 2, :, :], acc, transpose_B=True)
                            T.wait_wgmma(0)
                            T.mbarrier_arrive(free[step % 2])
                        else:
                            T.sync_threads(2 + step % 2, 256)
                            T.gemm(a[step % 2, :, :], b[step % 2, :, :], acc, transpose_B=True)
                            T.named_barrier_arrive(4 + step % 2, 256)
                    if native_async:
                        T.warpgroup_fence_operand(acc, num_regs=block_n * block_m // 128)
                    for i, j in T.Parallel(block_m, block_n):
                        if first + i < counts[expert] and column * block_n + j < n:
                            route = routes[expert, first + i]
                            out[route, column * block_n + j] = acc[j, i] * global_scale[expert]
                task += workers
    return nvfp4_pipelined
