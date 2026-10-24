"""Warp-level sliced-K matmul: one launch, with a block-local FP32 reduction."""


def sliced_matmul(
    m,
    n,
    k,
    *,
    block_m=8,
    block_n=8,
    warp_splits=4,
    chunk=64,
    stages=2,
    paired=False,
    out_dtype="bfloat16",
):
    import tilelang.language as T
    from tilelang.cuda.intrinsics.layout.utils import mma_store_index_map
    from tilelang.cuda.intrinsics.macro.mma_macro_generator import (
        TensorCoreIntrinEmitter,
    )
    from tilelang.layout import make_swizzled_layout

    if any(
        type(v) is not int or v < 1
        for v in (m, n, k, block_m, block_n, warp_splits, chunk, stages)
    ):
        raise ValueError("positive static shapes required")
    active_rows = block_n * (2 if paired else 1)
    if (
        block_m not in (8, 16)
        or active_rows > 16
        or warp_splits not in (2, 4, 8)
        or chunk % 16
        or k % (warp_splits * chunk)
    ):
        raise ValueError("invalid sliced-K tile")
    if out_dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported sliced-K output dtype")
    bk = warp_splits * chunk
    mma = TensorCoreIntrinEmitter(
        a_dtype="bfloat16",
        b_dtype="bfloat16",
        accum_dtype="float32",
        a_transposed=False,
        b_transposed=True,
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=block_m,
        chunk=chunk,
        reduce_k=warp_splits,
    )

    @T.prim_func
    def product(
        x: T.Tensor((m, k), "bfloat16"),
        weights: T.Tensor((n * (2 if paired else 1), k), "bfloat16"),
        out: T.Tensor((m, n), out_dtype),
    ):
        with T.Kernel(
            T.ceildiv(n, block_n), T.ceildiv(m, block_m), threads=32 * warp_splits
        ) as (
            bx,
            by,
        ):
            a = T.alloc_shared((16, bk), "bfloat16")
            b = T.alloc_shared((block_m, bk), "bfloat16")
            lhs = T.alloc_local((mma.local_size_a,), "bfloat16")
            rhs = T.alloc_local((mma.local_size_b,), "bfloat16")
            acc = T.alloc_local((mma.local_size_out,), "float32")
            partial = T.alloc_shared((warp_splits, 16, block_m), "float32")
            T.annotate_layout({a: make_swizzled_layout(a), b: make_swizzled_layout(b)})
            T.clear(acc)
            rk = T.get_thread_binding() // 32
            lane = T.get_thread_binding() % 32
            for tile in T.Pipelined(k // bk, num_stages=stages):
                T.copy(weights[bx * active_rows, tile * bk], a[0:active_rows, :])
                if active_rows < 16:
                    T.clear(a[active_rows:16, :])
                T.copy(x[by * block_m, tile * bk], b)
                for inner in T.serial(chunk // 16):
                    mma.ldmatrix_a(lhs, a, inner, rk=rk)
                    mma.ldmatrix_b(rhs, b, inner, rk=rk)
                    mma.mma(lhs, rhs, acc)
            for local in T.serial(mma.local_size_out):
                row, col = T.meta_var(mma_store_index_map(lane, local))
                partial[rk, row, col] = acc[local]
            T.sync_threads()
            for i, j in T.Parallel(block_m, block_n):
                if by * block_m + i < m and bx * block_n + j < n:
                    value = T.alloc_var("float32")
                    value = 0
                    for peer in T.serial(warp_splits):
                        if paired:
                            value += (
                                partial[peer, 2 * j, i] + partial[peer, 2 * j + 1, i]
                            )
                        else:
                            value += partial[peer, j, i]
                    out[by * block_m + i, bx * block_n + j] = value

    return product
