"""Batched projection GEMM, including transposes and tails, with FP32 accumulation."""


def make_kernel(p):
    import tilelang.language as T

    batch, m, k, n = p["batch"], p["m"], p["k"], p["n"]
    ta, tb, dtype = p["ta"], p["tb"], p["dtype"]

    @T.prim_func
    def kernel(
        x: T.Tensor((batch, k, m) if ta else (batch, m, k), dtype),
        y: T.Tensor((batch, n, k) if tb else (batch, k, n), dtype),
        out: T.Tensor((batch, m, n), dtype),
    ):
        with T.Kernel(T.ceildiv(n, 64), T.ceildiv(m, 32), batch, threads=128) as (
            bx,
            by,
            b,
        ):
            lhs = T.alloc_shared((32, 32), dtype)
            rhs = T.alloc_shared((32, 64), dtype)
            accum = T.alloc_fragment((32, 64), "float32")
            T.clear(accum)
            for tile in T.Pipelined(T.ceildiv(k, 32), num_stages=2):
                for i, j in T.Parallel(32, 32):
                    lhs[i, j] = T.if_then_else(
                        (by * 32 + i < m) & (tile * 32 + j < k),
                        x[b, tile * 32 + j, by * 32 + i]
                        if ta
                        else x[b, by * 32 + i, tile * 32 + j],
                        0,
                    )
                for i, j in T.Parallel(32, 64):
                    rhs[i, j] = T.if_then_else(
                        (tile * 32 + i < k) & (bx * 64 + j < n),
                        y[b, bx * 64 + j, tile * 32 + i]
                        if tb
                        else y[b, tile * 32 + i, bx * 64 + j],
                        0,
                    )
                T.gemm(lhs, rhs, accum)
            T.copy(accum, out[b, by * 32 : (by + 1) * 32, bx * 64 : (bx + 1) * 64])

    return kernel
