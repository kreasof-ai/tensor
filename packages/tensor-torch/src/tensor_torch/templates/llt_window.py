"""Windowed causal attention and one-softmax shared/local memory.

Score and value dimensions are independent. FP32 accumulators/statistics, no
global score matrix, no per-head KV replication, and deterministic KV reduction.
"""


def attention_forward(p):
    import tilelang.language as T

    b, h, kh, m, n, d, dv, dtype, causal, offset, scale = p[:11]
    window, prefix = p[11:13]
    bm, bn = 32, 64

    @T.prim_func
    def kernel(
        q: T.Tensor((b, h, m, d), dtype),
        k: T.Tensor((b, kh, n, d), dtype),
        v: T.Tensor((b, kh, n, dv), dtype),
        out: T.Tensor((b, h, m, dv), dtype),
        lse: T.Tensor((b, h, m), "float32"),
        exact: T.Tensor((b, h, m, dv), "float32"),
    ):
        with T.Kernel(T.ceildiv(m, bm), h, b, threads=128) as (blk, head, batch):
            qs = T.alloc_shared((bm, d), dtype)
            ks = T.alloc_shared((bn, d), dtype)
            vs = T.alloc_shared((bn, dv), dtype)
            ps = T.alloc_shared((bm, bn), dtype)
            scores = T.alloc_fragment((bm, bn), "float32")
            result = T.alloc_fragment((bm, dv), "float32")
            maximum = T.alloc_fragment((bm,), "float32")
            previous = T.alloc_fragment((bm,), "float32")
            factor = T.alloc_fragment((bm,), "float32")
            norm = T.alloc_fragment((bm,), "float32")
            sums = T.alloc_fragment((bm,), "float32")
            stat = T.alloc_shared((bm,), "float32")
            T.copy(q[batch, head, blk * bm : (blk + 1) * bm, :], qs)
            T.clear(result)
            T.clear(norm)
            T.fill(maximum, -T.infinity("float32"))
            count = T.ceildiv(n, bn)
            for tile in T.serial(count):
                if ((tile * bn < prefix) & (tile * bn < offset + (blk + 1) * bm)) | (
                    ((tile + 1) * bn > prefix + offset + blk * bm - window + 1)
                    & (tile * bn < prefix + offset + (blk + 1) * bm)
                    & ((tile + 1) * bn > prefix)
                ):
                    T.copy(
                        k[batch, head // (h // kh), tile * bn : (tile + 1) * bn, :], ks
                    )
                    T.gemm(
                        qs,
                        ks,
                        scores,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(bm, bn):
                        scores[i, j] = T.if_then_else(
                            (tile * bn + j < n)
                            & (
                                (
                                    (tile * bn + j < prefix)
                                    & (tile * bn + j <= offset + blk * bm + i)
                                )
                                | (
                                    (tile * bn + j >= prefix)
                                    & (tile * bn + j - prefix <= offset + blk * bm + i)
                                    & (
                                        tile * bn + j - prefix
                                        > offset + blk * bm + i - window
                                    )
                                )
                            ),
                            scores[i, j] * scale,
                            -T.infinity("float32"),
                        )
                    T.copy(maximum, previous)
                    T.reduce_max(scores, maximum, dim=1, clear=True)
                    for i in T.Parallel(bm):
                        maximum[i] = T.max(maximum[i], previous[i])
                        factor[i] = T.if_then_else(
                            maximum[i] == -T.infinity("float32"),
                            0,
                            T.exp(previous[i] - maximum[i]),
                        )
                    for i, j in T.Parallel(bm, bn):
                        scores[i, j] = T.if_then_else(
                            maximum[i] == -T.infinity("float32"),
                            0,
                            T.exp(scores[i, j] - maximum[i]),
                        )
                    T.reduce_sum(scores, sums, dim=1, clear=True)
                    for i in T.Parallel(bm):
                        norm[i] = norm[i] * factor[i] + sums[i]
                    for i, j in T.Parallel(bm, dv):
                        result[i, j] *= factor[i]
                    T.copy(scores, ps)
                    T.copy(
                        v[batch, head // (h // kh), tile * bn : (tile + 1) * bn, :], vs
                    )
                    T.gemm(ps, vs, result, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(bm, dv):
                result[i, j] = T.if_then_else(norm[i] > 0, result[i, j] / norm[i], 0)
            T.copy(result, exact[batch, head, blk * bm : (blk + 1) * bm, :])
            T.copy(result, out[batch, head, blk * bm : (blk + 1) * bm, :])
            for i in T.Parallel(bm):
                maximum[i] = T.if_then_else(
                    norm[i] > 0, maximum[i] + T.log(norm[i]), -T.infinity("float32")
                )
            T.copy(maximum, stat)
            for i in T.Parallel(bm):
                if blk * bm + i < m:
                    lse[batch, head, blk * bm + i] = stat[i]

    return kernel


def attention_delta(p):
    import tilelang.language as T

    b, h, m, dv, dtype = p
    width = 1 << (dv - 1).bit_length()

    @T.prim_func
    def kernel(
        o: T.Tensor((b, h, m, dv), "float32"),
        dout: T.Tensor((b, h, m, dv), dtype),
        delta: T.Tensor((b, h, m), "float32"),
    ):
        with T.Kernel(m, h, b, threads=128) as (row, head, batch):
            values = T.alloc_fragment((width,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(width):
                values[j] = T.if_then_else(
                    j < dv,
                    o[batch, head, row, j]
                    * T.cast(dout[batch, head, row, j], "float32"),
                    0,
                )
            T.reduce_sum(values, total, dim=0, clear=True)
            delta[batch, head, row] = total[0]

    return kernel


def attention_dq(p):
    import tilelang.language as T

    b, h, kh, m, n, d, dv, dtype, causal, offset, scale = p[:11]
    window, prefix = p[11:13]
    bm, bn = 32, 64

    @T.prim_func
    def kernel(
        q: T.Tensor((b, h, m, d), dtype),
        k: T.Tensor((b, kh, n, d), dtype),
        v: T.Tensor((b, kh, n, dv), dtype),
        dout: T.Tensor((b, h, m, dv), dtype),
        lse: T.Tensor((b, h, m), "float32"),
        delta: T.Tensor((b, h, m), "float32"),
        dq: T.Tensor((b, h, m, d), dtype),
    ):
        with T.Kernel(T.ceildiv(m, bm), h, b, threads=128) as (blk, head, batch):
            qs = T.alloc_shared((bm, d), dtype)
            ks = T.alloc_shared((bn, d), dtype)
            vs = T.alloc_shared((bn, dv), dtype)
            dos = T.alloc_shared((bm, dv), dtype)
            ds = T.alloc_shared((bm, bn), dtype)
            prob = T.alloc_fragment((bm, bn), "float32")
            dp = T.alloc_fragment((bm, bn), "float32")
            result = T.alloc_fragment((bm, d), "float32")
            T.copy(q[batch, head, blk * bm : (blk + 1) * bm, :], qs)
            T.copy(dout[batch, head, blk * bm : (blk + 1) * bm, :], dos)
            T.clear(result)
            for tile in T.serial(T.ceildiv(n, bn)):
                if ((tile * bn < prefix) & (tile * bn < offset + (blk + 1) * bm)) | (
                    ((tile + 1) * bn > prefix + offset + blk * bm - window + 1)
                    & (tile * bn < prefix + offset + (blk + 1) * bm)
                    & ((tile + 1) * bn > prefix)
                ):
                    T.copy(
                        k[batch, head // (h // kh), tile * bn : (tile + 1) * bn, :], ks
                    )
                    T.copy(
                        v[batch, head // (h // kh), tile * bn : (tile + 1) * bn, :], vs
                    )
                    T.gemm(
                        qs,
                        ks,
                        prob,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.gemm(
                        dos,
                        vs,
                        dp,
                        transpose_B=True,
                        clear_accum=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    for i, j in T.Parallel(bm, bn):
                        ds[i, j] = T.if_then_else(
                            (blk * bm + i < m)
                            & (tile * bn + j < n)
                            & (
                                (
                                    (tile * bn + j < prefix)
                                    & (tile * bn + j <= offset + blk * bm + i)
                                )
                                | (
                                    (tile * bn + j >= prefix)
                                    & (tile * bn + j - prefix <= offset + blk * bm + i)
                                    & (
                                        tile * bn + j - prefix
                                        > offset + blk * bm + i - window
                                    )
                                )
                            ),
                            T.exp(prob[i, j] * scale - lse[batch, head, blk * bm + i])
                            * (dp[i, j] - delta[batch, head, blk * bm + i])
                            * scale,
                            0,
                        )
                    T.gemm(ds, ks, result, policy=T.GemmWarpPolicy.FullRow)
            T.copy(result, dq[batch, head, blk * bm : (blk + 1) * bm, :])

    return kernel


def attention_dkv(p):
    import tilelang.language as T

    b, h, kh, m, n, d, dv, dtype, causal, offset, scale = p[:11]
    window, prefix = p[11:13]
    split = p[13]
    output_heads = h if split else kh
    output_dtype = "float32" if split else dtype
    bn, bm = 32, 64

    @T.prim_func
    def kernel(
        q: T.Tensor((b, h, m, d), dtype),
        k: T.Tensor((b, kh, n, d), dtype),
        v: T.Tensor((b, kh, n, dv), dtype),
        dout: T.Tensor((b, h, m, dv), dtype),
        lse: T.Tensor((b, h, m), "float32"),
        delta: T.Tensor((b, h, m), "float32"),
        dk: T.Tensor((b, output_heads, n, d), output_dtype),
        dvout: T.Tensor((b, output_heads, n, dv), output_dtype),
    ):
        with T.Kernel(T.ceildiv(n, bn), output_heads, b, threads=128) as (
            blk,
            output_head,
            batch,
        ):
            kvhead = output_head // (h // kh) if split else output_head
            qs = T.alloc_shared((bm, d), dtype)
            ks = T.alloc_shared((bn, d), dtype)
            vs = T.alloc_shared((bn, dv), dtype)
            dos = T.alloc_shared((bm, dv), dtype)
            ds = T.alloc_shared((bn, bm), dtype)
            ps = T.alloc_shared((bn, bm), dtype)
            prob = T.alloc_fragment((bn, bm), "float32")
            dp = T.alloc_fragment((bn, bm), "float32")
            rk = T.alloc_fragment((bn, d), "float32")
            rv = T.alloc_fragment((bn, dv), "float32")
            T.copy(k[batch, kvhead, blk * bn : (blk + 1) * bn, :], ks)
            T.copy(v[batch, kvhead, blk * bn : (blk + 1) * bn, :], vs)
            T.clear(rk)
            T.clear(rv)
            for group in T.serial(1 if split else h // kh):
                head = output_head if split else kvhead * (h // kh) + group
                for tile in T.serial(T.ceildiv(m, bm)):
                    if ((blk * bn < prefix) & (offset + (tile + 1) * bm > blk * bn)) | (
                        ((blk + 1) * bn > prefix)
                        & (offset + (tile + 1) * bm > blk * bn - prefix)
                        & (offset + tile * bm < (blk + 1) * bn - prefix + window)
                    ):
                        T.copy(q[batch, head, tile * bm : (tile + 1) * bm, :], qs)
                        T.copy(dout[batch, head, tile * bm : (tile + 1) * bm, :], dos)
                        T.gemm(
                            ks,
                            qs,
                            prob,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        T.gemm(
                            vs,
                            dos,
                            dp,
                            transpose_B=True,
                            clear_accum=True,
                            policy=T.GemmWarpPolicy.FullRow,
                        )
                        for i, j in T.Parallel(bn, bm):
                            prob[i, j] = T.if_then_else(
                                (blk * bn + i < n)
                                & (tile * bm + j < m)
                                & (
                                    (
                                        (blk * bn + i < prefix)
                                        & (blk * bn + i <= offset + tile * bm + j)
                                    )
                                    | (
                                        (blk * bn + i >= prefix)
                                        & (
                                            blk * bn + i - prefix
                                            <= offset + tile * bm + j
                                        )
                                        & (
                                            blk * bn + i - prefix
                                            > offset + tile * bm + j - window
                                        )
                                    )
                                ),
                                T.exp(
                                    prob[i, j] * scale - lse[batch, head, tile * bm + j]
                                ),
                                0,
                            )
                            ds[i, j] = T.if_then_else(
                                tile * bm + j < m,
                                prob[i, j]
                                * (dp[i, j] - delta[batch, head, tile * bm + j])
                                * scale,
                                0,
                            )
                            ps[i, j] = prob[i, j]
                        T.gemm(ds, qs, rk, policy=T.GemmWarpPolicy.FullRow)
                        T.gemm(ps, dos, rv, policy=T.GemmWarpPolicy.FullRow)
            T.copy(rk, dk[batch, output_head, blk * bn : (blk + 1) * bn, :])
            T.copy(rv, dvout[batch, output_head, blk * bn : (blk + 1) * bn, :])

    return kernel


from .llt import attention_delta, attention_sum_heads
