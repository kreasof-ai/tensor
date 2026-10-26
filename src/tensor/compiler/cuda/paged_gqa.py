"""Paged FP8-cache GQA decode/verify with fused boundaries and split softmax.

These factories target head dimension 64, eight query heads per KV head and
query lengths one or eight. Compile with automatic thread synchronization and
ordinary MMA; they use pointer/scalar launch arguments, including scattered
page tables and runtime request lengths. Consumers own the shape-based split
policy, scratch tensors and the final merge launch.
"""


def make_prepare(tokens, hq, hkv, rows, theta, qfactor, query_dtype="bfloat16"):
    import tilelang.language as T

    width = (2 * hq + 2 * hkv) * 64

    @T.prim_func
    def gqa_prepare(
        fused: T.Tensor((tokens, width), "bfloat16"),
        positions: T.Tensor((tokens,), "int64"),
        locations: T.Tensor((tokens,), "int64"),
        ks: T.Tensor((1,), "float32"),
        vs: T.Tensor((1,), "float32"),
        kc: T.Tensor((rows, hkv, 64), "float8_e4m3"),
        vc: T.Tensor((rows, hkv, 64), "float8_e4m3"),
        query: T.Tensor((tokens, hq, 64), query_dtype),
    ):
        with T.Kernel(tokens, hq, threads=128) as (token, head):
            q = T.alloc_fragment((64,), "float32")
            sq = T.alloc_fragment((64,), "float32")
            total = T.alloc_fragment((1,), "float32")
            for d in T.Parallel(64):
                q[d] = T.cast(fused[token, head * 64 + d], "float32")
                sq[d] = q[d] * q[d]
            T.reduce_sum(sq, total, dim=0)
            for d in T.Parallel(64):
                q[d] *= T.rsqrt(total[0] / 64 + 1e-5) * qfactor
            if theta > 0:
                for d in T.Parallel(64):
                    angle = T.cast(positions[token], "float32") * T.pow(
                        theta, -T.cast(d // 2 * 2, "float32") / 64
                    )
                    peer = (
                        T.cast(fused[token, head * 64 + (d ^ 1)], "float32")
                        * T.rsqrt(total[0] / 64 + 1e-5)
                        * qfactor
                    )
                    query[token, head, d] = q[d] * T.cos(angle) + T.if_then_else(
                        d % 2 == 0, -peer, peer
                    ) * T.sin(angle)
            else:
                for d in T.Parallel(64):
                    query[token, head, d] = q[d]
            if head < hkv:
                key = T.alloc_fragment((64,), "float32")
                squares = T.alloc_fragment((64,), "float32")
                norm = T.alloc_fragment((1,), "float32")
                for d in T.Parallel(64):
                    key[d] = T.cast(fused[token, hq * 64 + head * 64 + d], "float32")
                    squares[d] = key[d] * key[d]
                T.reduce_sum(squares, norm, dim=0)
                for d in T.Parallel(64):
                    value = T.alloc_var("float32")
                    value = key[d] * T.rsqrt(norm[0] / 64 + 1e-5)
                    if theta > 0:
                        angle = T.cast(positions[token], "float32") * T.pow(
                            theta, -T.cast(d // 2 * 2, "float32") / 64
                        )
                        peer = T.cast(
                            fused[token, hq * 64 + head * 64 + (d ^ 1)], "float32"
                        ) * T.rsqrt(norm[0] / 64 + 1e-5)
                        value = value * T.cos(angle) + T.if_then_else(
                            d % 2 == 0, -peer, peer
                        ) * T.sin(angle)
                    kc[locations[token], head, d] = value / ks[0]
                    vc[locations[token], head, d] = (
                        T.cast(fused[token, (hq + hkv) * 64 + head * 64 + d], "float32") / vs[0]
                    )

    return gqa_prepare


def make_partial(batch, length, hq, hkv, pages, rows, page, window, splits, bk=128, threads=128):
    import tilelang.language as T

    group = hq // hkv
    qm = max(16, length * group)

    @T.prim_func
    def gqa_partial(
        query: T.Tensor((batch * length, hq, 64), "bfloat16"),
        kc: T.Tensor((rows, hkv, 64), "float8_e4m3"),
        vc: T.Tensor((rows, hkv, 64), "float8_e4m3"),
        ks: T.Tensor((1,), "float32"),
        vs: T.Tensor((1,), "float32"),
        table: T.Tensor((batch, pages), "int32"),
        lengths: T.Tensor((batch,), "int32"),
        out: T.Tensor((batch, hkv, splits, qm, 64), "float32"),
        stats: T.Tensor((batch, hkv, splits, qm, 2), "float32"),
    ):
        with T.Kernel(batch * hkv, splits, threads=threads) as (index, part):
            request, head = index // hkv, index % hkv
            seq = lengths[request]
            start = T.max(seq - window + 1, 0) if window else 0
            stop = seq + length
            span = T.ceildiv(T.ceildiv(stop - start, bk), splits)
            first = start + part * span * bk
            count = T.max(T.min(span, T.ceildiv(stop - first, bk)), 0)
            q = T.alloc_shared((qm, 64), "bfloat16")
            key = T.alloc_shared((bk, 64), "bfloat16")
            value = T.alloc_shared((bk, 64), "bfloat16")
            prob = T.alloc_shared((qm, bk), "bfloat16")
            score = T.alloc_fragment((qm, bk), "float32")
            acc = T.alloc_fragment((qm, 64), "float32")
            maximum = T.alloc_fragment((qm,), "float32")
            previous = T.alloc_fragment((qm,), "float32")
            alpha = T.alloc_fragment((qm,), "float32")
            denom = T.alloc_fragment((qm,), "float32")
            added = T.alloc_fragment((qm,), "float32")
            for i, d in T.Parallel(qm, 64):
                q[i, d] = T.if_then_else(
                    i < length * group,
                    query[
                        request * length + T.min(i // group, length - 1),
                        head * group + i % group,
                        d,
                    ],
                    0,
                )
            T.clear(acc)
            T.clear(denom)
            T.fill(maximum, -T.infinity("float32"))
            for tile in T.Pipelined(count, num_stages=2):
                for j, d in T.Parallel(bk, 64):
                    position = first + tile * bk + j
                    physical = (
                        table[request, T.min(position // page, pages - 1)] * page + position % page
                    )
                    key[j, d] = T.if_then_else(
                        position < stop,
                        T.cast(kc[T.max(physical, 0), head, d], "float32") * ks[0],
                        0,
                    )
                    value[j, d] = T.if_then_else(
                        position < stop,
                        T.cast(vc[T.max(physical, 0), head, d], "float32") * vs[0],
                        0,
                    )
                T.gemm(
                    q,
                    key,
                    score,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                T.copy(maximum, previous)
                for i, j in T.Parallel(qm, bk):
                    position = first + tile * bk + j
                    visible = (
                        i < length * group
                        and position <= seq + i // group
                        and (position > seq + i // group - window if window else True)
                    )
                    score[i, j] = T.if_then_else(
                        visible, score[i, j] * (0.125 * 1.4426950408889634), -T.infinity("float32")
                    )
                T.reduce_max(score, maximum, dim=1, clear=False)
                for i in T.Parallel(qm):
                    alpha[i] = T.if_then_else(
                        T.isfinite(previous[i]), T.exp2(previous[i] - maximum[i]), 0
                    )
                for i, j in T.Parallel(qm, bk):
                    score[i, j] = T.if_then_else(
                        T.isfinite(score[i, j]), T.exp2(score[i, j] - maximum[i]), 0
                    )
                T.reduce_sum(score, added, dim=1)
                for i in T.Parallel(qm):
                    denom[i] = denom[i] * alpha[i] + added[i]
                for i, d in T.Parallel(qm, 64):
                    acc[i, d] *= alpha[i]
                T.copy(score, prob)
                T.gemm(prob, value, acc, policy=T.GemmWarpPolicy.FullRow)
            for i, d in T.Parallel(qm, 64):
                out[request, head, part, i, d] = acc[i, d]
            for i in T.Parallel(qm):
                stats[request, head, part, i, 0] = maximum[i]
                stats[request, head, part, i, 1] = denom[i]

    return gqa_partial


def make_merge(batch, length, hq, hkv, splits):
    import tilelang.language as T

    group, width = hq // hkv, (2 * hq + 2 * hkv) * 64
    qm = max(16, length * group)

    @T.prim_func
    def gqa_merge(
        parts: T.Tensor((batch, hkv, splits, qm, 64), "float32"),
        stats: T.Tensor((batch, hkv, splits, qm, 2), "float32"),
        fused: T.Tensor((batch * length, width), "bfloat16"),
        out: T.Tensor((batch, length, hq * 64), "bfloat16"),
    ):
        with T.Kernel(batch * length, hq, threads=128) as (token, head):
            request, row = token // length, token % length * group + head % group
            kv = head // group
            maximum = T.alloc_fragment((splits,), "float32")
            largest = T.alloc_fragment((1,), "float32")
            weights = T.alloc_fragment((splits,), "float32")
            den = T.alloc_fragment((1,), "float32")
            acc = T.alloc_fragment((64,), "float32")
            square = T.alloc_fragment((64,), "float32")
            norm = T.alloc_fragment((1,), "float32")
            for s in T.Parallel(splits):
                maximum[s] = stats[request, kv, s, row, 0]
            T.reduce_max(maximum, largest, dim=0)
            for s in T.Parallel(splits):
                weights[s] = T.if_then_else(
                    T.isfinite(maximum[s]),
                    T.exp2(maximum[s] - largest[0]) * stats[request, kv, s, row, 1],
                    0,
                )
            T.reduce_sum(weights, den, dim=0)
            T.clear(acc)
            for s in T.serial(splits):
                for d in T.Parallel(64):
                    acc[d] += (
                        T.if_then_else(
                            T.isfinite(stats[request, kv, s, row, 0]),
                            T.exp2(stats[request, kv, s, row, 0] - largest[0]),
                            0,
                        )
                        * parts[request, kv, s, row, d]
                    )
            for d in T.Parallel(64):
                acc[d] /= den[0]
                square[d] = acc[d] * acc[d]
            T.reduce_sum(square, norm, dim=0)
            for d in T.Parallel(64):
                gate = T.cast(fused[token, (hq + 2 * hkv) * 64 + head * 64 + d], "float32")
                out[request, token % length, head * 64 + d] = (
                    acc[d] * T.rsqrt(norm[0] / 64 + 1e-5) / (1 + T.exp(-gate))
                )

    return gqa_merge


def make_gate(tokens, hq, hkv):
    """Normalize and gate four heads per CTA, with warp-local reductions."""
    import tilelang.language as T

    width = (2 * hq + 2 * hkv) * 64

    @T.prim_func
    def gqa_gate(
        attention: T.Tensor((tokens, hq, 64), "bfloat16"),
        fused: T.Tensor((tokens, width), "bfloat16"),
        out: T.Tensor((tokens, hq * 64), "bfloat16"),
    ):
        with T.Kernel(tokens, T.ceildiv(hq, 4), threads=128) as (token, block):
            values = T.alloc_fragment((4, 64), "float32")
            squares = T.alloc_fragment((4, 64), "float32")
            norm = T.alloc_fragment((4,), "float32")
            T.annotate_layout(
                {
                    values: T.Fragment(
                        (4, 64),
                        forward_thread_fn=lambda i, j: i * 32 + j % 32,
                        forward_index_fn=lambda i, j: j // 32,
                    ),
                    squares: T.Fragment(
                        (4, 64),
                        forward_thread_fn=lambda i, j: i * 32 + j % 32,
                        forward_index_fn=lambda i, j: j // 32,
                    ),
                }
            )
            for i, d in T.Parallel(4, 64):
                values[i, d] = T.if_then_else(
                    block * 4 + i < hq,
                    T.cast(attention[token, T.min(block * 4 + i, hq - 1), d], "float32"),
                    0,
                )
                squares[i, d] = values[i, d] * values[i, d]
            T.reduce_sum(squares, norm, dim=1)
            for i, d in T.Parallel(4, 64):
                if block * 4 + i < hq:
                    gate = T.cast(
                        fused[token, (hq + 2 * hkv) * 64 + (block * 4 + i) * 64 + d], "float32"
                    )
                    out[token, (block * 4 + i) * 64 + d] = (
                        values[i, d] * T.rsqrt(norm[i] / 64 + 1e-5) / (1 + T.exp(-gate))
                    )

    return gqa_gate
