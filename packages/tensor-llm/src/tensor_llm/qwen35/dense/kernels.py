"""Dense projections and hybrid attention with indirect request-owned state."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    if kind == "linear":
        k, o = p["k"], p["o"]
        m, n, bk = p.get("m", 16), p.get("n", 64), p.get("bk", 64)

        @T.prim_func
        def kernel(
            x: T.Tensor((r, k), "bfloat16"),
            w: T.Tensor((o, k), "bfloat16"),
            out: T.Tensor((r, o), "float32"),
        ):
            with T.Kernel(T.ceildiv(o, n), T.ceildiv(r, m), threads=128) as (bx, by):
                a = T.alloc_shared((m, bk), "bfloat16")
                b = T.alloc_shared((n, bk), "bfloat16")
                acc = T.alloc_fragment((m, n), "float32")
                T.clear(acc)
                for tile in T.Pipelined(
                    T.ceildiv(k, bk), num_stages=p.get("stages", 2)
                ):
                    for i, j in T.Parallel(m, bk):
                        a[i, j] = T.if_then_else(
                            (by * m + i < r) & (tile * bk + j < k),
                            x[T.min(by * m + i, r - 1), T.min(tile * bk + j, k - 1)],
                            0,
                        )
                    for i, j in T.Parallel(n, bk):
                        b[i, j] = T.if_then_else(
                            (bx * n + i < o) & (tile * bk + j < k),
                            w[T.min(bx * n + i, o - 1), T.min(tile * bk + j, k - 1)],
                            0,
                        )
                    T.gemm(a, b, acc, transpose_B=True)
                for i, j in T.Parallel(m, n):
                    if (by * m + i < r) & (bx * n + j < o):
                        out[by * m + i, bx * n + j] = acc[i, j]

        return kernel

    if kind == "gemv":
        k, o, cols = p["k"], p["o"], p.get("columns", 16)

        @T.prim_func
        def kernel(
            x: T.Tensor((r, k), "bfloat16"),
            w: T.Tensor((o, k), "bfloat16"),
            out: T.Tensor((r, o), "float32"),
        ):
            with T.Kernel(T.ceildiv(o, cols), r, threads=256) as (block, row):
                sums = T.alloc_fragment((cols, 32), "float32")
                total = T.alloc_fragment((cols,), "float32")
                T.clear(sums)
                for tile in T.serial(T.ceildiv(k, 32)):
                    for i, j in T.Parallel(cols, 32):
                        if (block * cols + i < o) & (tile * 32 + j < k):
                            sums[i, j] = T.ieee_fmaf(
                                T.cast(x[row, tile * 32 + j], "float32"),
                                T.cast(w[block * cols + i, tile * 32 + j], "float32"),
                                sums[i, j],
                            )
                T.reduce_sum(sums, total, dim=1)
                for i in T.Parallel(cols):
                    if block * cols + i < o:
                        out[row, block * cols + i] = total[i]

        return kernel

    slots, chunk, pool = p.get("slots", r), p.get("chunk", 1), p.get("pool", 256)
    rows = slots * chunk
    if kind == "gather_hidden":
        c = p["c"]

        @T.prim_func
        def kernel(
            saved: T.Tensor((pool, c), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            mode: T.Tensor((1,), "int32"),
            out: T.Tensor((rows, c), "bfloat16"),
        ):
            with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
                if (mode[0] != 0) & (mapping[batch] >= 0):
                    for j in T.Parallel(256):
                        col = tile * 256 + j
                        if col < c:
                            out[batch * chunk, col] = saved[mapping[batch], col]

        return kernel
    if kind == "store_hidden":
        c = p["c"]

        @T.prim_func
        def kernel(
            x: T.Tensor((slots, c), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            saved: T.Tensor((pool, c), "bfloat16"),
        ):
            with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
                if (mapping[batch] >= 0) & (lengths[batch] > 0):
                    for j in T.Parallel(256):
                        col = tile * 256 + j
                        if col < c:
                            saved[mapping[batch], col] = x[batch, col]

        return kernel
    if kind == "controls":

        @T.prim_func
        def kernel(
            position: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            flat_position: T.Tensor((rows,), "int32"),
            active: T.Tensor((rows,), "int32"),
        ):
            with T.Kernel(T.ceildiv(rows, 256), threads=256) as block:
                for i in T.Parallel(256):
                    row = block * 256 + i
                    if row < rows:
                        flat_position[row] = position[row // chunk] + row % chunk
                        active[row] = T.cast(
                            row % chunk < lengths[row // chunk], "int32"
                        )

        return kernel

    if kind == "gdn_conv":
        c, projection = p["channels"], p["projection"]

        @T.prim_func
        def kernel(
            x: T.Tensor((rows, projection), "float32"),
            w: T.Tensor((c, 1, 4), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            state: T.Tensor((pool, c, 3), "float32"),
            out: T.Tensor((rows, c), "float32"),
        ):
            with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
                request = mapping[batch]
                if request >= 0:
                    for j in T.Parallel(256):
                        col = tile * 256 + j
                        if col < c:
                            h0 = T.alloc_var("float32")
                            h1 = T.alloc_var("float32")
                            h2 = T.alloc_var("float32")
                            h0 = state[request, col, 0]
                            h1 = state[request, col, 1]
                            h2 = state[request, col, 2]
                            for t in T.serial(chunk):
                                if t < lengths[batch]:
                                    current = T.cast(
                                        T.cast(x[batch * chunk + t, col], "bfloat16"),
                                        "float32",
                                    )
                                    value = h0 * T.cast(
                                        w[col, 0, 0], "float32"
                                    ) + h1 * T.cast(w[col, 0, 1], "float32")
                                    value += h2 * T.cast(
                                        w[col, 0, 2], "float32"
                                    ) + current * T.cast(w[col, 0, 3], "float32")
                                    out[batch * chunk + t, col] = T.cast(
                                        value / (1 + T.exp(-value)), "bfloat16"
                                    )
                                    h0 = h1
                                    h1 = h2
                                    h2 = current
                                else:
                                    out[batch * chunk + t, col] = 0
                            if p.get("commit", True):
                                state[request, col, 0] = h0
                                state[request, col, 1] = h1
                                state[request, col, 2] = h2

        return kernel

    if kind == "gdn_prepare":
        heads, kh, d, channels, projection = (
            p[n] for n in ("heads", "key_heads", "d", "channels", "projection")
        )
        value_width = heads * d

        @T.prim_func
        def kernel(
            conv: T.Tensor((rows, channels), "float32"),
            proj: T.Tensor((rows, projection), "float32"),
            dt: T.Tensor((heads,), "bfloat16"),
            A: T.Tensor((heads,), "float32"),
            q: T.Tensor((rows, heads, d), "float32"),
            k: T.Tensor((rows, heads, d), "float32"),
            v: T.Tensor((rows, heads, d), "float32"),
            g: T.Tensor((rows, heads), "float32"),
            beta: T.Tensor((rows, heads), "float32"),
        ):
            with T.Kernel(rows, heads, threads=128) as (row, head):
                qq = T.alloc_fragment((d,), "float32")
                kk = T.alloc_fragment((d,), "float32")
                qs = T.alloc_fragment((d,), "float32")
                ks = T.alloc_fragment((d,), "float32")
                qt = T.alloc_fragment((1,), "float32")
                kt = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    qq[j] = conv[row, head // (heads // kh) * d + j]
                    kk[j] = conv[row, kh * d + head // (heads // kh) * d + j]
                    qs[j] = qq[j] * qq[j]
                    ks[j] = kk[j] * kk[j]
                T.reduce_sum(qs, qt, dim=0)
                T.reduce_sum(ks, kt, dim=0)
                for j in T.Parallel(d):
                    q[row, head, j] = qq[j] * T.rsqrt(qt[0] + T.float32(1e-6)) * d**-0.5
                    k[row, head, j] = kk[j] * T.rsqrt(kt[0] + T.float32(1e-6))
                    v[row, head, j] = conv[row, 2 * kh * d + head * d + j]
                for j in T.Parallel(1):
                    a = T.cast(
                        T.cast(proj[row, channels + value_width + head], "bfloat16"),
                        "float32",
                    ) + T.cast(dt[head], "float32")
                    g[row, head] = -T.exp(A[head]) * (
                        T.max(a, 0) + T.log(1 + T.exp(-T.abs(a)))
                    )
                    beta[row, head] = 1 / (
                        1
                        + T.exp(
                            -T.cast(
                                T.cast(
                                    proj[row, channels + value_width + heads + head],
                                    "bfloat16",
                                ),
                                "float32",
                            )
                        )
                    )

        return kernel

    if kind == "gdn_scan":
        heads, d, tile = p["heads"], p["d"], p.get("tile", 32)

        @T.prim_func
        def kernel(
            q: T.Tensor((rows, heads, d), "float32"),
            k: T.Tensor((rows, heads, d), "float32"),
            v: T.Tensor((rows, heads, d), "float32"),
            g: T.Tensor((rows, heads), "float32"),
            beta: T.Tensor((rows, heads), "float32"),
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            state: T.Tensor((pool, heads, d, d), "float32"),
            out: T.Tensor((rows, heads, d), "float32"),
        ):
            with T.Kernel(slots, heads, d // tile, threads=128) as (batch, head, part):
                request = mapping[batch]
                if request >= 0:
                    matrix = T.alloc_fragment((tile, d), "float32")
                    prediction = T.alloc_fragment((tile, d), "float32")
                    inner = T.alloc_fragment((tile,), "float32")
                    result = T.alloc_fragment((tile,), "float32")
                    qq = T.alloc_shared((d,), "float32")
                    kk = T.alloc_shared((d,), "float32")
                    T.copy(
                        state[request, head, part * tile : part * tile + tile, :],
                        matrix,
                    )
                    for t in T.serial(chunk):
                        row = batch * chunk + t
                        if t < lengths[batch]:
                            for j in T.Parallel(d):
                                qq[j] = q[row, head, j]
                                kk[j] = k[row, head, j]
                            for i, j in T.Parallel(tile, d):
                                matrix[i, j] *= T.exp(g[row, head])
                                prediction[i, j] = matrix[i, j] * kk[j]
                            T.reduce_sum(prediction, inner, dim=1)
                            for i, j in T.Parallel(tile, d):
                                matrix[i, j] += (
                                    kk[j]
                                    * beta[row, head]
                                    * (v[row, head, part * tile + i] - inner[i])
                                )
                                prediction[i, j] = matrix[i, j] * qq[j]
                            T.reduce_sum(prediction, result, dim=1)
                            for i in T.Parallel(tile):
                                out[row, head, part * tile + i] = result[i]
                        else:
                            for i in T.Parallel(tile):
                                out[row, head, part * tile + i] = 0
                    if p.get("commit", True):
                        T.copy(
                            matrix,
                            state[request, head, part * tile : part * tile + tile, :],
                        )

        return kernel

    if kind == "gdn_norm":
        heads, d, channels, projection, eps = (
            p[n] for n in ("heads", "d", "channels", "projection", "eps")
        )

        @T.prim_func
        def kernel(
            x: T.Tensor((rows, heads, d), "float32"),
            proj: T.Tensor((rows, projection), "float32"),
            w: T.Tensor((d,), "float32"),
            out: T.Tensor((rows, heads * d), "bfloat16"),
        ):
            with T.Kernel(rows, heads, threads=128) as (row, head):
                values = T.alloc_fragment((d,), "float32")
                squares = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(d):
                    values[j] = T.cast(T.cast(x[row, head, j], "bfloat16"), "float32")
                    squares[j] = values[j] * values[j]
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    gate = T.cast(
                        T.cast(proj[row, channels + head * d + j], "bfloat16"),
                        "float32",
                    )
                    out[row, head * d + j] = (
                        values[j]
                        * T.rsqrt(total[0] / d + eps)
                        * w[j]
                        * gate
                        / (1 + T.exp(-gate))
                    )

        return kernel

    if kind == "swiglu":
        c = p["c"]

        @T.prim_func
        def kernel(
            x: T.Tensor((rows, 2 * c), "float32"), out: T.Tensor((rows, c), "bfloat16")
        ):
            with T.Kernel(rows, T.ceildiv(c, 256), threads=256) as (row, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        gate = T.cast(T.cast(x[row, col], "bfloat16"), "float32")
                        up = T.cast(T.cast(x[row, c + col], "bfloat16"), "float32")
                        out[row, col] = gate / (1 + T.exp(-gate)) * up

        return kernel

    if kind == "last_rows":
        c = p["c"]

        @T.prim_func
        def kernel(
            x: T.Tensor((rows, c), "bfloat16"),
            lengths: T.Tensor((slots,), "int32"),
            out: T.Tensor((slots, c), "bfloat16"),
        ):
            with T.Kernel(slots, T.ceildiv(c, 256), threads=256) as (batch, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        out[batch, col] = T.if_then_else(
                            lengths[batch] > 0,
                            x[batch * chunk + T.max(lengths[batch] - 1, 0), col],
                            0,
                        )

        return kernel

    if kind == "attention_qkv":
        heads, kh, d, cap, eps, theta = (
            p[n] for n in ("heads", "kv_heads", "d", "capacity", "eps", "theta")
        )
        projection = 2 * heads * d + 2 * kh * d

        @T.prim_func
        def kernel(
            proj: T.Tensor((rows, projection), "float32"),
            qw: T.Tensor((d,), "bfloat16"),
            kw: T.Tensor((d,), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            positions: T.Tensor((rows,), "int32"),
            active: T.Tensor((rows,), "int32"),
            kc: T.Tensor((pool, kh, cap, d), "bfloat16"),
            vc: T.Tensor((pool, kh, cap, d), "bfloat16"),
            q: T.Tensor((rows, heads, d), "bfloat16"),
        ):
            with T.Kernel(rows, heads + kh, threads=256) as (row, head):
                if active[row] != 0:
                    values = T.alloc_shared((d,), "bfloat16")
                    squares = T.alloc_fragment((d,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    for j in T.Parallel(d):
                        raw = T.cast(
                            T.cast(
                                proj[
                                    row,
                                    T.if_then_else(
                                        head < heads,
                                        head * 2 * d,
                                        2 * heads * d + (head - heads) * d,
                                    )
                                    + j,
                                ],
                                "bfloat16",
                            ),
                            "float32",
                        )
                        squares[j] = raw * raw
                    T.reduce_sum(squares, total, dim=0)
                    for j in T.Parallel(d):
                        raw = T.cast(
                            T.cast(
                                proj[
                                    row,
                                    T.if_then_else(
                                        head < heads,
                                        head * 2 * d,
                                        2 * heads * d + (head - heads) * d,
                                    )
                                    + j,
                                ],
                                "bfloat16",
                            ),
                            "float32",
                        )
                        weight = T.if_then_else(head < heads, qw[j], kw[j])
                        values[j] = (
                            raw
                            * T.rsqrt(total[0] / d + eps)
                            * (1 + T.cast(weight, "float32"))
                        )
                    for j in T.Parallel(d):
                        value = T.alloc_var("float32")
                        value = T.cast(values[j], "float32")
                        if j < 64:
                            angle = positions[row] * T.exp(
                                T.cast(-2 * (j % 32), "float32")
                                / 64
                                * T.log(T.float32(theta))
                            )
                            pair = T.cast(values[(j + 32) % 64], "float32")
                            value = value * T.cos(angle) + T.if_then_else(
                                j < 32, -pair, pair
                            ) * T.sin(angle)
                        if head < heads:
                            q[row, head, j] = value
                        else:
                            kc[
                                mapping[row // chunk], head - heads, positions[row], j
                            ] = value
                            vc[
                                mapping[row // chunk], head - heads, positions[row], j
                            ] = T.cast(
                                proj[
                                    row, 2 * heads * d + kh * d + (head - heads) * d + j
                                ],
                                "bfloat16",
                            )

        return kernel

    if kind == "attention":
        heads, kh, d, cap, eps = (
            p[n] for n in ("heads", "kv_heads", "d", "capacity", "eps")
        )
        group = heads // kh
        qt = min(chunk, 4)
        qm = max(16, qt * group)
        projection = 2 * heads * d + 2 * kh * d

        @T.prim_func
        def kernel(
            q: T.Tensor((rows, heads, d), "bfloat16"),
            kc: T.Tensor((pool, kh, cap, d), "bfloat16"),
            vc: T.Tensor((pool, kh, cap, d), "bfloat16"),
            proj: T.Tensor((rows, projection), "float32"),
            mapping: T.Tensor((slots,), "int32"),
            positions: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            out: T.Tensor((rows, heads * d), "bfloat16"),
        ):
            with T.Kernel(slots, kh, T.ceildiv(chunk, qt), threads=128) as (
                batch,
                head,
                qblock,
            ):
                request = mapping[batch]
                if (request >= 0) & (qblock * qt < lengths[batch]):
                    query = T.alloc_shared((qm, d), "bfloat16")
                    key = T.alloc_shared((64, d), "bfloat16")
                    value = T.alloc_shared((64, d), "bfloat16")
                    prob = T.alloc_shared((qm, 64), "bfloat16")
                    scores = T.alloc_fragment((qm, 64), "float32")
                    result = T.alloc_fragment((qm, d), "float32")
                    maximum = T.alloc_fragment((qm,), "float32")
                    previous = T.alloc_fragment((qm,), "float32")
                    factor = T.alloc_fragment((qm,), "float32")
                    normalizer = T.alloc_fragment((qm,), "float32")
                    total = T.alloc_fragment((qm,), "float32")
                    T.clear(result)
                    T.clear(normalizer)
                    T.fill(maximum, -T.infinity("float32"))
                    for i, j in T.Parallel(qm, d):
                        query[i, j] = T.if_then_else(
                            (i < qt * group)
                            & (qblock * qt + i // group < lengths[batch]),
                            q[
                                batch * chunk
                                + T.min(qblock * qt + i // group, chunk - 1),
                                head * group + i % group,
                                j,
                            ],
                            0,
                        )
                    count = positions[batch] + T.min((qblock + 1) * qt, lengths[batch])
                    for tile in T.serial(T.ceildiv(count, 64)):
                        for i, j in T.Parallel(64, d):
                            pos = tile * 64 + i
                            key[i, j] = T.if_then_else(
                                pos < count,
                                kc[request, head, T.min(pos, cap - 1), j],
                                0,
                            )
                            value[i, j] = T.if_then_else(
                                pos < count,
                                vc[request, head, T.min(pos, cap - 1), j],
                                0,
                            )
                        T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                        T.copy(maximum, previous)
                        for i, j in T.Parallel(qm, 64):
                            scores[i, j] = T.if_then_else(
                                (tile * 64 + j < count)
                                & (
                                    tile * 64 + j
                                    <= positions[batch] + qblock * qt + i // group
                                ),
                                scores[i, j] * d**-0.5,
                                -T.infinity("float32"),
                            )
                        T.reduce_max(scores, maximum, dim=1, clear=False)
                        for i in T.Parallel(qm):
                            factor[i] = T.if_then_else(
                                previous[i] > -T.infinity("float32"),
                                T.exp(previous[i] - maximum[i]),
                                0,
                            )
                        for i, j in T.Parallel(qm, 64):
                            scores[i, j] = T.if_then_else(
                                scores[i, j] > -T.infinity("float32"),
                                T.exp(scores[i, j] - maximum[i]),
                                0,
                            )
                        T.reduce_sum(scores, total, dim=1)
                        for i in T.Parallel(qm):
                            normalizer[i] = normalizer[i] * factor[i] + total[i]
                        for i, j in T.Parallel(qm, d):
                            result[i, j] *= factor[i]
                        T.copy(scores, prob)
                        T.gemm(prob, value, result)
                    for i, j in T.Parallel(qt * group, d):
                        row = batch * chunk + qblock * qt + i // group
                        if qblock * qt + i // group < lengths[batch]:
                            gate = T.cast(
                                T.cast(
                                    proj[
                                        row, (head * group + i % group) * 2 * d + d + j
                                    ],
                                    "bfloat16",
                                ),
                                "float32",
                            )
                            value_out = T.cast(
                                T.cast(result[i, j] / normalizer[i], "bfloat16"),
                                "float32",
                            )
                            out[row, (head * group + i % group) * d + j] = value_out / (
                                1 + T.exp(-gate)
                            )

        return kernel

    raise ValueError("unsupported dense Qwen primitive: " + kind)
