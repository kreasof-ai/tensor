"""Block-scaled E4M3 KV with indirect ownership and split causal attention."""


def make_kernel(kind, p):
    import tilelang.language as T

    slots, chunk, pool = (p[name] for name in ("slots", "chunk", "pool"))
    rows = slots * chunk
    heads, kh, d, cap, eps = (
        p[name] for name in ("heads", "kv_heads", "d", "capacity", "eps")
    )
    projection = 2 * heads * d + 2 * kh * d
    if kind == "attention_qkv":
        theta = p["theta"]

        @T.prim_func
        def kernel(
            proj: T.Tensor((rows, projection), "float32"),
            qw: T.Tensor((d,), "bfloat16"),
            kw: T.Tensor((d,), "bfloat16"),
            mapping: T.Tensor((slots,), "int32"),
            positions: T.Tensor((rows,), "int32"),
            active: T.Tensor((rows,), "int32"),
            kc: T.Tensor((pool, kh, cap, d), "uint8"),
            vc: T.Tensor((pool, kh, cap, d), "uint8"),
            ks: T.Tensor((pool, kh, cap, 2), "float32"),
            vs: T.Tensor((pool, kh, cap, 2), "float32"),
            q: T.Tensor((rows, heads, d), "bfloat16"),
        ):
            with T.Kernel(rows, heads + kh, threads=256) as (row, head):
                if active[row] != 0:
                    normal = T.alloc_shared((d,), "bfloat16")
                    rotated = T.alloc_shared((d,), "bfloat16")
                    squares = T.alloc_fragment((d,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    for j in T.Parallel(d):
                        offset = T.if_then_else(
                            head < heads,
                            head * 2 * d,
                            2 * heads * d + (head - heads) * d,
                        )
                        value = T.cast(
                            T.cast(proj[row, offset + j], "bfloat16"), "float32"
                        )
                        squares[j] = value * value
                    T.reduce_sum(squares, total, dim=0)
                    for j in T.Parallel(d):
                        offset = T.if_then_else(
                            head < heads,
                            head * 2 * d,
                            2 * heads * d + (head - heads) * d,
                        )
                        value = T.cast(
                            T.cast(proj[row, offset + j], "bfloat16"), "float32"
                        )
                        weight = T.if_then_else(head < heads, qw[j], kw[j])
                        normal[j] = (
                            value
                            * T.rsqrt(total[0] / d + eps)
                            * (1 + T.cast(weight, "float32"))
                        )
                    for j in T.Parallel(d):
                        value = T.alloc_var("float32")
                        value = T.cast(normal[j], "float32")
                        if j < 64:
                            angle = positions[row] * T.exp(
                                T.cast(-2 * (j % 32), "float32")
                                / 64
                                * T.log(T.float32(theta))
                            )
                            pair = T.cast(normal[(j + 32) % 64], "float32")
                            value = value * T.cos(angle) + T.if_then_else(
                                j < 32, -pair, pair
                            ) * T.sin(angle)
                        rotated[j] = value
                    if head < heads:
                        for j in T.Parallel(d):
                            q[row, head, j] = rotated[j]
                    else:
                        key_abs = T.alloc_fragment((2, 128), "float32")
                        value_abs = T.alloc_fragment((2, 128), "float32")
                        key_max = T.alloc_fragment((2,), "float32")
                        value_max = T.alloc_fragment((2,), "float32")
                        key_scale = T.alloc_shared((2,), "float32")
                        value_scale = T.alloc_shared((2,), "float32")
                        for block, j in T.Parallel(2, 128):
                            key_abs[block, j] = T.abs(
                                T.cast(rotated[block * 128 + j], "float32")
                            )
                            value_abs[block, j] = T.abs(
                                T.cast(
                                    T.cast(
                                        proj[
                                            row,
                                            2 * heads * d
                                            + kh * d
                                            + (head - heads) * d
                                            + block * 128
                                            + j,
                                        ],
                                        "bfloat16",
                                    ),
                                    "float32",
                                )
                            )
                        T.reduce_max(key_abs, key_max, dim=1)
                        T.reduce_max(value_abs, value_max, dim=1)
                        for block in T.Parallel(2):
                            key_scale[block] = (
                                T.max(key_max[block], T.float32(1e-12)) / 448
                            )
                            value_scale[block] = (
                                T.max(value_max[block], T.float32(1e-12)) / 448
                            )
                            ks[
                                mapping[row // chunk],
                                head - heads,
                                positions[row],
                                block,
                            ] = key_scale[block]
                            vs[
                                mapping[row // chunk],
                                head - heads,
                                positions[row],
                                block,
                            ] = value_scale[block]
                        T.sync_threads()
                        for j in T.Parallel(d):
                            key = T.cast(rotated[j], "float32") / key_scale[j // 128]
                            value = (
                                T.cast(
                                    T.cast(
                                        proj[
                                            row,
                                            2 * heads * d
                                            + kh * d
                                            + (head - heads) * d
                                            + j,
                                        ],
                                        "bfloat16",
                                    ),
                                    "float32",
                                )
                                / value_scale[j // 128]
                            )
                            kc[
                                mapping[row // chunk], head - heads, positions[row], j
                            ] = T.cast(
                                T.call_extern("uint32", "tensor_encode_e4m3", key),
                                "uint8",
                            )
                            vc[
                                mapping[row // chunk], head - heads, positions[row], j
                            ] = T.cast(
                                T.call_extern("uint32", "tensor_encode_e4m3", value),
                                "uint8",
                            )

        return kernel
    group = heads // kh
    qt = min(chunk, 4)
    qm = max(16, qt * group)
    blocks = T.ceildiv(chunk, qt)
    splits = p["splits"]
    shape = (slots, kh, blocks, splits, qt * group)
    if kind == "attention":

        @T.prim_func
        def kernel(
            q: T.Tensor((rows, heads, d), "bfloat16"),
            kc: T.Tensor((pool, kh, cap, d), "uint8"),
            vc: T.Tensor((pool, kh, cap, d), "uint8"),
            ks: T.Tensor((pool, kh, cap, 2), "float32"),
            vs: T.Tensor((pool, kh, cap, 2), "float32"),
            mapping: T.Tensor((slots,), "int32"),
            positions: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            out: T.Tensor(shape + (d,), "float32"),
            stats: T.Tensor(shape + (2,), "float32"),
        ):
            with T.Kernel(slots, kh, blocks * splits, threads=128) as (
                batch,
                head,
                which,
            ):
                qblock = which // splits
                part = which % splits
                request = mapping[batch]
                query = T.alloc_shared((qm, d), "bfloat16")
                key = T.alloc_shared((64, d), "bfloat16")
                value = T.alloc_shared((64, d), "bfloat16")
                encoded = T.alloc_shared((64, d), "float8_e4m3fn")
                scales = T.alloc_shared((64, 2), "float32")
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
                count = positions[batch] + T.min((qblock + 1) * qt, lengths[batch])
                tiles = T.ceildiv(T.ceildiv(count, 64), splits)
                if (
                    (request >= 0)
                    & (qblock * qt < lengths[batch])
                    & (part * tiles * 64 < count)
                ):
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
                    for tile in T.serial(tiles):
                        start = (part * tiles + tile) * 64
                        if start < cap:
                            keys = T.view(kc, dtype="float8_e4m3fn")
                            values = T.view(vc, dtype="float8_e4m3fn")
                            T.copy(keys[request, head, start : start + 64, :], encoded)
                            T.copy(ks[request, head, start : start + 64, :], scales)
                            for i, j in T.Parallel(64, d):
                                key[i, j] = T.if_then_else(
                                    start + i < count,
                                    T.cast(encoded[i, j], "float32")
                                    * scales[i, j // 128],
                                    0,
                                )
                            T.copy(
                                values[request, head, start : start + 64, :], encoded
                            )
                            T.copy(vs[request, head, start : start + 64, :], scales)
                            for i, j in T.Parallel(64, d):
                                value[i, j] = T.if_then_else(
                                    start + i < count,
                                    T.cast(encoded[i, j], "float32")
                                    * scales[i, j // 128],
                                    0,
                                )
                        else:
                            T.clear(key)
                            T.clear(value)
                        T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                        T.copy(maximum, previous)
                        for i, j in T.Parallel(qm, 64):
                            scores[i, j] = T.if_then_else(
                                (start + j < count)
                                & (
                                    start + j
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
                    out[batch, head, qblock, part, i, j] = result[i, j]
                for i in T.Parallel(qt * group):
                    stats[batch, head, qblock, part, i, 0] = maximum[i]
                    stats[batch, head, qblock, part, i, 1] = normalizer[i]

        return kernel
    if kind == "attention_merge":

        @T.prim_func
        def kernel(
            partial: T.Tensor(shape + (d,), "float32"),
            stats: T.Tensor(shape + (2,), "float32"),
            proj: T.Tensor((rows, projection), "float32"),
            lengths: T.Tensor((slots,), "int32"),
            out: T.Tensor((rows, heads * d), "bfloat16"),
        ):
            with T.Kernel(slots, heads, chunk, threads=256) as (batch, head, t):
                if t < lengths[batch]:
                    maximum = T.alloc_var("float32")
                    denominator = T.alloc_var("float32")
                    maximum = -T.infinity("float32")
                    denominator = 0
                    for part in T.serial(splits):
                        maximum = T.max(
                            maximum,
                            stats[
                                batch,
                                head // group,
                                t // qt,
                                part,
                                t % qt * group + head % group,
                                0,
                            ],
                        )
                    for part in T.serial(splits):
                        m = stats[
                            batch,
                            head // group,
                            t // qt,
                            part,
                            t % qt * group + head % group,
                            0,
                        ]
                        if m > -T.infinity("float32"):
                            denominator += stats[
                                batch,
                                head // group,
                                t // qt,
                                part,
                                t % qt * group + head % group,
                                1,
                            ] * T.exp(m - maximum)
                    for j in T.Parallel(d):
                        value = T.alloc_var("float32")
                        value = 0
                        for part in T.serial(splits):
                            m = stats[
                                batch,
                                head // group,
                                t // qt,
                                part,
                                t % qt * group + head % group,
                                0,
                            ]
                            if m > -T.infinity("float32"):
                                value += partial[
                                    batch,
                                    head // group,
                                    t // qt,
                                    part,
                                    t % qt * group + head % group,
                                    j,
                                ] * T.exp(m - maximum)
                        normal = T.cast(
                            T.cast(value / denominator, "bfloat16"), "float32"
                        )
                        gate = T.cast(
                            T.cast(
                                proj[batch * chunk + t, head * 2 * d + d + j],
                                "bfloat16",
                            ),
                            "float32",
                        )
                        out[batch * chunk + t, head * d + j] = normal / (
                            1 + T.exp(-gate)
                        )

        return kernel
    raise ValueError("unknown FP8 dense attention operation")
