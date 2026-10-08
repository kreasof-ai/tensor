"""Partitioned cached decode with dynamic prefix lengths and FP32 merge.

Adapted from the LLT L40S forward study, with BF16 and separate value dimensions.
"""


def decode_partial(p):
    batch, heads, kv_heads, n, dim, dv, dtype, scale, partitions = p
    bn = 64
    import tilelang.language as T

    bm = 32

    @T.prim_func
    def decode(
        q: T.Tensor((batch, heads, 1, dim), dtype),
        k: T.Tensor((batch, kv_heads, n, dim), dtype),
        v: T.Tensor((batch, kv_heads, n, dv), dtype),
        lengths: T.Tensor((batch,), "int32"),
        partial: T.Tensor((batch, heads, partitions, dv), "float32"),
        stats: T.Tensor((batch, heads, partitions, 2), "float32"),
    ):
        with T.Kernel(partitions, heads, batch, threads=128) as (p, h, b):
            query = T.alloc_shared((bm, dim), dtype)
            key = T.alloc_shared((bn, dim), dtype)
            value = T.alloc_shared((bn, dv), dtype)
            probability = T.alloc_shared((bm, bn), dtype)
            scores = T.alloc_fragment((bm, bn), "float32")
            result = T.alloc_fragment((bm, dv), "float32")
            maximum = T.alloc_fragment((bm,), "float32")
            previous = T.alloc_fragment((bm,), "float32")
            factor = T.alloc_fragment((bm,), "float32")
            normalizer = T.alloc_fragment((bm,), "float32")
            tile_sum = T.alloc_fragment((bm,), "float32")
            # A direct scalar extraction from a reduced fragment constrains its
            # replicated layout incompatibly in TileLang 0.1.14. Stage through
            # shared memory before exporting the single valid query row.
            result_shared = T.alloc_shared((bm, dv), "float32")
            maximum_shared = T.alloc_shared((bm,), "float32")
            normalizer_shared = T.alloc_shared((bm,), "float32")
            tiles = T.ceildiv(lengths[b], bn)
            per = T.ceildiv(tiles, partitions)
            kh = h // (heads // kv_heads)
            T.copy(q[b, h, 0:bm, :], query)
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            for local in T.serial(T.max(0, T.min(per, tiles - p * per))):
                tile = p * per + local
                for i, j in T.Parallel(bn, dim):
                    key[i, j] = T.if_then_else(
                        tile * bn + i < lengths[b], k[b, kh, tile * bn + i, j], 0
                    )
                T.gemm(
                    query,
                    key,
                    scores,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(bm, bn):
                    scores[i, j] = T.if_then_else(
                        tile * bn + j < lengths[b],
                        scores[i, j] * scale,
                        -T.infinity("float32"),
                    )
                T.copy(maximum, previous)
                T.reduce_max(scores, maximum, dim=1, clear=True)
                for i in T.Parallel(bm):
                    maximum[i] = T.max(previous[i], maximum[i])
                    factor[i] = T.exp(previous[i] - maximum[i])
                for i, j in T.Parallel(bm, bn):
                    scores[i, j] = T.exp(scores[i, j] - maximum[i])
                T.reduce_sum(scores, tile_sum, dim=1, clear=True)
                for i in T.Parallel(bm):
                    normalizer[i] = normalizer[i] * factor[i] + tile_sum[i]
                for i, j in T.Parallel(bm, dv):
                    result[i, j] *= factor[i]
                T.copy(scores, probability)
                for i, j in T.Parallel(bn, dv):
                    value[i, j] = T.if_then_else(
                        tile * bn + i < lengths[b], v[b, kh, tile * bn + i, j], 0
                    )
                T.gemm(probability, value, result, policy=T.GemmWarpPolicy.FullRow)
            T.copy(result, result_shared)
            T.copy(maximum, maximum_shared)
            T.copy(normalizer, normalizer_shared)
            for j in T.Parallel(dv):
                partial[b, h, p, j] = result_shared[0, j]
            stats[b, h, p, 0] = maximum_shared[0]
            stats[b, h, p, 1] = normalizer_shared[0]

    return decode


def decode_merge(p):
    batch, heads, dim, dtype, partitions = p
    import tilelang.language as T

    @T.prim_func
    def merge(
        partial: T.Tensor((batch, heads, partitions, dim), "float32"),
        stats: T.Tensor((batch, heads, partitions, 2), "float32"),
        out: T.Tensor((batch, heads, 1, dim), dtype),
    ):
        with T.Kernel(heads, batch, threads=128) as (h, b):
            maxima = T.alloc_fragment((partitions,), "float32")
            weights = T.alloc_fragment((partitions,), "float32")
            normal = T.alloc_fragment((partitions,), "float32")
            total = T.alloc_fragment((1,), "float32")
            largest = T.alloc_fragment((1,), "float32")
            values = T.alloc_fragment((partitions, dim), "float32")
            accum = T.alloc_fragment((dim,), "float32")
            for p in T.Parallel(partitions):
                maxima[p] = stats[b, h, p, 0]
            T.reduce_max(maxima, largest, dim=0, clear=True)
            for p in T.Parallel(partitions):
                weights[p] = T.exp(maxima[p] - largest[0])
                normal[p] = weights[p] * stats[b, h, p, 1]
            T.reduce_sum(normal, total, dim=0, clear=True)
            for p, j in T.Parallel(partitions, dim):
                values[p, j] = weights[p] * partial[b, h, p, j]
            T.reduce_sum(values, accum, dim=0, clear=True)
            for j in T.Parallel(dim):
                out[b, h, 0, j] = accum[j] / total[0]

    return merge


def cache_advance(p):
    import tilelang.language as T

    batch, capacity, size = p

    @T.prim_func
    def kernel(
        lengths: T.Tensor((batch,), "int32"), overflow: T.Tensor((batch,), "int32")
    ):
        with T.Kernel(T.ceildiv(batch, 128), threads=128) as block:
            for lane in T.Parallel(128):
                i = block * 128 + lane
                if i < batch:
                    if lengths[i] + size > capacity:
                        overflow[i] = 1
                    else:
                        lengths[i] += size

    return kernel


def cache_append(p):
    import tilelang.language as T

    batch, heads, capacity, d, dv, size, dtype, shared = p
    if shared:

        @T.prim_func
        def kernel(
            k: T.Tensor((batch, heads, size, d), dtype),
            lengths: T.Tensor((batch,), "int32"),
            overflow: T.Tensor((batch,), "int32"),
            keys: T.Tensor((batch, heads, capacity, d), dtype),
        ):
            with T.Kernel(T.ceildiv(size * d, 256), heads, batch, threads=256) as (
                block,
                head,
                b,
            ):
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if overflow[b] == 0:
                        if i < size * d:
                            keys[b, head, lengths[b] - size + i // d, i % d] = k[
                                b, head, i // d, i % d
                            ]

        return kernel

    @T.prim_func
    def kernel(
        k: T.Tensor((batch, heads, size, d), dtype),
        v: T.Tensor((batch, heads, size, dv), dtype),
        lengths: T.Tensor((batch,), "int32"),
        overflow: T.Tensor((batch,), "int32"),
        keys: T.Tensor((batch, heads, capacity, d), dtype),
        values: T.Tensor((batch, heads, capacity, dv), dtype),
    ):
        with T.Kernel(
            T.ceildiv(size * T.max(d, dv), 256), heads, batch, threads=256
        ) as (block, head, b):
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if overflow[b] == 0:
                    if i < size * d:
                        keys[b, head, lengths[b] - size + i // d, i % d] = k[
                            b, head, i // d, i % d
                        ]
                    if i < size * dv:
                        values[b, head, lengths[b] - size + i // dv, i % dv] = v[
                            b, head, i // dv, i % dv
                        ]

    return kernel
