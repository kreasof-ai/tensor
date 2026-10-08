"""Partitioned cached decode with dynamic prefix lengths and FP32 merge.

Adapted from the LLT L40S forward study, with BF16 and separate value dimensions.
"""


def shared_decode_partial(p):
    batch, heads, kv_heads, ng, nl, dim, dv, dtype, scale, partitions = p
    n = ng + nl
    bn = 64
    import tilelang.language as T

    bm = 16

    @T.prim_func
    def decode(
        q: T.Tensor((batch, heads, 1, dim), dtype),
        k: T.Tensor((batch, kv_heads, ng, dim), dtype),
        lk: T.Tensor((batch, kv_heads, nl, dim), dtype),
        v: T.Tensor((batch, kv_heads, ng, dv), dtype),
        lv: T.Tensor((batch, kv_heads, nl, dv), dtype),
        lengths: T.Tensor((batch,), "int32"),
        local_lengths: T.Tensor((batch,), "int32"),
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
            tiles = T.ceildiv(n, bn)
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
                        tile * bn + i < ng,
                        T.if_then_else(
                            tile * bn + i < lengths[b], k[b, kh, tile * bn + i, j], 0
                        ),
                        T.if_then_else(
                            tile * bn + i - ng < local_lengths[b],
                            lk[b, kh, tile * bn + i - ng, j],
                            0,
                        ),
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
                        (
                            (tile * bn + j < lengths[b])
                            | (
                                (tile * bn + j >= ng)
                                & (tile * bn + j - ng < local_lengths[b])
                            )
                        ),
                        scores[i, j] * scale,
                        -T.infinity("float32"),
                    )
                T.copy(maximum, previous)
                T.reduce_max(scores, maximum, dim=1, clear=True)
                for i in T.Parallel(bm):
                    maximum[i] = T.max(previous[i], maximum[i])
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
                T.reduce_sum(scores, tile_sum, dim=1, clear=True)
                for i in T.Parallel(bm):
                    normalizer[i] = normalizer[i] * factor[i] + tile_sum[i]
                for i, j in T.Parallel(bm, dv):
                    result[i, j] *= factor[i]
                T.copy(scores, probability)
                for i, j in T.Parallel(bn, dv):
                    value[i, j] = T.if_then_else(
                        tile * bn + i < ng,
                        T.if_then_else(
                            tile * bn + i < lengths[b], v[b, kh, tile * bn + i, j], 0
                        ),
                        T.if_then_else(
                            tile * bn + i - ng < local_lengths[b],
                            lv[b, kh, tile * bn + i - ng, j],
                            0,
                        ),
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


from .llt_decode import decode_merge
