"""Producer factories used by the CUDA showcase (forward inference only)."""

import tilelang.language as T


def mla_partial(batch, heads, length, splits=8, block_n=32, threads=128, value_dim=576):
    """Dense V3-style absorbed MLA: QK=576, V=first 512 cache channels.

    Cache pages are contiguous, equally sized and have no padding. Each split
    writes a normalized partial and its log-sum-exp; merge is mandatory.
    """
    depth, values, block_m = 576, 512, 16
    if value_dim not in (512, 576):
        raise ValueError("MLA value tile must cover 512 or 576 cache channels")
    chunk = T.ceildiv(length, splits * block_n)
    scale = depth**-0.5

    @T.prim_func
    def partial(
        q: T.Tensor((batch, heads, depth), "float16"),
        cache: T.Tensor((batch, length, depth), "float16"),
        parts: T.Tensor((batch, heads, splits, values), "float32"),
        lse: T.Tensor((batch, heads, splits), "float32"),
    ):
        with T.Kernel(T.ceildiv(heads, block_m), splits, batch, threads=threads) as (
            h,
            s,
            b,
        ):
            query = T.alloc_shared((block_m, depth), "float16")
            key = T.alloc_shared((block_n, depth), "float16")
            prob = T.alloc_shared((block_m, block_n), "float16")
            score = T.alloc_fragment((block_m, block_n), "float32")
            acc = T.alloc_fragment((block_m, value_dim), "float32")
            mx = T.alloc_fragment((block_m,), "float32")
            old = T.alloc_fragment((block_m,), "float32")
            alpha = T.alloc_fragment((block_m,), "float32")
            denom = T.alloc_fragment((block_m,), "float32")
            total = T.alloc_fragment((block_m,), "float32")
            T.copy(q[b, h * block_m, 0], query)
            T.clear(acc)
            T.clear(denom)
            T.fill(mx, -T.infinity("float32"))
            for step in T.serial(chunk):
                start = (s * chunk + step) * block_n
                T.copy(cache[b, start, 0], key)
                T.gemm(
                    query,
                    key,
                    score,
                    transpose_B=True,
                    clear_accum=True,
                    policy=T.GemmWarpPolicy.FullRow,
                )
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.if_then_else(
                        start + j < length, score[i, j] * scale, -T.infinity("float32")
                    )
                T.copy(mx, old)
                T.reduce_max(score, mx, dim=1, clear=True)
                for i in T.Parallel(block_m):
                    mx[i] = T.max(old[i], mx[i])
                    alpha[i] = T.exp(old[i] - mx[i])
                for i, j in T.Parallel(block_m, block_n):
                    score[i, j] = T.exp(score[i, j] - mx[i])
                T.reduce_sum(score, total, dim=1, clear=True)
                for i in T.Parallel(block_m):
                    denom[i] = denom[i] * alpha[i] + total[i]
                for i, j in T.Parallel(block_m, value_dim):
                    acc[i, j] *= alpha[i]
                T.copy(score, prob)
                T.gemm(
                    prob,
                    key[0:block_n, 0:value_dim],
                    acc,
                    policy=T.GemmWarpPolicy.FullRow,
                )
            for i, j in T.Parallel(block_m, values):
                if h * block_m + i < heads:
                    parts[b, h * block_m + i, s, j] = acc[i, j] / denom[i]
            for i in T.Parallel(block_m):
                if h * block_m + i < heads:
                    lse[b, h * block_m + i, s] = mx[i] + T.log(denom[i])

    return partial


def mla_merge(batch, heads, splits=8):
    @T.prim_func
    def merge(
        parts: T.Tensor((batch, heads, splits, 512), "float32"),
        lse: T.Tensor((batch, heads, splits), "float32"),
        out: T.Tensor((batch, heads, 512), "float16"),
    ):
        with T.Kernel(heads, batch, threads=128) as (h, b):
            weights = T.alloc_fragment((splits,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            denom = T.alloc_fragment((1,), "float32")
            T.copy(lse[b, h, 0], weights)
            T.reduce_max(weights, maximum, dim=0, clear=True)
            for s in T.Parallel(splits):
                weights[s] = T.exp(weights[s] - maximum[0])
            T.reduce_sum(weights, denom, dim=0, clear=True)
            for j in T.Parallel(512):
                value = T.alloc_var("float32")
                value = 0
                for s in T.serial(splits):
                    value += parts[b, h, s, j] * weights[s]
                out[b, h, j] = value / denom[0]

    return merge


def gguf(m, n, k, encoding, **schedule):
    """Reuse AKAS's packed GGML GEMV and fused dequant/Tensor Core kernels."""
    from akas.ops.linear.cuda import _packed, gemv_packed

    p = {"r": m, "o": n, "k": k, "type": encoding, **schedule}
    return gemv_packed(p) if m == 1 else _packed(p)
