"""Tensor DSL factories for attention operations; compiler imports are lazy."""

import math


def make_kernel(kind, p, schedule=None):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    n, c, r = p.get("n"), p.get("c"), p.get("r")
    f16, f32 = p.get("dtype", "float16"), "float32"
    a = lambda name, count, dtype=f16: (name, count, dtype)

    if kind in ("pack_qkv", "unpack_qkv", "merge_heads", "split_heads"):
        b, s, c, h = p["b"], p["s"], p["c"], p["h"]
        d, n = c // h, b * s * c

        @T.macro
        def rearrange(x, out, k=None, v=None):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        j = (
                            (i // (s * d) // h * s + i // d % s) * c
                            + (i // (s * d) % h) * d
                            + i % d
                        )
                        if kind == "pack_qkv":
                            out[i] = x[j // c * (3 * c) + j % c]
                            k[i] = x[j // c * (3 * c) + c + j % c]
                            v[i] = x[j // c * (3 * c) + 2 * c + j % c]
                        elif kind == "unpack_qkv":
                            out[j // c * (3 * c) + j % c] = x[i]
                            out[j // c * (3 * c) + c + j % c] = k[i]
                            out[j // c * (3 * c) + 2 * c + j % c] = v[i]
                        elif kind == "merge_heads":
                            out[j] = x[i]
                        else:
                            out[i] = x[j]

        if kind == "pack_qkv":

            @T.macro
            def call(x, q, k, v):
                rearrange(x, q, k, v)

            return primitive([a("x", 3 * n), a("q", n), a("k", n), a("v", n)], call)
        if kind == "unpack_qkv":

            @T.macro
            def call(q, k, v, out):
                rearrange(q, out, k, v)

            return primitive([a("q", n), a("k", n), a("v", n), a("out", 3 * n)], call)
        return primitive([a("x", n), a("out", n)], rearrange)

    if kind in ("softmax", "softmax_backward"):
        s, bh, d = p["s"], p["bh"], p["d"]
        tile, scale = 1 << (s - 1).bit_length(), d**-0.5
        if kind == "softmax":

            @T.prim_func
            def kernel(
                scores: T.Tensor((bh * s * s,), f16),
                prob: T.Tensor((bh * s * s,), f16),
                saved: T.Tensor((bh * s * s,), f32),
            ):
                with T.Kernel(bh * s, threads=256) as row:
                    values = T.alloc_fragment((tile,), f32)
                    maximum = T.alloc_fragment((1,), f32)
                    total = T.alloc_fragment((1,), f32)
                    for i in T.Parallel(tile):
                        values[i] = T.if_then_else(
                            (i < s) & (i <= row % s),
                            T.cast(
                                T.cast(T.cast(scores[row * s + i], f32) * scale, f16),
                                f32,
                            ),
                            -T.infinity(f32),
                        )
                    T.reduce_max(values, maximum, dim=0)
                    for i in T.Parallel(tile):
                        values[i] = T.exp(values[i] - maximum[0])
                    T.reduce_sum(values, total, dim=0)
                    for i in T.Parallel(tile):
                        if i < s:
                            saved[row * s + i] = values[i] / total[0]
                            prob[row * s + i] = values[i] / total[0]

        else:

            @T.prim_func
            def kernel(
                dp: T.Tensor((bh * s * s,), f16),
                prob: T.Tensor((bh * s * s,), f32),
                ds: T.Tensor((bh * s * s,), f16),
            ):
                with T.Kernel(bh * s, threads=256) as row:
                    product = T.alloc_fragment((tile,), f32)
                    total = T.alloc_fragment((1,), f32)
                    for i in T.Parallel(tile):
                        product[i] = T.if_then_else(
                            i < s, T.cast(dp[row * s + i], f32) * prob[row * s + i], 0
                        )
                    T.reduce_sum(product, total, dim=0)
                    for i in T.Parallel(tile):
                        if i < s:
                            ds[row * s + i] = (
                                T.cast(
                                    T.cast(
                                        (T.cast(dp[row * s + i], f32) - total[0])
                                        * prob[row * s + i],
                                        f16,
                                    ),
                                    f32,
                                )
                                * scale
                            )

        return kernel

    raise ValueError(f"unsupported attention kernel {kind}")
