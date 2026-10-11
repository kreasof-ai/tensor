"""Tensor DSL factories for embedding operations; compiler imports are lazy."""

import math


def make_kernel(kind, p, schedule=None):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    n, c, r = p.get("n"), p.get("c"), p.get("r")
    f16, f32 = p.get("dtype", "float16"), "float32"
    a = lambda name, count, dtype=f16: (name, count, dtype)

    if kind in ("embedding", "embedding_backward"):
        b, s, c, v = p["b"], p["s"], p["c"], p["v"]
        n = b * s * c
        if kind == "embedding":

            @T.prim_func
            def kernel(
                tokens: T.Tensor((b * s,), "int32"),
                w: T.Tensor((v * c,), f16),
                position: T.Tensor((s * c,), f16),
                out: T.Tensor((n,), f16),
            ):
                with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                    for lane in T.Parallel(256):
                        i = block * 256 + lane
                        if i < n:
                            out[i] = T.cast(
                                w[tokens[i // c] * c + i % c], f32
                            ) + T.cast(position[(i // c % s) * c + i % c], f32)

        else:

            @T.prim_func
            def kernel(
                tokens: T.Tensor((b * s,), "int32"),
                dy: T.Tensor((n,), f16),
                dw: T.Tensor((v * c,), f32),
                dp: T.Tensor((s * c,), f32),
            ):
                with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                    for lane in T.Parallel(256):
                        i = block * 256 + lane
                        if i < n:
                            T.atomic_add(
                                dw[tokens[i // c] * c + i % c], T.cast(dy[i], f32)
                            )
                            T.atomic_add(
                                dp[(i // c % s) * c + i % c], T.cast(dy[i], f32)
                            )

        return kernel

    raise ValueError(f"unsupported embedding kernel {kind}")
