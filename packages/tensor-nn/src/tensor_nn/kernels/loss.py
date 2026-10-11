"""Tensor DSL factories for loss operations; compiler imports are lazy."""

import math


def make_kernel(kind, p, schedule=None):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    n, c, r = p.get("n"), p.get("c"), p.get("r")
    f16, f32 = p.get("dtype", "float16"), "float32"
    a = lambda name, count, dtype=f16: (name, count, dtype)

    if kind == "ce_parts":
        r, v = p["r"], p["v"]
        chunks = math.ceil(v / 1024)

        @T.prim_func
        def kernel(x: T.Tensor((r * v,), f16), parts: T.Tensor((r * chunks * 2,), f32)):
            with T.Kernel(chunks, r, threads=256) as (block, row):
                values = T.alloc_fragment((1024,), f32)
                maximum = T.alloc_fragment((1,), f32)
                total = T.alloc_fragment((1,), f32)
                for i in T.Parallel(1024):
                    values[i] = T.if_then_else(
                        block * 1024 + i < v,
                        T.cast(x[row * v + block * 1024 + i], f32),
                        -T.infinity(f32),
                    )
                T.reduce_max(values, maximum, dim=0)
                for i in T.Parallel(1024):
                    values[i] = T.exp(values[i] - maximum[0])
                T.reduce_sum(values, total, dim=0)
                parts[(row * chunks + block) * 2] = maximum[0]
                parts[(row * chunks + block) * 2 + 1] = total[0]

        return kernel

    if kind == "ce_loss":
        r, v = p["r"], p["v"]
        chunks = math.ceil(v / 1024)
        tile = 1 << (chunks - 1).bit_length()

        @T.prim_func
        def kernel(
            x: T.Tensor((r * v,), f16),
            target: T.Tensor((r,), "int32"),
            parts: T.Tensor((r * chunks * 2,), f32),
            lse: T.Tensor((r,), f32),
            loss: T.Tensor((r,), f32),
        ):
            with T.Kernel(r, threads=128) as row:
                maxima = T.alloc_fragment((tile,), f32)
                sums = T.alloc_fragment((tile,), f32)
                maximum = T.alloc_fragment((1,), f32)
                total = T.alloc_fragment((1,), f32)
                for i in T.Parallel(tile):
                    maxima[i] = T.if_then_else(
                        i < chunks, parts[(row * chunks + i) * 2], -T.infinity(f32)
                    )
                T.reduce_max(maxima, maximum, dim=0)
                for i in T.Parallel(tile):
                    sums[i] = T.if_then_else(
                        i < chunks,
                        parts[(row * chunks + i) * 2 + 1]
                        * T.exp(maxima[i] - maximum[0]),
                        0,
                    )
                T.reduce_sum(sums, total, dim=0)
                lse[row] = maximum[0] + T.log(total[0])
                loss[row] = (
                    maximum[0] + T.log(total[0]) - T.cast(x[row * v + target[row]], f32)
                )

        return kernel

    if kind == "ce_backward":
        r, v = p["r"], p["v"]
        n = r * v

        @T.prim_func
        def kernel(
            x: T.Tensor((n,), f16),
            target: T.Tensor((r,), "int32"),
            lse: T.Tensor((r,), f32),
            dy: T.Tensor((r,), f32),
            out: T.Tensor((n,), f16),
        ):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        out[i] = (
                            T.exp(T.cast(x[i], f32) - lse[i // v])
                            - T.if_then_else(i % v == target[i // v], 1.0, 0.0)
                        ) * dy[i // v]

        return kernel

    raise ValueError(f"unsupported loss kernel {kind}")
