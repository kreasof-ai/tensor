"""Tensor DSL factories for normalization operations; compiler imports are lazy."""

import math


def make_kernel(kind, p, schedule=None):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    n, c, r = p.get("n"), p.get("c"), p.get("r")
    f16, f32 = p.get("dtype", "float16"), "float32"
    a = lambda name, count, dtype=f16: (name, count, dtype)

    if kind == "norm":
        tile = 1 << (c - 1).bit_length()

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), f16),
            weight: T.Tensor((c,), f32),
            out: T.Tensor((r * c,), f16),
            normal: T.Tensor((r * c,), f32),
            inverse: T.Tensor((r,), f32),
        ):
            with T.Kernel(r, threads=256) as row:
                values = T.alloc_fragment((tile,), f32)
                square = T.alloc_fragment((tile,), f32)
                total = T.alloc_fragment((1,), f32)
                variance = T.alloc_fragment((1,), f32)
                for i in T.Parallel(tile):
                    values[i] = T.if_then_else(i < c, T.cast(x[row * c + i], f32), 0)
                T.reduce_sum(values, total, dim=0)
                for i in T.Parallel(tile):
                    square[i] = T.if_then_else(
                        i < c,
                        (values[i] - total[0] / c) * (values[i] - total[0] / c),
                        0,
                    )
                T.reduce_sum(square, variance, dim=0)
                for i in T.Parallel(tile):
                    if i < c:
                        z = (values[i] - total[0] / c) * T.rsqrt(
                            variance[0] / c + 0.00001
                        )
                        normal[row * c + i] = z
                        out[row * c + i] = z * weight[i]
                inverse[row] = T.rsqrt(variance[0] / c + 0.00001)

        return kernel

    if kind == "norm_backward":
        tile = 1 << (c - 1).bit_length()

        @T.prim_func
        def kernel(
            dy: T.Tensor((r * c,), f16),
            weight: T.Tensor((c,), f32),
            normal: T.Tensor((r * c,), f32),
            inverse: T.Tensor((r,), f32),
            dx: T.Tensor((r * c,), f16),
            parts: T.Tensor((r * c,), f32),
        ):
            with T.Kernel(r, threads=256) as row:
                values = T.alloc_fragment((tile,), f32)
                product = T.alloc_fragment((tile,), f32)
                total = T.alloc_fragment((1,), f32)
                dot = T.alloc_fragment((1,), f32)
                for i in T.Parallel(tile):
                    values[i] = T.if_then_else(
                        i < c, T.cast(dy[row * c + i], f32) * weight[i], 0
                    )
                    product[i] = T.if_then_else(
                        i < c, values[i] * normal[row * c + i], 0
                    )
                T.reduce_sum(values, total, dim=0)
                T.reduce_sum(product, dot, dim=0)
                for i in T.Parallel(tile):
                    if i < c:
                        dx[row * c + i] = inverse[row] * (
                            values[i] - (total[0] + normal[row * c + i] * dot[0]) / c
                        )
                        parts[row * c + i] = (
                            T.cast(dy[row * c + i], f32) * normal[row * c + i]
                        )

        return kernel

    if kind == "column_sum":

        @T.prim_func
        def kernel(x: T.Tensor((r * c,), f32), out: T.Tensor((c,), f32)):
            with T.Kernel(T.ceildiv(c, 256), threads=256) as block:
                sums = T.alloc_fragment((256,), f32)
                T.clear(sums)
                for row in T.serial(r):
                    for i in T.Parallel(256):
                        if block * 256 + i < c:
                            sums[i] += x[row * c + block * 256 + i]
                for i in T.Parallel(256):
                    if block * 256 + i < c:
                        out[block * 256 + i] = sums[i]

        return kernel

    raise ValueError(f"unsupported normalization kernel {kind}")
