"""Tensor DSL factories for optimizer operations; compiler imports are lazy."""

import math


def make_kernel(kind, p, schedule=None):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    n, c, r = p.get("n"), p.get("c"), p.get("r")
    f16, f32 = p.get("dtype", "float16"), "float32"
    a = lambda name, count, dtype=f16: (name, count, dtype)

    if kind in ("sumsq", "sum_parts", "clip"):
        chunks = math.ceil(n / 1024)
        tile = 1 << (n - 1).bit_length() if kind == "clip" else 1024
        blocks = 1 if kind == "clip" else chunks

        @T.macro
        def reduce(x, out, offset=0):
            with T.Kernel(blocks, threads=256) as block:
                values = T.alloc_fragment((tile,), f32)
                total = T.alloc_fragment((1,), f32)
                for i in T.Parallel(tile):
                    index = block * tile + i
                    if kind == "sumsq":
                        values[i] = T.if_then_else(
                            index < n,
                            (x[index] / p["scale"]) * (x[index] / p["scale"]),
                            0,
                        )
                    else:
                        values[i] = T.if_then_else(index < n, x[index], 0)
                T.reduce_sum(values, total, dim=0)
                if kind == "clip":
                    out[0] = T.if_then_else(
                        T.isnan(total[0]) | (total[0] == T.infinity(f32)),
                        T.infinity(f32),
                        T.min(1.0, p["limit"] / (T.sqrt(total[0]) + 0.000001)),
                    )
                    out[1] = T.sqrt(total[0])
                elif kind == "sumsq":
                    out[offset + block] = total[0]
                else:
                    out[block] = total[0]

        arguments = [
            a("x", n, f32),
            a(
                "out",
                p["total"] if kind == "sumsq" else 2 if kind == "clip" else chunks,
                f32,
            ),
        ]
        if kind == "sumsq":
            arguments.append(("offset", None, "int32"))
        return primitive(arguments, reduce)

    if kind == "adamw":

        @T.prim_func
        def kernel(
            weight: T.Tensor((n,), f32),
            half: T.Tensor((n,), f16),
            grad: T.Tensor((n,), f32),
            moment: T.Tensor((n,), f32),
            variance: T.Tensor((n,), f32),
            clip: T.Tensor((2,), f32),
            lr: T.float32,
            correction1: T.float32,
            correction2: T.float32,
        ):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        if clip[0] != T.infinity(f32):
                            g = grad[i] / p["scale"] * clip[0]
                            m = 0.9 * moment[i] + 0.1 * g
                            v = 0.95 * variance[i] + 0.05 * g * g
                            value = weight[i] * (1 - lr * p["decay"]) - lr * (
                                m / correction1
                            ) / (T.sqrt(v / correction2) + 0.00000001)
                            moment[i] = m
                            variance[i] = v
                            weight[i] = value
                            half[i] = value

        return kernel

    raise ValueError(f"unsupported optimizer kernel {kind}")
