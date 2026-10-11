"""Tensor DSL factories for elementwise operations; compiler imports are lazy."""

import math


def make_kernel(kind, p, schedule=None):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    n, c, r = p.get("n"), p.get("c"), p.get("r")
    f16, f32 = p.get("dtype", "float16"), "float32"
    a = lambda name, count, dtype=f16: (name, count, dtype)

    if kind == "zero":

        @T.prim_func
        def kernel(out: T.Tensor((n,), p.get("dtype", f32))):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        out[i] = 0

        return kernel

    if kind == "cast":

        @T.prim_func
        def kernel(
            x: T.Tensor((n,), p.get("input", f16)),
            out: T.Tensor((n,), p.get("output", f32)),
        ):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        if p.get("add"):
                            out[i] = T.cast(x[i], f32) + out[i]
                        else:
                            out[i] = T.cast(x[i], f32)

        return kernel

    if kind in ("add", "gelu", "gelu_backward"):

        @T.macro
        def elementwise(x, out, y=None):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        if kind == "add":
                            out[i] = T.cast(x[i], f32) + T.cast(y[i], f32)
                        elif kind == "gelu":
                            out[i] = (
                                0.5
                                * T.cast(x[i], f32)
                                * (1 + T.erf(T.cast(x[i], f32) * 0.7071067811865476))
                            )
                        else:
                            out[i] = T.cast(y[i], f32) * (
                                0.5
                                * (1 + T.erf(T.cast(x[i], f32) * 0.7071067811865476))
                                + T.cast(x[i], f32)
                                * 0.3989422804014327
                                * T.exp(-0.5 * T.cast(x[i], f32) * T.cast(x[i], f32))
                            )

        if kind == "gelu":

            @T.prim_func
            def kernel(x: T.Tensor((n,), f16), out: T.Tensor((n,), f16)):
                elementwise(x, out)

        elif kind == "gelu_backward":

            @T.prim_func
            def kernel(
                x: T.Tensor((n,), f16),
                dy: T.Tensor((n,), f16),
                out: T.Tensor((n,), f16),
            ):
                elementwise(x, out, dy)

        else:

            @T.prim_func
            def kernel(
                x: T.Tensor((n,), f16), y: T.Tensor((n,), f16), out: T.Tensor((n,), f16)
            ):
                elementwise(x, out, y)

        return kernel

    raise ValueError(f"unsupported elementwise kernel {kind}")
