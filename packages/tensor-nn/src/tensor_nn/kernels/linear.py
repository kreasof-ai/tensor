"""Tensor DSL factories for linear operations; compiler imports are lazy."""

import math


def make_kernel(kind, p, schedule=None):
    import tilelang.language as T
    from tensor.compiler.frontend.source import primitive

    n, c, r = p.get("n"), p.get("c"), p.get("r")
    f16, f32 = p.get("dtype", "float16"), "float32"
    a = lambda name, count, dtype=f16: (name, count, dtype)

    if kind in ("gemm", "gemm_gelu", "gemm_residual"):
        b, m, k, cols = p["batch"], p["m"], p["k"], p["cols"]
        ta, tb = p.get("ta", False), p.get("tb", False)
        bm, bn, bk, stages = schedule or (32, 64, 32, 2)
        aa, bb = ((bk, bm) if ta else (bm, bk)), ((bn, bk) if tb else (bk, bn))

        @T.macro
        def matmul(x, w, out, extra=None):
            with T.Kernel(T.ceildiv(cols, bn), T.ceildiv(m, bm), b, threads=128) as (
                bx,
                by,
                batch,
            ):
                lhs = T.alloc_shared(aa, f16)
                rhs = T.alloc_shared(bb, f16)
                accum = T.alloc_fragment((bm, bn), f32)
                T.clear(accum)
                for tile in T.Pipelined(T.ceildiv(k, bk), num_stages=stages):
                    for i, j in T.Parallel(aa[0], aa[1]):
                        if ta:
                            lhs[i, j] = T.if_then_else(
                                (tile * bk + i < k) & (by * bm + j < m),
                                x[(batch * k + tile * bk + i) * m + by * bm + j],
                                0,
                            )
                        else:
                            lhs[i, j] = T.if_then_else(
                                (by * bm + i < m) & (tile * bk + j < k),
                                x[(batch * m + by * bm + i) * k + tile * bk + j],
                                0,
                            )
                    for i, j in T.Parallel(bb[0], bb[1]):
                        if tb:
                            rhs[i, j] = T.if_then_else(
                                (bx * bn + i < cols) & (tile * bk + j < k),
                                w[(batch * cols + bx * bn + i) * k + tile * bk + j],
                                0,
                            )
                        else:
                            rhs[i, j] = T.if_then_else(
                                (tile * bk + i < k) & (bx * bn + j < cols),
                                w[(batch * k + tile * bk + i) * cols + bx * bn + j],
                                0,
                            )
                    T.gemm(lhs, rhs, accum, transpose_A=ta, transpose_B=tb)
                for i, j in T.Parallel(bm, bn):
                    if (by * bm + i < m) & (bx * bn + j < cols):
                        index = (batch * m + by * bm + i) * cols + bx * bn + j
                        if kind == "gemm_gelu":
                            value = T.cast(T.cast(accum[i, j], f16), f32)
                            extra[index] = value
                            out[index] = (
                                0.5 * value * (1 + T.erf(value * 0.7071067811865476))
                            )
                        elif kind == "gemm_residual":
                            out[index] = T.cast(T.cast(accum[i, j], f16), f32) + T.cast(
                                extra[index], f32
                            )
                        else:
                            out[index] = accum[i, j]

        if kind == "gemm":

            @T.prim_func
            def kernel(
                x: T.Tensor((b * m * k,), f16),
                w: T.Tensor((b * k * cols,), f16),
                out: T.Tensor((b * m * cols,), p.get("out_dtype", f16)),
            ):
                matmul(x, w, out)

        else:
            arguments = [
                a("x", b * m * k),
                a("w", b * k * cols),
                a("pre" if kind == "gemm_gelu" else "residual", b * m * cols),
                a("out", b * m * cols),
            ]

            @T.macro
            def call(x, w, extra, out):
                matmul(x, w, out, extra)

            return primitive(arguments, call)
        return kernel

    raise ValueError(f"unsupported linear kernel {kind}")
