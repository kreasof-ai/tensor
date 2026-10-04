"""TileLang DSL for the bounded manual-training operator library.

Factories specialize contiguous, flattened buffers. Compiler imports are lazy;
runtime identities remain available without TileLang, TVM or Torch.
"""

import hashlib
import json
import math


def identity(kind, parameters):
    return hashlib.sha256(json.dumps([kind, parameters], sort_keys=True).encode()).hexdigest()[:24]


def source(kind, parameters, schedule=None):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "make_kernel",
        kind,
        parameters,
        schedule,
        dependencies=("tensor_nn.kernels", "tensor.compiler.entry"),
    )


def make_kernel(kind, p, schedule=None):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    n, c, r = p.get("n"), p.get("c"), p.get("r")
    f16, f32 = "float16", "float32"
    a = lambda name, count, dtype=f16: (name, count, dtype)
    scalar = lambda name: (name, None, f32)

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
            x: T.Tensor((n,), p.get("input", f16)), out: T.Tensor((n,), p.get("output", f32))
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
                                0.5 * (1 + T.erf(T.cast(x[i], f32) * 0.7071067811865476))
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
            def kernel(x: T.Tensor((n,), f16), dy: T.Tensor((n,), f16), out: T.Tensor((n,), f16)):
                elementwise(x, out, dy)
        else:

            @T.prim_func
            def kernel(x: T.Tensor((n,), f16), y: T.Tensor((n,), f16), out: T.Tensor((n,), f16)):
                elementwise(x, out, y)

        return kernel

    if kind in ("gemm", "gemm_gelu", "gemm_residual"):
        b, m, k, cols = p["batch"], p["m"], p["k"], p["cols"]
        ta, tb = p.get("ta", False), p.get("tb", False)
        bm, bn, bk, stages = schedule or (32, 64, 32, 2)
        aa, bb = ((bk, bm) if ta else (bm, bk)), ((bn, bk) if tb else (bk, bn))

        @T.macro
        def matmul(x, w, out, extra=None):
            with T.Kernel(T.ceildiv(cols, bn), T.ceildiv(m, bm), b, threads=128) as (bx, by, batch):
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
                            out[index] = 0.5 * value * (1 + T.erf(value * 0.7071067811865476))
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
                out: T.Tensor((b * m * cols,), f16),
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
                            out[i] = T.cast(w[tokens[i // c] * c + i % c], f32) + T.cast(
                                position[(i // c % s) * c + i % c], f32
                            )
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
                            T.atomic_add(dw[tokens[i // c] * c + i % c], T.cast(dy[i], f32))
                            T.atomic_add(dp[(i // c % s) * c + i % c], T.cast(dy[i], f32))

        return kernel

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
                        i < c, (values[i] - total[0] / c) * (values[i] - total[0] / c), 0
                    )
                T.reduce_sum(square, variance, dim=0)
                for i in T.Parallel(tile):
                    if i < c:
                        z = (values[i] - total[0] / c) * T.rsqrt(variance[0] / c + 0.00001)
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
                    values[i] = T.if_then_else(i < c, T.cast(dy[row * c + i], f32) * weight[i], 0)
                    product[i] = T.if_then_else(i < c, values[i] * normal[row * c + i], 0)
                T.reduce_sum(values, total, dim=0)
                T.reduce_sum(product, dot, dim=0)
                for i in T.Parallel(tile):
                    if i < c:
                        dx[row * c + i] = inverse[row] * (
                            values[i] - (total[0] + normal[row * c + i] * dot[0]) / c
                        )
                        parts[row * c + i] = T.cast(dy[row * c + i], f32) * normal[row * c + i]

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
                            T.cast(T.cast(T.cast(scores[row * s + i], f32) * scale, f16), f32),
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
                        parts[(row * chunks + i) * 2 + 1] * T.exp(maxima[i] - maximum[0]),
                        0,
                    )
                T.reduce_sum(sums, total, dim=0)
                lse[row] = maximum[0] + T.log(total[0])
                loss[row] = maximum[0] + T.log(total[0]) - T.cast(x[row * v + target[row]], f32)

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
                            index < n, (x[index] / p["scale"]) * (x[index] / p["scale"]), 0
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
            a("out", p["total"] if kind == "sumsq" else 2 if kind == "clip" else chunks, f32),
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
                            value = weight[i] * (1 - lr * p["decay"]) - lr * (m / correction1) / (
                                T.sqrt(v / correction2) + 0.00000001
                            )
                            moment[i] = m
                            variance[i] = v
                            weight[i] = value
                            half[i] = value

        return kernel
    raise ValueError(f"unknown training kernel {kind}")
