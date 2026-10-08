"""Inspectable CUDA training kernels; FP32 reductions and bounded scratch."""

import math


def make_kernel(p):
    import tilelang.language as T

    kind = p["kind"]
    dt = p.get("dtype", "float32")
    n = p.get("n", 1)
    if kind == "gemm":
        m, k, c = p["m"], p["k"], p["c"]
        ta, tb = p["ta"], p["tb"]
        outdt = p["out_dtype"]

        @T.prim_func
        def kernel(
            x: T.Tensor((k, m) if ta else (m, k), dt),
            w: T.Tensor((c, k) if tb else (k, c), dt),
            out: T.Tensor((m, c), outdt),
        ):
            with T.Kernel(T.ceildiv(c, 64), T.ceildiv(m, 32), threads=128) as (bx, by):
                a = T.alloc_shared((32, 32), dt)
                b = T.alloc_shared((32, 64), dt)
                acc = T.alloc_fragment((32, 64), "float32")
                T.clear(acc)
                for tile in T.Pipelined(T.ceildiv(k, 32), num_stages=2):
                    for i, j in T.Parallel(32, 32):
                        a[i, j] = T.if_then_else(
                            (by * 32 + i < m) & (tile * 32 + j < k),
                            x[tile * 32 + j, by * 32 + i]
                            if ta
                            else x[by * 32 + i, tile * 32 + j],
                            0,
                        )
                    for i, j in T.Parallel(32, 64):
                        b[i, j] = T.if_then_else(
                            (tile * 32 + i < k) & (bx * 64 + j < c),
                            w[bx * 64 + j, tile * 32 + i]
                            if tb
                            else w[tile * 32 + i, bx * 64 + j],
                            0,
                        )
                    T.gemm(a, b, acc)
                T.copy(acc, out[by * 32 : (by + 1) * 32, bx * 64 : (bx + 1) * 64])

        return kernel
    if kind in ("rms", "rms_dx", "rms_dw"):
        r, c = p["r"], p["c"]
        width = 1 << (c - 1).bit_length()
        eps = p.get("eps", 1e-6)
        if kind == "rms":

            @T.prim_func
            def kernel(
                x: T.Tensor((r, c), dt),
                w: T.Tensor((c,), "float32"),
                out: T.Tensor((r, c), dt),
                inv: T.Tensor((r,), "float32"),
            ):
                with T.Kernel(r, threads=256) as row:
                    vals = T.alloc_fragment((width,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    for j in T.Parallel(width):
                        vals[j] = T.if_then_else(
                            j < c,
                            T.cast(x[row, j], "float32") * T.cast(x[row, j], "float32"),
                            0,
                        )
                    T.reduce_sum(vals, total, dim=0, clear=True)
                    total[0] = T.rsqrt(total[0] / c + eps)
                    inv[row] = total[0]
                    for j in T.Parallel(width):
                        if j < c:
                            out[row, j] = T.cast(x[row, j], "float32") * total[0] * w[j]
        elif kind == "rms_dx":

            @T.prim_func
            def kernel(
                x: T.Tensor((r, c), dt),
                w: T.Tensor((c,), "float32"),
                dy: T.Tensor((r, c), dt),
                inv: T.Tensor((r,), "float32"),
                out: T.Tensor((r, c), dt),
            ):
                with T.Kernel(r, threads=256) as row:
                    vals = T.alloc_fragment((width,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    for j in T.Parallel(width):
                        vals[j] = T.if_then_else(
                            j < c,
                            T.cast(dy[row, j], "float32")
                            * w[j]
                            * T.cast(x[row, j], "float32"),
                            0,
                        )
                    T.reduce_sum(vals, total, dim=0, clear=True)
                    for j in T.Parallel(width):
                        if j < c:
                            out[row, j] = (
                                T.cast(dy[row, j], "float32") * w[j]
                                - T.cast(x[row, j], "float32")
                                * inv[row]
                                * inv[row]
                                * total[0]
                                / c
                            ) * inv[row]
        else:

            @T.prim_func
            def kernel(
                x: T.Tensor((r, c), dt),
                dy: T.Tensor((r, c), dt),
                inv: T.Tensor((r,), "float32"),
                out: T.Tensor((c,), "float32"),
            ):
                with T.Kernel(T.ceildiv(c, 32), threads=128) as block:
                    vals = T.alloc_fragment((32, 32), "float32")
                    sums = T.alloc_fragment((32,), "float32")
                    accum = T.alloc_fragment((32,), "float32")
                    T.clear(accum)
                    for tile in T.serial(T.ceildiv(r, 32)):
                        for i, j in T.Parallel(32, 32):
                            vals[i, j] = T.if_then_else(
                                (tile * 32 + i < r) & (block * 32 + j < c),
                                T.cast(x[tile * 32 + i, block * 32 + j], "float32")
                                * T.cast(dy[tile * 32 + i, block * 32 + j], "float32")
                                * inv[tile * 32 + i],
                                0,
                            )
                        T.reduce_sum(vals, sums, dim=0, clear=True)
                        for j in T.Parallel(32):
                            accum[j] += sums[j]
                    for j in T.Parallel(32):
                        if block * 32 + j < c:
                            out[block * 32 + j] = accum[j]

        return kernel
    if kind in ("gelu", "gelu_dx", "zero", "scale", "add", "cast"):
        from tensor.compiler.entry import primitive

        @T.macro
        def body(*args):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        if kind == "zero":
                            args[0][i] = 0
                        elif kind == "cast":
                            args[1][i] = args[0][i]
                        elif kind == "add":
                            args[2][i] = T.cast(args[0][i], "float32") + T.cast(
                                args[1][i], "float32"
                            )
                        elif kind == "scale":
                            args[1][i] = T.cast(args[0][i], "float32") * p["factor"]
                        elif kind == "gelu":
                            args[1][i] = (
                                0.5
                                * T.cast(args[0][i], "float32")
                                * (
                                    1
                                    + T.erf(
                                        T.cast(args[0][i], "float32")
                                        * 0.7071067811865476
                                    )
                                )
                            )
                        else:
                            x = T.cast(args[0][i], "float32")
                            args[2][i] = T.cast(args[1][i], "float32") * (
                                0.5 * (1 + T.erf(x * 0.7071067811865476))
                                + x * 0.3989422804014327 * T.exp(-0.5 * x * x)
                            )

        inputs = [] if kind == "zero" else [("x", (n,), dt)]
        if kind in ("add", "gelu_dx"):
            inputs += [("y", (n,), p.get("other_dtype", dt))]
        return primitive(inputs + [("out", (n,), p.get("out_dtype", dt))], body)
    if kind in ("embedding", "embedding_scatter"):
        r, c, v = p["r"], p["c"], p["v"]
        if kind == "embedding":

            @T.prim_func
            def kernel(
                index: T.Tensor((r,), "int64"),
                weight: T.Tensor((v, c), dt),
                out: T.Tensor((r, c), dt),
            ):
                with T.Kernel(T.ceildiv(r * c, 256), threads=256) as block:
                    for lane in T.Parallel(256):
                        i = block * 256 + lane
                        if i < r * c:
                            out[i // c, i % c] = T.if_then_else(
                                (index[i // c] >= 0) & (index[i // c] < v),
                                weight[index[i // c], i % c],
                                float("nan"),
                            )
        else:

            @T.prim_func
            def kernel(
                index: T.Tensor((r,), "int64"),
                dy: T.Tensor((r, c), dt),
                dw: T.Tensor((v, c), "float32"),
            ):
                with T.Kernel(T.ceildiv(r * c, 256), threads=256) as block:
                    for lane in T.Parallel(256):
                        i = block * 256 + lane
                        if i < r * c:
                            if (index[i // c] >= 0) & (index[i // c] < v):
                                T.atomic_add(
                                    dw[index[i // c], i % c],
                                    T.cast(dy[i // c, i % c], "float32"),
                                )

        return kernel
    if kind == "rotary":
        b, h, s, c = p["shape"]
        half = c // 2
        offset = p["offset"]
        base = p["base"]
        sign = p["sign"]

        @T.prim_func
        def kernel(x: T.Tensor((b, h, s, c), dt), out: T.Tensor((b, h, s, c), dt)):
            with T.Kernel(T.ceildiv(b * h * s * c, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < b * h * s * c:
                        j = i % c
                        pos = (i // c) % s
                        head = (i // (s * c)) % h
                        batch = i // (h * s * c)
                        angle = (offset + pos) * T.pow(
                            base, -T.cast(j % half, "float32") / half
                        )
                        pair = j + half if j < half else j - half
                        out[batch, head, pos, j] = T.cast(
                            x[batch, head, pos, j], "float32"
                        ) * T.cos(angle) + sign * (-1 if j < half else 1) * T.cast(
                            x[batch, head, pos, pair], "float32"
                        ) * T.sin(angle)

        return kernel
    if kind in ("ce_parts", "ce_finish", "ce_dx"):
        r, v = p["r"], p["v"]
        chunks = (v + 1023) // 1024
        if kind == "ce_parts":

            @T.prim_func
            def kernel(
                x: T.Tensor((r, v), dt), parts: T.Tensor((r, chunks, 2), "float32")
            ):
                with T.Kernel(chunks, r, threads=256) as (block, row):
                    vals = T.alloc_fragment((1024,), "float32")
                    mx = T.alloc_fragment((1,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    for j in T.Parallel(1024):
                        vals[j] = T.if_then_else(
                            block * 1024 + j < v,
                            T.cast(x[row, block * 1024 + j], "float32"),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(vals, mx, dim=0, clear=True)
                    for j in T.Parallel(1024):
                        vals[j] = T.exp(vals[j] - mx[0])
                    T.reduce_sum(vals, total, dim=0, clear=True)
                    parts[row, block, 0] = mx[0]
                    parts[row, block, 1] = total[0]
        elif kind == "ce_finish":
            width = 1 << (chunks - 1).bit_length()

            @T.prim_func
            def kernel(
                x: T.Tensor((r, v), dt),
                target: T.Tensor((r,), "int64"),
                parts: T.Tensor((r, chunks, 2), "float32"),
                lse: T.Tensor((r,), "float32"),
                loss: T.Tensor((r,), "float32"),
            ):
                with T.Kernel(r, threads=128) as row:
                    vals = T.alloc_fragment((width,), "float32")
                    sums = T.alloc_fragment((width,), "float32")
                    mx = T.alloc_fragment((1,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    for j in T.Parallel(width):
                        vals[j] = T.if_then_else(
                            j < chunks, parts[row, j, 0], -T.infinity("float32")
                        )
                    T.reduce_max(vals, mx, dim=0, clear=True)
                    for j in T.Parallel(width):
                        sums[j] = T.if_then_else(
                            j < chunks, parts[row, j, 1] * T.exp(vals[j] - mx[0]), 0
                        )
                    T.reduce_sum(sums, total, dim=0, clear=True)
                    lse[row] = mx[0] + T.log(total[0])
                    loss[row] = T.if_then_else(
                        target[row] == p["ignore_index"],
                        0,
                        T.if_then_else(
                            (target[row] >= 0) & (target[row] < v),
                            lse[row] - T.cast(x[row, target[row]], "float32"),
                            float("nan"),
                        ),
                    )
        else:

            @T.prim_func
            def kernel(
                x: T.Tensor((r, v), dt),
                target: T.Tensor((r,), "int64"),
                lse: T.Tensor((r,), "float32"),
                dy: T.Tensor((r,), "float32"),
                out: T.Tensor((r, v), dt),
            ):
                with T.Kernel(T.ceildiv(r * v, 256), threads=256) as block:
                    for lane in T.Parallel(256):
                        i = block * 256 + lane
                        if i < r * v:
                            out[i // v, i % v] = T.if_then_else(
                                target[i // v] == p["ignore_index"],
                                0,
                                (
                                    T.exp(
                                        T.cast(x[i // v, i % v], "float32")
                                        - lse[i // v]
                                    )
                                    - T.if_then_else(i % v == target[i // v], 1.0, 0.0)
                                )
                                * dy[i // v],
                            )

        return kernel
    if kind == "ce_reduce":
        r = p["r"]
        width = 1 << (r - 1).bit_length()

        @T.prim_func
        def kernel(
            loss: T.Tensor((r,), "float32"),
            target: T.Tensor((r,), "int64"),
            out: T.Tensor((1,), "float32"),
            count: T.Tensor((1,), "float32"),
        ):
            with T.Kernel(1, threads=256):
                vals = T.alloc_fragment((width,), "float32")
                valid = T.alloc_fragment((width,), "float32")
                total = T.alloc_fragment((1,), "float32")
                num = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(width):
                    vals[j] = T.if_then_else(j < r, loss[j], 0)
                    valid[j] = T.if_then_else(
                        j < r,
                        T.if_then_else(target[j] != p["ignore_index"], 1.0, 0.0),
                        0,
                    )
                T.reduce_sum(vals, total, dim=0, clear=True)
                T.reduce_sum(valid, num, dim=0, clear=True)
                out[0] = total[0] / num[0]
                count[0] = num[0]

        return kernel
    if kind == "ce_seed":
        r = p["r"]

        @T.prim_func
        def kernel(
            seed: T.Tensor((1,), "float32"),
            count: T.Tensor((1,), "float32"),
            out: T.Tensor((r,), "float32"),
        ):
            with T.Kernel(T.ceildiv(r, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < r:
                        out[i] = T.if_then_else(count[0] > 0, seed[0] / count[0], 0)

        return kernel
    if kind in ("sumsq", "sum", "clip"):
        chunks = (n + 1023) // 1024
        width = 1 << (n - 1).bit_length() if kind == "clip" else 1024

        @T.prim_func
        def kernel(
            x: T.Tensor((n,), dt),
            out: T.Tensor((2 if kind == "clip" else chunks,), "float32"),
        ):
            with T.Kernel(1 if kind == "clip" else chunks, threads=256) as block:
                vals = T.alloc_fragment((width,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for j in T.Parallel(width):
                    idx = block * width + j
                    vals[j] = T.if_then_else(
                        idx < n,
                        T.cast(x[idx], "float32")
                        if kind in ("clip", "sum")
                        else T.cast(x[idx], "float32") * T.cast(x[idx], "float32"),
                        0,
                    )
                T.reduce_sum(vals, total, dim=0, clear=True)
                if kind == "clip":
                    out[1] = T.sqrt(total[0])
                    out[0] = T.if_then_else(
                        T.isnan(total[0]) | (total[0] == T.infinity("float32")),
                        T.infinity("float32"),
                        T.min(1.0, p["limit"] / (T.sqrt(total[0]) + 1e-6)),
                    )
                else:
                    out[block] = total[0]

        return kernel
    if kind == "adam_step":

        @T.prim_func
        def kernel(step: T.Tensor((1,), "float32"), clip: T.Tensor((2,), "float32")):
            with T.Kernel(1, threads=32):
                for i in T.Parallel(1):
                    if clip[0] != T.infinity("float32"):
                        step[i] += 1

        return kernel
    if kind == "adamw":

        @T.prim_func
        def kernel(
            weight: T.Tensor((n,), "float32"),
            grad: T.Tensor((n,), "float32"),
            moment: T.Tensor((n,), "float32"),
            variance: T.Tensor((n,), "float32"),
            step: T.Tensor((1,), "float32"),
            clip: T.Tensor((2,), "float32"),
            hyper: T.Tensor((5,), "float32"),
        ):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        if clip[0] != T.infinity("float32"):
                            g = grad[i] * clip[0]
                            mm = hyper[1] * moment[i] + (1 - hyper[1]) * g
                            vv = hyper[2] * variance[i] + (1 - hyper[2]) * g * g
                            moment[i] = mm
                            variance[i] = vv
                            weight[i] = weight[i] * (1 - hyper[0] * hyper[4]) - hyper[
                                0
                            ] * (mm / (1 - T.pow(hyper[1], step[0]))) / (
                                T.sqrt(vv / (1 - T.pow(hyper[2], step[0]))) + hyper[3]
                            )

        return kernel
    raise ValueError("unsupported LLT training kernel " + kind)
