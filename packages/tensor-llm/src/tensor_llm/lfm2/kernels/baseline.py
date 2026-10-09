"""TileLang DSL for packed GGML weights and LFM2 forward execution.

Factories import TileLang only when producing a kernel. Runtime identity and
storage metadata remain usable without the compiler dependencies.
"""

import math
from ...common.artifacts import identity
from ...common.gguf import TYPES


def weight_storage(kind, elements, *, packed_words=False):
    _, block, size = TYPES[kind]
    if kind in (0, 1):
        return elements, "float16" if kind == 1 else "float32"
    count = elements // block * size
    if packed_words:
        if count % 4:
            raise ValueError("WebGPU packed matrices must align to four bytes")
        return count // 4, "uint32"
    return count, "uint8"


def weight_decoder(kind, depth, *, packed_words=False):
    """Inline the exact GGML scalar decoder into a caller's TIRx function."""
    import tilelang.language as T

    _, block, size = TYPES[kind]

    @T.macro
    def byte(w, base, offset):
        if packed_words:
            return (w[(base + offset) // 4] >> ((base + offset) % 4 * 8)) & T.uint32(255)
        else:
            return T.cast(w[base + offset], "uint32")

    @T.macro
    def half(w, base, offset):
        bits = byte(w, base, offset) | (byte(w, base, offset + 1) << 8)
        if packed_words:
            return T.call_extern("float32", "tensor_unpack_f16", bits)
        else:
            return T.cast(T.reinterpret("float16", T.cast(bits, "uint16")), "float32")

    @T.macro
    def signed(value):
        if packed_words:
            return T.cast(
                T.cast(value, "int32") - T.if_then_else(T.cast(value, "int32") >= 128, 256, 0),
                "float32",
            )
        else:
            return T.cast(T.reinterpret("int8", T.cast(value, "uint8")), "float32")

    @T.macro
    def read_weight(w, row, col):
        if kind in (0, 1):
            return T.cast(w[row * depth + col], "float32")
        else:
            base = (row * depth + col) // block * size
            j = col % block
            if kind == 2:
                return half(w, base, 0) * (
                    T.cast((byte(w, base, 2 + j % 16) >> (4 * (j // 16))) & 15, "float32") - 8
                )
            elif kind == 8:
                return half(w, base, 0) * signed(byte(w, base, 2 + j))
            elif kind == 12:
                group = j // 32
                scale = T.if_then_else(
                    group < 4,
                    byte(w, base, 4 + group) & 63,
                    (byte(w, base, 8 + group) & 15) | ((byte(w, base, group) >> 6) << 4),
                )
                minimum = T.if_then_else(
                    group < 4,
                    byte(w, base, 8 + group) & 63,
                    (byte(w, base, 8 + group) >> 4) | ((byte(w, base, 4 + group) >> 6) << 4),
                )
                quant = (byte(w, base, 16 + j // 64 * 32 + j % 32) >> (4 * (group % 2))) & 15
                return half(w, base, 0) * T.cast(scale, "float32") * T.cast(
                    quant, "float32"
                ) - half(w, base, 2) * T.cast(minimum, "float32")
            elif kind == 14:
                group = j % 128 // 32
                low = byte(w, base, j // 128 * 64 + group % 2 * 32 + j % 32)
                high = byte(w, base, 128 + j // 128 * 32 + j % 32)
                quant = ((low >> (4 * (group // 2))) & 15) | (((high >> (2 * group)) & 3) << 4)
                scale = byte(w, base, 192 + j // 16)
                return half(w, base, 208) * signed(scale) * (T.cast(quant, "float32") - 32)
            else:
                raise ValueError("unsupported GGML encoding")

    return read_weight


def source(kind, parameters):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "make_kernel",
        kind,
        parameters,
        dependencies=("tensor_llm.lfm2.kernels.baseline", "tensor_llm.common.gguf", "tensor.compiler.entry"),
    )


def make_kernel(kind, p):
    import tilelang.language as T

    c = p.get("c")
    r = p.get("r", 1)
    if kind == "linear":
        k, o, q = (p["k"], p["o"], p["type"])
        count, dtype = weight_storage(q, k * o)
        read_weight = weight_decoder(q, k)
        _, block, size = TYPES[q]
        if r == 1:

            @T.prim_func
            def kernel(
                x: T.Tensor((r * k,), "float32"),
                w: T.Tensor((count,), dtype),
                out: T.Tensor((r * o,), "float32"),
            ):
                with T.Kernel(T.ceildiv(o, 4), threads=128) as bx:
                    accum = T.alloc_fragment((4, 32), "float32")
                    total = T.alloc_fragment((4,), "float32")
                    T.clear(accum)
                    for tile in T.serial(k // 32):
                        for row, lane in T.Parallel(4, 32):
                            accum[row, lane] += x[tile * 32 + lane] * read_weight(
                                w, bx * 4 + row, tile * 32 + lane
                            )
                    T.reduce_sum(accum, total, dim=1)
                    for row in T.Parallel(4):
                        if bx * 4 + row < o:
                            out[bx * 4 + row] = total[row]

            return kernel

        @T.prim_func
        def kernel(
            x: T.Tensor((r * k,), "float32"),
            w: T.Tensor((count,), dtype),
            out: T.Tensor((r * o,), "float32"),
        ):
            with T.Kernel(T.ceildiv(r, 32), T.ceildiv(o, 64), threads=128) as (by, bx):
                lhs = T.alloc_shared((32, 32), "float16")
                rhs = T.alloc_shared((64, 32), "float16")
                accum = T.alloc_fragment((32, 64), "float32")
                T.clear(accum)
                for tile in T.Pipelined(k // 32, num_stages=2):
                    for i, j in T.Parallel(32, 32):
                        lhs[i, j] = T.if_then_else(
                            by * 32 + i < r, x[(by * 32 + i) * k + tile * 32 + j], 0
                        )
                    for i, j in T.Parallel(64, 32):
                        if bx * 64 + i < o:
                            rhs[i, j] = read_weight(w, bx * 64 + i, tile * 32 + j)
                        else:
                            rhs[i, j] = 0
                    T.gemm(lhs, rhs, accum, transpose_B=True)
                for i, j in T.Parallel(32, 64):
                    if (by * 32 + i < r) & (bx * 64 + j < o):
                        out[(by * 32 + i) * o + bx * 64 + j] = accum[i, j]

        return kernel
    if kind == "rms":

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            w: T.Tensor((c,), "float32"),
            out: T.Tensor((r * c,), "float32"),
        ):
            with T.Kernel(r, threads=256) as row:
                square = T.alloc_fragment((c,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(c):
                    square[i] = x[row * c + i] * x[row * c + i]
                T.reduce_sum(square, total, dim=0)
                for i in T.Parallel(c):
                    out[row * c + i] = x[row * c + i] * T.rsqrt(total[0] / c + p["eps"]) * w[i]

        return kernel
    if kind in ("add", "swiglu"):
        n = r * c

        @T.prim_func
        def kernel(
            x: T.Tensor((n,), "float32"),
            y: T.Tensor((n,), "float32"),
            out: T.Tensor((n,), "float32"),
        ):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    i = block * 256 + lane
                    if i < n:
                        out[i] = x[i] + y[i] if kind == "add" else x[i] / (1 + T.exp(-x[i])) * y[i]

        return kernel
    if kind == "embedding":
        q, v = (p["type"], p["v"])
        count, dtype = weight_storage(q, c * v)
        read_weight = weight_decoder(q, c)
        _, block, size = TYPES[q]

        @T.prim_func
        def kernel(
            tokens: T.Tensor((r,), "int32"),
            w: T.Tensor((count,), dtype),
            out: T.Tensor((r * c,), "float32"),
        ):
            with T.Kernel(T.ceildiv(c, 256), r, threads=256) as (bx, row):
                for lane in T.Parallel(256):
                    col = bx * 256 + lane
                    if col < c:
                        out[row * c + col] = read_weight(w, tokens[row], col)

        return kernel
    if kind == "conv":

        @T.prim_func
        def kernel(
            x: T.Tensor((r * 3 * c,), "float32"),
            w: T.Tensor((c * 3,), "float32"),
            state: T.Tensor((2 * c,), "float32"),
            out: T.Tensor((r * c,), "float32"),
            pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(c, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    col = block * 256 + lane
                    if col < c:
                        left = T.alloc_var("float32")
                        right = T.alloc_var("float32")
                        current = T.alloc_var("float32")
                        left = state[col]
                        right = state[c + col]
                        for row in T.serial(r):
                            if row < pos[1]:
                                current = x[row * (3 * c) + col] * x[row * (3 * c) + 2 * c + col]
                                out[row * c + col] = x[row * (3 * c) + c + col] * (
                                    left * w[col * 3]
                                    + right * w[col * 3 + 1]
                                    + current * w[col * 3 + 2]
                                )
                                left = right
                                right = current
                            else:
                                out[row * c + col] = 0
                        state[col] = left
                        state[c + col] = right

        return kernel
    if kind == "qkv":
        h, kh, d, cap = (p["h"], p["kh"], p["d"], p["cap"])
        eps, theta = (p["eps"], p["theta"])

        @T.prim_func
        def kernel(
            q: T.Tensor((r * h * d,), "float32"),
            k: T.Tensor((r * kh * d,), "float32"),
            v: T.Tensor((r * kh * d,), "float32"),
            qw: T.Tensor((d,), "float32"),
            kw: T.Tensor((d,), "float32"),
            qo: T.Tensor((r * h * d,), "float32"),
            kc: T.Tensor((cap * kh * d,), "float16"),
            vc: T.Tensor((cap * kh * d,), "float16"),
            pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(h, r, threads=64) as (head, row):
                square = T.alloc_fragment((d,), "float32")
                total = T.alloc_fragment((1,), "float32")
                norm = T.alloc_shared((d,), "float32")
                for i in T.Parallel(d):
                    square[i] = q[(row * h + head) * d + i] * q[(row * h + head) * d + i]
                T.reduce_sum(square, total, dim=0)
                for i in T.Parallel(d):
                    norm[i] = q[(row * h + head) * d + i] * T.rsqrt(total[0] / d + eps) * qw[i]
                for i in T.Parallel(d // 2):
                    angle = T.cast(pos[0] + row, "float32") * T.exp(-(math.log(theta) * 2 / d) * i)
                    qo[(row * h + head) * d + i] = norm[i] * T.cos(angle) - norm[
                        i + d // 2
                    ] * T.sin(angle)
                    qo[(row * h + head) * d + i + d // 2] = norm[i] * T.sin(angle) + norm[
                        i + d // 2
                    ] * T.cos(angle)
                if head < kh:
                    for i in T.Parallel(d):
                        square[i] = k[(row * kh + head) * d + i] * k[(row * kh + head) * d + i]
                    T.reduce_sum(square, total, dim=0)
                    for i in T.Parallel(d):
                        norm[i] = k[(row * kh + head) * d + i] * T.rsqrt(total[0] / d + eps) * kw[i]
                        if row < pos[1]:
                            vc[((pos[0] + row) * kh + head) * d + i] = v[(row * kh + head) * d + i]
                    for i in T.Parallel(d // 2):
                        angle = T.cast(pos[0] + row, "float32") * T.exp(
                            -(math.log(theta) * 2 / d) * i
                        )
                        if row < pos[1]:
                            kc[((pos[0] + row) * kh + head) * d + i] = norm[i] * T.cos(
                                angle
                            ) - norm[i + d // 2] * T.sin(angle)
                            kc[((pos[0] + row) * kh + head) * d + i + d // 2] = norm[i] * T.sin(
                                angle
                            ) + norm[i + d // 2] * T.cos(angle)

        return kernel
    if kind == "attention":
        h, kh, d, cap = (p["h"], p["kh"], p["d"], p["cap"])
        if r == 1:

            @T.prim_func
            def kernel(
                q: T.Tensor((r * h * d,), "float32"),
                kc: T.Tensor((cap * kh * d,), "float16"),
                vc: T.Tensor((cap * kh * d,), "float16"),
                out: T.Tensor((r * h * d,), "float32"),
                pos: T.Tensor((2,), "int32"),
            ):
                with T.Kernel(h, threads=128) as head:
                    dots = T.alloc_fragment((64, d), "float32")
                    scores = T.alloc_fragment((64,), "float32")
                    products = T.alloc_fragment((64, d), "float32")
                    partial = T.alloc_fragment((d,), "float32")
                    result = T.alloc_fragment((d,), "float32")
                    maximum = T.alloc_fragment((1,), "float32")
                    previous = T.alloc_fragment((1,), "float32")
                    normalizer = T.alloc_fragment((1,), "float32")
                    total = T.alloc_fragment((1,), "float32")
                    T.fill(maximum, -T.infinity("float32"))
                    T.clear(result)
                    T.clear(normalizer)
                    for tile in T.serial(T.ceildiv(pos[0] + 1, 64)):
                        for i, j in T.Parallel(64, d):
                            dots[i, j] = q[head * d + j] * T.cast(
                                kc[((tile * 64 + i) * kh + head // (h // kh)) * d + j], "float32"
                            )
                        T.reduce_sum(dots, scores, dim=1)
                        T.copy(maximum, previous)
                        for i in T.Parallel(64):
                            scores[i] = T.if_then_else(
                                tile * 64 + i <= pos[0],
                                scores[i] * d ** (-0.5),
                                -T.infinity("float32"),
                            )
                        T.reduce_max(scores, maximum, dim=0, clear=False)
                        for i in T.Parallel(64):
                            scores[i] = T.exp(scores[i] - maximum[0])
                        T.reduce_sum(scores, total, dim=0)
                        normalizer[0] = normalizer[0] * T.exp(previous[0] - maximum[0]) + total[0]
                        for i, j in T.Parallel(64, d):
                            products[i, j] = scores[i] * T.cast(
                                vc[((tile * 64 + i) * kh + head // (h // kh)) * d + j], "float32"
                            )
                        T.reduce_sum(products, partial, dim=0)
                        for j in T.Parallel(d):
                            result[j] = result[j] * T.exp(previous[0] - maximum[0]) + partial[j]
                    for j in T.Parallel(d):
                        out[head * d + j] = result[j] / normalizer[0]

            return kernel

        @T.prim_func
        def kernel(
            q: T.Tensor((r * h * d,), "float32"),
            kc: T.Tensor((cap * kh * d,), "float16"),
            vc: T.Tensor((cap * kh * d,), "float16"),
            out: T.Tensor((r * h * d,), "float32"),
            pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(r, 32), h, threads=128) as (bx, head):
                query = T.alloc_shared((32, d), "float16")
                key = T.alloc_shared((64, d), "float16")
                value = T.alloc_shared((64, d), "float16")
                prob = T.alloc_shared((32, 64), "float16")
                scores = T.alloc_fragment((32, 64), "float32")
                result = T.alloc_fragment((32, d), "float32")
                maximum = T.alloc_fragment((32,), "float32")
                previous = T.alloc_fragment((32,), "float32")
                factor = T.alloc_fragment((32,), "float32")
                normalizer = T.alloc_fragment((32,), "float32")
                total = T.alloc_fragment((32,), "float32")
                for i, j in T.Parallel(32, d):
                    query[i, j] = T.if_then_else(
                        bx * 32 + i < r, q[((bx * 32 + i) * h + head) * d + j], 0
                    )
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity("float32"))
                for tile in T.serial(T.ceildiv(pos[0] + T.min(r, (bx + 1) * 32), 64)):
                    for i, j in T.Parallel(64, d):
                        key[i, j] = kc[((tile * 64 + i) * kh + head // (h // kh)) * d + j]
                        value[i, j] = vc[((tile * 64 + i) * kh + head // (h // kh)) * d + j]
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(32, 64):
                        scores[i, j] = T.if_then_else(
                            tile * 64 + j <= pos[0] + bx * 32 + i,
                            scores[i, j] * d ** (-0.5),
                            -T.infinity("float32"),
                        )
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(32):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(32, 64):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(32):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(32, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
                for i, j in T.Parallel(32, d):
                    if bx * 32 + i < r:
                        out[((bx * 32 + i) * h + head) * d + j] = T.if_then_else(
                            bx * 32 + i < pos[1], result[i, j] / normalizer[i], 0
                        )

        return kernel
    if kind == "last":

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            out: T.Tensor((c,), "float32"),
            pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(c, 256), threads=256) as block:
                for lane in T.Parallel(256):
                    col = block * 256 + lane
                    if col < c:
                        out[col] = x[(pos[1] - 1) * c + col]

        return kernel
    if kind == "advance":

        @T.prim_func
        def kernel(pos: T.Tensor((2,), "int32")):
            with T.Kernel(1, threads=32):
                for i in T.Parallel(1):
                    pos[0] += pos[1]

        return kernel
    raise ValueError(f"unknown LFM2 kernel {kind}")


def argmax_kernel(p, explicit_unroll=False):
    import tilelang.language as T

    n, threads = (p["n"], 256)

    @T.prim_func
    def kernel(
        logits: T.Tensor((n,), "float32"),
        token: T.Tensor((1,), "int32"),
        pos: T.Tensor((2,), "int32"),
    ):
        if explicit_unroll:
            T.func_attr({"tensor.webgpu.loop_unroll": "explicit"})
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            best = T.alloc_var("float32")
            index = T.alloc_var("int32")
            values = T.alloc_shared((threads,), "float32")
            indices = T.alloc_shared((threads,), "int32")
            best = -T.infinity("float32")
            index = n
            for tile in T.serial(T.ceildiv(n, threads)):
                i = tile * threads + tx
                if i < n:
                    if (logits[i] > best) | (logits[i] == best) & (i < index):
                        best = logits[i]
                        index = i
            values[tx] = best
            indices[tx] = index
            T.sync_threads()
            for step in T.unroll(8):
                stride = 128 >> step
                if tx < stride:
                    if (values[tx + stride] > values[tx]) | (values[tx + stride] == values[tx]) & (
                        indices[tx + stride] < indices[tx]
                    ):
                        values[tx] = values[tx + stride]
                        indices[tx] = indices[tx + stride]
                T.sync_threads()
            if tx == 0:
                token[0] = indices[0]
                pos[1] = 1

    return kernel
