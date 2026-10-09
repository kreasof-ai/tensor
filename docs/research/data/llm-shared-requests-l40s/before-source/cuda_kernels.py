"""TileLang DSL for the measured packed CUDA inference schedules.

All arithmetic and control flow remain visible in TIRx. Runtime constants can
be imported without loading TileLang; factories import the compiler lazily.
"""

from .kernels import make_kernel as baseline_kernel, weight_decoder, weight_storage, argmax_kernel
from .gguf import TYPES

CUDA_PROFILES = ("default", "optimized")
CUDA_GROUPED_THRESHOLD = 4096


def source(kind, parameters):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "make_kernel",
        kind,
        parameters,
        dependencies=("tensor_llm.kernels", "tensor_llm.gguf", "tensor.compiler.entry"),
    )


def gemv_source(kind, parameters):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "gemv_kernel",
        kind,
        parameters,
        dependencies=("tensor_llm.kernels", "tensor_llm.gguf", "tensor.compiler.entry"),
    )


def prefill_source(kind, parameters):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "prefill_kernel",
        kind,
        parameters,
        dependencies=("tensor_llm.kernels", "tensor_llm.gguf", "tensor.compiler.entry"),
    )


def fused_source(kind, parameters):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "fused_kernel",
        kind,
        parameters,
        dependencies=("tensor_llm.kernels", "tensor_llm.gguf", "tensor.compiler.entry"),
    )


def hardware_loads():
    import tilelang.language as T

    @T.macro
    def u16(w, offset):
        return T.call_extern("uint32", "tensor_load_u16", T.address_of(w[offset]))

    @T.macro
    def word(w, offset):
        return u16(w, offset) | (u16(w, offset + 2) << 16)

    @T.macro
    def half(w, offset):
        return T.cast(T.reinterpret("float16", T.cast(u16(w, offset), "uint16")), "float32")

    return u16, word, half


def packed_dot(q, k, row_bytes, width=4):
    """One lane's packed weight dot product, inlined at each GEMV use."""
    import tilelang.language as T

    u16, word, half_value = hardware_loads()

    @T.macro
    def dot_product(x, w, tile, lane, row):
        if q in (0, 1):
            j = tile * (32 * width) + lane * width
            activation = T.alloc_local((width,), "float32")
            coefficient = T.alloc_local((width,), "float16" if q == 1 else "float32")
            for component in T.vectorized(width):
                activation[component] = x[j + component]
            for component in T.vectorized(width):
                coefficient[component] = w[row * k + j + component]
            dot = T.alloc_var("float32")
            dot = 0
            for component in T.unroll(width):
                dot = T.ieee_fmaf(
                    activation[component], T.cast(coefficient[component], "float32"), dot
                )
            return dot
        elif q == 2:
            base = row * row_bytes + (tile * 8 + lane // 4) * 18
            scale = half_value(w, base)
            packed = word(w, base + 2 + lane % 4 * 4)
            j = tile * 256 + lane // 4 * 32 + lane % 4 * 4
            activation = T.alloc_local((4,), "float32")
            activation_hi = T.alloc_local((4,), "float32")
            for component in T.vectorized(4):
                activation[component] = x[j + component]
            for component in T.vectorized(4):
                activation_hi[component] = x[j + 16 + component]
            dot = T.alloc_var("float32")
            dot = 0
            for component in T.unroll(4):
                dot = T.ieee_fmaf(
                    activation[component],
                    T.cast((packed >> (component * 8)) & 15, "float32") - 8,
                    dot,
                )
            for component in T.unroll(4):
                dot = T.ieee_fmaf(
                    activation_hi[component],
                    T.cast((packed >> (component * 8 + 4)) & 15, "float32") - 8,
                    dot,
                )
            dot = scale * dot
            return dot
        elif q == 12:
            base = row * row_bytes + tile * 144
            group = lane // 8 * 2
            packed = word(w, base + 16 + lane // 8 * 32 + lane % 8 * 4)
            j = tile * 256 + lane // 8 * 64 + lane % 8 * 4
            activation = T.alloc_local((4,), "float32")
            activation_hi = T.alloc_local((4,), "float32")
            for component in T.vectorized(4):
                activation[component] = x[j + component]
            for component in T.vectorized(4):
                activation_hi[component] = x[j + 32 + component]
            lo = T.alloc_var("float32")
            hi = T.alloc_var("float32")
            lo = 0
            hi = 0
            for component in T.unroll(4):
                lo = T.ieee_fmaf(
                    activation[component], T.cast((packed >> (component * 8)) & 15, "float32"), lo
                )
                hi = T.ieee_fmaf(
                    activation_hi[component],
                    T.cast((packed >> (component * 8 + 4)) & 15, "float32"),
                    hi,
                )
            s0 = T.if_then_else(
                group < 4,
                w[base + 4 + group] & 63,
                (w[base + 8 + group] & 15) | ((w[base + group] >> 6) << 4),
            )
            s1 = T.if_then_else(
                group < 4,
                w[base + 5 + group] & 63,
                (w[base + 9 + group] & 15) | ((w[base + 1 + group] >> 6) << 4),
            )
            m0 = T.if_then_else(
                group < 4,
                w[base + 8 + group] & 63,
                (w[base + 8 + group] >> 4) | ((w[base + 4 + group] >> 6) << 4),
            )
            m1 = T.if_then_else(
                group < 4,
                w[base + 9 + group] & 63,
                (w[base + 9 + group] >> 4) | ((w[base + 5 + group] >> 6) << 4),
            )
            sum0 = (activation[0] + activation[1]) + (activation[2] + activation[3])
            sum1 = (activation_hi[0] + activation_hi[1]) + (activation_hi[2] + activation_hi[3])
            dot = half_value(w, base) * (
                T.cast(s0, "float32") * lo + T.cast(s1, "float32") * hi
            ) - half_value(w, base + 2) * (
                T.cast(m0, "float32") * sum0 + T.cast(m1, "float32") * sum1
            )
            return dot
        elif q == 14:
            base = row * row_bytes + tile * 210
            dot = T.alloc_var("float32")
            dot = 0
            for half in T.unroll(2):
                low = word(w, base + half * 64 + (lane // 8 % 2) * 32 + lane % 8 * 4)
                high = word(w, base + 128 + half * 32 + lane % 8 * 4)
                activation = T.alloc_local((4,), "float32")
                for component in T.vectorized(4):
                    activation[component] = x[tile * 256 + half * 128 + lane * 4 + component]
                value = T.alloc_var("float32")
                value = 0
                for component in T.unroll(4):
                    quant = T.cast(
                        (low >> (component * 8 + lane // 16 * 4)) & 15, "int32"
                    ) | T.cast(((high >> (component * 8 + lane // 8 * 2)) & 3) << 4, "int32")
                    value = T.ieee_fmaf(activation[component], T.cast(quant - 32, "float32"), value)
                scale = T.cast(
                    T.reinterpret("int8", w[base + 192 + half * 8 + lane // 4]), "float32"
                )
                dot = T.ieee_fmaf(scale, value, dot)
            dot = half_value(w, base + 208) * dot
            return dot
        else:
            raise ValueError("unsupported packed dot encoding")

    return dot_product


def projection_abi(kind, p, algorithm):
    """Select the fused operator ABI without generating Python signatures."""
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    count, dtype = weight_storage(q, k * o)
    if kind == "ffn":

        @T.macro
        def call(x, w, w2, out):
            algorithm(x, w, out, w2)

        return primitive(
            [
                ("x", r * k, "float32"),
                ("w", count, dtype),
                ("w2", count, dtype),
                ("out", r * o, "float32"),
            ],
            call,
        )
    if kind == "linear_add":

        @T.macro
        def call(x, w, residual, out):
            algorithm(x, w, out, None, residual)

        return primitive(
            [
                ("x", r * k, "float32"),
                ("w", count, dtype),
                ("residual", r * o, "float32"),
                ("out", r * o, "float32"),
            ],
            call,
        )
    return primitive(
        [("x", r * k, "float32"), ("w", count, dtype), ("out", r * o, "float32")], algorithm
    )


def gemv_kernel(kind, p):
    import tilelang.language as T

    k, o, q = p["k"], p["o"], p["type"]
    if q not in (0, 1, 2, 12, 14) or k % 256:
        return fused_kernel(kind, p)
    threads, unroll, width = p.get("threads", 128), p.get("unroll", 4), p.get("f16_values", 4)
    if (
        threads not in (64, 128, 256)
        or unroll not in (1, 2, 4, 8)
        or width not in (4, 8, 16)
        or k % (32 * width)
    ):
        raise ValueError("unsupported CUDA packed GEMV schedule")
    _, block, size = TYPES[q]
    row_bytes = k * (4 if q == 0 else 2) if q in (0, 1) else k // block * size
    tiles = k // (32 * width) if q in (0, 1) else k // 256
    paired = kind == "ffn"
    dot_product = packed_dot(q, k, row_bytes, width)

    @T.macro
    def gemv(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(o, threads // 32), threads=threads) as bx:
            tx = T.get_thread_binding()
            lane = tx % 32
            row = bx * (threads // 32) + tx // 32
            sums = T.alloc_local((unroll,), "float32")
            if paired:
                ups = T.alloc_local((unroll,), "float32")
            for slot in T.unroll(unroll):
                sums[slot] = 0
                if paired:
                    ups[slot] = 0
            if row < o:
                for chunk in T.serial(T.ceildiv(tiles, unroll)):
                    for slot in T.unroll(unroll):
                        tile = chunk * unroll + slot
                        if tile < tiles:
                            dot = dot_product(x, w, tile, lane, row)
                            sums[slot] = sums[slot] + dot
                            if paired:
                                dot2 = dot_product(x, w2, tile, lane, row)
                                ups[slot] = ups[slot] + dot2
            total = T.alloc_var("float32")
            total = 0
            if paired:
                up = T.alloc_var("float32")
                up = 0
            for slot in T.unroll(unroll):
                total = total + sums[slot]
                if paired:
                    up = up + ups[slot]
            for step in T.unroll(5):
                total = total + T.shfl_down(total, 16 >> step)
                if paired:
                    up = up + T.shfl_down(up, 16 >> step)
            if (lane == 0) & (row < o):
                if paired:
                    out[row] = total / (1 + T.exp(-total)) * up
                elif kind == "linear_add":
                    out[row] = residual[row] + total
                else:
                    out[row] = total

    return projection_abi(kind, p, gemv)


def pair_decoder(q, k):
    import tilelang.language as T

    u16, word, half_value = hardware_loads()
    _, block, size = TYPES[q]
    row_bytes = k * 2 if q == 1 else k // block * size

    @T.macro
    def pair_load(w, row, col):
        if q == 1:
            return T.ldg32(w[row * k + col])
        elif q == 2:
            base = row * row_bytes + col // 32 * 18
            scale = half_value(w, base)
            packed = u16(w, base + 2 + col % 16)
            shift = col % 32 // 16 * 4
            lo = scale * (T.cast((packed >> shift) & 15, "float32") - 8)
            hi = scale * (T.cast((packed >> (shift + 8)) & 15, "float32") - 8)
            return T.call_extern("uint32", "tensor_pack_f16x2", lo, hi)
        elif q == 12:
            base = row * row_bytes + col // 256 * 144
            weight_col = col % 256
            group = weight_col // 32
            scale = T.if_then_else(
                group < 4,
                w[base + 4 + group] & 63,
                (w[base + 8 + group] & 15) | ((w[base + group] >> 6) << 4),
            )
            minimum = T.if_then_else(
                group < 4,
                w[base + 8 + group] & 63,
                (w[base + 8 + group] >> 4) | ((w[base + 4 + group] >> 6) << 4),
            )
            ds = half_value(w, base) * T.cast(scale, "float32")
            dm = half_value(w, base + 2) * T.cast(minimum, "float32")
            packed = u16(w, base + 16 + weight_col // 64 * 32 + weight_col % 32)
            shift = group % 2 * 4
            lo = ds * T.cast((packed >> shift) & 15, "float32") - dm
            hi = ds * T.cast((packed >> (shift + 8)) & 15, "float32") - dm
            return T.call_extern("uint32", "tensor_pack_f16x2", lo, hi)
        elif q == 14:
            base = row * row_bytes + col // 256 * 210
            weight_col = col % 256
            group = weight_col % 128 // 32
            low = u16(w, base + weight_col // 128 * 64 + group % 2 * 32 + weight_col % 32)
            high = u16(w, base + 128 + weight_col // 128 * 32 + weight_col % 32)
            a = T.cast((low >> (group // 2 * 4)) & 15, "int32") | T.cast(
                ((high >> (group * 2)) & 3) << 4, "int32"
            )
            b = T.cast((low >> (group // 2 * 4 + 8)) & 15, "int32") | T.cast(
                ((high >> (group * 2 + 8)) & 3) << 4, "int32"
            )
            scale = half_value(w, base + 208) * T.cast(
                T.reinterpret("int8", w[base + 192 + weight_col // 16]), "float32"
            )
            return T.call_extern(
                "uint32",
                "tensor_pack_f16x2",
                scale * T.cast(a - 32, "float32"),
                scale * T.cast(b - 32, "float32"),
            )
        else:
            raise ValueError("unsupported packed pair encoding")

    return pair_load


def prefill_kernel(kind, p):
    """FP16 tensor-core operands, FP32 accumulation, and paired FFN tiles."""
    import tilelang.language as T

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    bm, bn, bk = p.get("block_m", 32), p.get("block_n", 64), p.get("block_k", 32)
    stages, threads = p.get("stages", 2), p.get("threads", 128)
    paired, packed = kind == "ffn", p.get("packed_pairs", False)
    read_weight = weight_decoder(q, k)
    read_pair = pair_decoder(q, k) if packed else None

    @T.macro
    def load_rhs(w, rhs, bx, tile, i, j):
        if packed:
            if bx * bn + i < o:
                bits = read_pair(w, bx * bn + i, tile * bk + j * 2)
                rhs[i, j * 2] = T.reinterpret(
                    "float16", T.cast(bits & T.uint32(65535), "uint16")
                )
                rhs[i, j * 2 + 1] = T.reinterpret("float16", T.cast(bits >> 16, "uint16"))
            else:
                rhs[i, j * 2] = 0
                rhs[i, j * 2 + 1] = 0
        else:
            if bx * bn + i < o:
                rhs[i, j] = read_weight(w, bx * bn + i, tile * bk + j)
            else:
                rhs[i, j] = 0

    @T.macro
    def matmul(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(r, bm), T.ceildiv(o, bn), threads=threads) as (by, bx):
            lhs = T.alloc_shared((bm, bk), "float16")
            rhs = T.alloc_shared((bn, bk), "float16")
            accum = T.alloc_fragment((bm, bn), "float32")
            T.clear(accum)
            if paired:
                rhs2 = T.alloc_shared((bn, bk), "float16")
                accum2 = T.alloc_fragment((bm, bn), "float32")
                T.clear(accum2)
            for tile in T.Pipelined(k // bk, num_stages=stages):
                for i, j in T.Parallel(bm, bk):
                    lhs[i, j] = T.if_then_else(
                        by * bm + i < r, x[(by * bm + i) * k + tile * bk + j], 0
                    )
                # Keep paired FFN operands in the same staging iteration. Splitting
                # the loops changes the pipelined loader's layout and scheduling.
                for i, j in T.Parallel(bn, bk // 2 if packed else bk):
                    load_rhs(w, rhs, bx, tile, i, j)
                    if paired:
                        load_rhs(w2, rhs2, bx, tile, i, j)
                T.gemm(lhs, rhs, accum, transpose_B=True)
                if paired:
                    T.gemm(lhs, rhs2, accum2, transpose_B=True)
            for i, j in T.Parallel(bm, bn):
                if (by * bm + i < r) & (bx * bn + j < o):
                    if paired:
                        out[(by * bm + i) * o + bx * bn + j] = (
                            accum[i, j] / (1 + T.exp(-accum[i, j])) * accum2[i, j]
                        )
                    elif kind == "linear_add":
                        out[(by * bm + i) * o + bx * bn + j] = (
                            residual[(by * bm + i) * o + bx * bn + j] + accum[i, j]
                        )
                    else:
                        out[(by * bm + i) * o + bx * bn + j] = accum[i, j]

    return projection_abi(kind, p, matmul)


def fused_kernel(kind, p):
    import tilelang.language as T

    if kind == "linear":
        return baseline_kernel("linear", p)
    if p["r"] > 1:
        return prefill_kernel(
            kind,
            {
                **p,
                "block_m": 32,
                "block_n": 64,
                "block_k": 32,
                "stages": 2,
                "threads": 128,
                "packed_pairs": False,
            },
        )
    k, o, q = p["k"], p["o"], p["type"]
    paired = kind == "ffn"
    read_weight = weight_decoder(q, k)

    @T.macro
    def gemv(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(o, 4), threads=128) as bx:
            accum = T.alloc_fragment((4, 32), "float32")
            if paired:
                accum2 = T.alloc_fragment((4, 32), "float32")
            total = T.alloc_fragment((4,), "float32")
            if paired:
                total2 = T.alloc_fragment((4,), "float32")
            T.clear(accum)
            if paired:
                T.clear(accum2)
            for tile in T.serial(k // 32):
                for row, lane in T.Parallel(4, 32):
                    accum[row, lane] += x[tile * 32 + lane] * read_weight(
                        w, bx * 4 + row, tile * 32 + lane
                    )
                    if paired:
                        accum2[row, lane] += x[tile * 32 + lane] * read_weight(
                            w2, bx * 4 + row, tile * 32 + lane
                        )
            T.reduce_sum(accum, total, dim=1)
            if paired:
                T.reduce_sum(accum2, total2, dim=1)
            for row in T.Parallel(4):
                if bx * 4 + row < o:
                    if paired:
                        out[bx * 4 + row] = total[row] / (1 + T.exp(-total[row])) * total2[row]
                    else:
                        out[bx * 4 + row] = residual[bx * 4 + row] + total[row]

    return projection_abi(kind, p, gemv)


def warp_partial_kernel(p, splits):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    h, kh, d, cap = (p[n] for n in ("h", "kh", "d", "cap"))
    if d != 64 or type(splits) is not int or splits < 1:
        raise ValueError("unsupported warp attention schedule")

    @T.macro
    def algorithm(q, kc, vc, parts, pos):
        with T.Kernel(h, splits, threads=128) as (head, split):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            chunk = T.ceildiv(pos[0] + 1, splits)
            begin = split * chunk
            end = T.min(begin + chunk, pos[0] + 1)
            kvhead = head // (h // kh)
            q0 = q[head * 64 + lane]
            q1 = q[head * 64 + lane + 32]
            maxima = T.alloc_shared((4,), "float32")
            sums = T.alloc_shared((4,), "float32")
            values = T.alloc_shared((4, 64), "float32")
            maximum = T.alloc_var("float32")
            normalizer = T.alloc_var("float32")
            a0 = T.alloc_var("float32")
            a1 = T.alloc_var("float32")
            maximum = -T.infinity("float32")
            normalizer = 0
            a0 = 0
            a1 = 0
            for offset in T.serial(T.ceildiv(T.max(end - begin - warp, 0), 4)):
                token = begin + warp + offset * 4
                base = (token * kh + kvhead) * 64
                score = q0 * T.cast(kc[base + lane], "float32") + q1 * T.cast(
                    kc[base + lane + 32], "float32"
                )
                score = score + T.shfl_down(score, 16)
                score = score + T.shfl_down(score, 8)
                score = score + T.shfl_down(score, 4)
                score = score + T.shfl_down(score, 2)
                score = score + T.shfl_down(score, 1)
                score = T.shfl_sync(score, 0) * 0.125
                next_max = T.max(maximum, score)
                correction = T.exp(maximum - next_max)
                probability = T.exp(score - next_max)
                a0 = a0 * correction + probability * T.cast(vc[base + lane], "float32")
                a1 = a1 * correction + probability * T.cast(vc[base + lane + 32], "float32")
                normalizer = normalizer * correction + probability
                maximum = next_max
            values[warp, lane] = a0
            values[warp, lane + 32] = a1
            if lane == 0:
                maxima[warp] = maximum
                sums[warp] = normalizer
            T.sync_threads()
            if warp == 0:
                merged_max = T.max(T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3]))
                total = T.alloc_var("float32")
                result0 = T.alloc_var("float32")
                result1 = T.alloc_var("float32")
                total = 0
                result0 = 0
                result1 = 0
                for other in T.unroll(4):
                    factor = T.if_then_else(
                        T.isfinite(merged_max), T.exp(maxima[other] - merged_max), 0
                    )
                    total = total + sums[other] * factor
                    result0 = result0 + values[other, lane] * factor
                    result1 = result1 + values[other, lane + 32] * factor
                base = (head * splits + split) * 66
                parts[base + lane] = result0
                parts[base + lane + 32] = result1
                if lane == 0:
                    parts[base + 64] = merged_max
                    parts[base + 65] = total

    return primitive(
        [
            ("q", h * d, "float32"),
            ("kc", cap * kh * d, "float16"),
            ("vc", cap * kh * d, "float16"),
            ("parts", h * splits * 66, "float32"),
            ("pos", 2, "int32"),
        ],
        algorithm,
    )


def warp_partial_source(p, splits):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "warp_partial_kernel",
        p,
        splits,
        dependencies=("tensor_llm.kernels", "tensor_llm.gguf", "tensor.compiler.entry"),
    )


def grouped_kernel(p, stage=32, warps=2):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    h, kh, d, cap, splits = (p[n] for n in ("h", "kh", "d", "cap", "splits"))
    group = h // kh
    threads = group * warps * 32
    if (
        d != 64
        or h % kh
        or group not in (2, 4)
        or (stage not in (16, 32, 64))
        or (warps not in (2, 4))
    ):
        raise ValueError("unsupported grouped attention schedule")

    @T.macro
    def algorithm(q, kc, vc, parts, pos):
        with T.Kernel(kh, splits, threads=threads) as (kvhead, split):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            local = warp // warps
            worker = warp % warps
            head = kvhead * group + local
            chunk = T.ceildiv(pos[0] + 1, splits)
            begin = split * chunk
            end = T.min(begin + chunk, pos[0] + 1)
            keys = T.alloc_shared((stage, 64), "float16")
            vals = T.alloc_shared((stage, 64), "float16")
            maxima = T.alloc_shared((group * warps,), "float32")
            sums = T.alloc_shared((group * warps,), "float32")
            values = T.alloc_shared((group * warps, 64), "float32")
            q0 = q[head * 64 + lane]
            q1 = q[head * 64 + lane + 32]
            maximum = T.alloc_var("float32")
            normalizer = T.alloc_var("float32")
            a0 = T.alloc_var("float32")
            a1 = T.alloc_var("float32")
            maximum = -T.infinity("float32")
            normalizer = 0
            a0 = 0
            a1 = 0
            for block in T.serial(T.ceildiv(T.max(end - begin, 0), stage)):
                base = begin + block * stage
                for token, col in T.Parallel(stage, 64):
                    keys[token, col] = T.if_then_else(
                        base + token < end, kc[((base + token) * kh + kvhead) * 64 + col], 0
                    )
                    vals[token, col] = T.if_then_else(
                        base + token < end, vc[((base + token) * kh + kvhead) * 64 + col], 0
                    )
                T.sync_threads()
                for offset in T.serial(
                    T.ceildiv(T.max(T.min(stage, end - base) - worker, 0), warps)
                ):
                    token = worker + offset * warps
                    score = q0 * T.cast(keys[token, lane], "float32") + q1 * T.cast(
                        keys[token, lane + 32], "float32"
                    )
                    score = score + T.shfl_down(score, 16)
                    score = score + T.shfl_down(score, 8)
                    score = score + T.shfl_down(score, 4)
                    score = score + T.shfl_down(score, 2)
                    score = score + T.shfl_down(score, 1)
                    score = T.shfl_sync(score, 0) * 0.125
                    next_max = T.max(maximum, score)
                    correction = T.exp(maximum - next_max)
                    probability = T.exp(score - next_max)
                    a0 = a0 * correction + probability * T.cast(vals[token, lane], "float32")
                    a1 = a1 * correction + probability * T.cast(vals[token, lane + 32], "float32")
                    normalizer = normalizer * correction + probability
                    maximum = next_max
                T.sync_threads()
            values[warp, lane] = a0
            values[warp, lane + 32] = a1
            if lane == 0:
                maxima[warp] = maximum
                sums[warp] = normalizer
            T.sync_threads()
            if worker == 0:
                merged_max = T.alloc_var("float32")
                total = T.alloc_var("float32")
                result0 = T.alloc_var("float32")
                result1 = T.alloc_var("float32")
                merged_max = -T.infinity("float32")
                total = 0
                result0 = 0
                result1 = 0
                for other in T.unroll(warps):
                    merged_max = T.max(merged_max, maxima[local * warps + other])
                for other in T.unroll(warps):
                    index = local * warps + other
                    factor = T.if_then_else(
                        T.isfinite(merged_max), T.exp(maxima[index] - merged_max), 0
                    )
                    total = total + sums[index] * factor
                    result0 = result0 + values[index, lane] * factor
                    result1 = result1 + values[index, lane + 32] * factor
                base = (head * splits + split) * 66
                parts[base + lane] = result0
                parts[base + lane + 32] = result1
                if lane == 0:
                    parts[base + 64] = merged_max
                    parts[base + 65] = total

    return primitive(
        [
            ("q", h * d, "float32"),
            ("kc", cap * kh * d, "float16"),
            ("vc", cap * kh * d, "float16"),
            ("parts", h * splits * 66, "float32"),
            ("pos", 2, "int32"),
        ],
        algorithm,
    )


def grouped_source(p, stage=32, warps=2):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "grouped_kernel",
        p,
        stage,
        warps,
        dependencies=("tensor_llm.kernels", "tensor_llm.gguf", "tensor.compiler.entry"),
    )


def merge_kernel(p, splits):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    h, d = (p["h"], p["d"])
    stride = d + 2

    @T.macro
    def algorithm(parts, out):
        with T.Kernel(h, threads=128) as head:
            maxima = T.alloc_fragment((splits,), "float32")
            maximum = T.alloc_fragment((1,), "float32")
            sums = T.alloc_fragment((splits,), "float32")
            normalizer = T.alloc_fragment((1,), "float32")
            products = T.alloc_fragment((splits, d), "float32")
            result = T.alloc_fragment((d,), "float32")
            for i in T.Parallel(splits):
                maxima[i] = parts[(head * splits + i) * stride + d]
            T.reduce_max(maxima, maximum, dim=0)
            for i in T.Parallel(splits):
                sums[i] = parts[(head * splits + i) * stride + (d + 1)] * T.exp(
                    maxima[i] - maximum[0]
                )
            T.reduce_sum(sums, normalizer, dim=0)
            for i, j in T.Parallel(splits, d):
                products[i, j] = parts[(head * splits + i) * stride + j] * T.exp(
                    maxima[i] - maximum[0]
                )
            T.reduce_sum(products, result, dim=0)
            for j in T.Parallel(d):
                out[head * d + j] = result[j] / normalizer[0]

    return primitive(
        [("parts", h * splits * stride, "float32"), ("out", h * d, "float32")], algorithm
    )


def merge_source(p, splits):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "merge_kernel",
        p,
        splits,
        dependencies=("tensor_llm.kernels", "tensor_llm.gguf", "tensor.compiler.entry"),
    )


def normalization_kernel(kind, p):
    import math
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    r, h, kh, d, cap = (p[n] for n in ("r", "h", "kh", "d", "cap"))
    heads = h if kind == "qnorm" else kh
    eps, angle_scale = p["eps"], math.log(p["theta"]) * 2 / d

    @T.macro
    def normalize(x, w, out, pos, v=None, vc=None):
        with T.Kernel(heads, r, threads=64) as (head, row):
            square = T.alloc_fragment((d,), "float32")
            total = T.alloc_fragment((1,), "float32")
            norm = T.alloc_shared((d,), "float32")
            for i in T.Parallel(d):
                square[i] = x[(row * heads + head) * d + i] * x[(row * heads + head) * d + i]
            T.reduce_sum(square, total, dim=0)
            for i in T.Parallel(d):
                norm[i] = x[(row * heads + head) * d + i] * T.rsqrt(total[0] / d + eps) * w[i]
                if kind == "kvnorm":
                    if row < pos[1]:
                        vc[((pos[0] + row) * kh + head) * d + i] = v[(row * kh + head) * d + i]
            for i in T.Parallel(d // 2):
                angle = T.cast(pos[0] + row, "float32") * T.exp(-angle_scale * i)
                if kind == "qnorm":
                    out[(row * heads + head) * d + i] = norm[i] * T.cos(angle) - norm[
                        i + d // 2
                    ] * T.sin(angle)
                    out[(row * heads + head) * d + i + d // 2] = norm[i] * T.sin(angle) + norm[
                        i + d // 2
                    ] * T.cos(angle)
                else:
                    if row < pos[1]:
                        out[((pos[0] + row) * heads + head) * d + i] = norm[i] * T.cos(
                            angle
                        ) - norm[i + d // 2] * T.sin(angle)
                        out[((pos[0] + row) * heads + head) * d + i + d // 2] = norm[i] * T.sin(
                            angle
                        ) + norm[i + d // 2] * T.cos(angle)

    if kind == "qnorm":
        return primitive(
            [
                ("q", r * h * d, "float32"),
                ("qw", d, "float32"),
                ("qo", r * h * d, "float32"),
                ("pos", 2, "int32"),
            ],
            normalize,
        )

    @T.macro
    def call(k, v, kw, kc, vc, pos):
        normalize(k, kw, kc, pos, v, vc)

    return primitive(
        [
            ("k", r * kh * d, "float32"),
            ("v", r * kh * d, "float32"),
            ("kw", d, "float32"),
            ("kc", cap * kh * d, "float16"),
            ("vc", cap * kh * d, "float16"),
            ("pos", 2, "int32"),
        ],
        call,
    )


def make_kernel(kind, p):
    import tilelang.language as T

    if kind in ("linear", "linear_add", "ffn"):
        if p["r"] == 1:
            return gemv_kernel(kind, p)
        return prefill_kernel(kind, p) if "block_m" in p else fused_kernel(kind, p)
    if kind == "attention_partial":
        return warp_partial_kernel(p, p["splits"])
    if kind == "attention_grouped":
        return grouped_kernel(p, p.get("stage", 32), p.get("warps", 2))
    if kind == "attention_merge":
        return merge_kernel(p, p["splits"])
    if kind in ("qnorm", "kvnorm"):
        return normalization_kernel(kind, p)
    if kind == "argmax":
        return argmax_kernel(p)
    if kind == "prefill_tail":
        r, c, t = p["r"], p["c"], p["t"]

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            out: T.Tensor((t * c,), "float32"),
            pos: T.Tensor((2,), "int32"),
            tail_pos: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(t * c, 256), threads=256) as bx:
                for lane in T.Parallel(256):
                    i = bx * 256 + lane
                    if i < t * c:
                        out[i] = T.if_then_else(
                            i // c < T.min(pos[1], t),
                            x[(T.max(pos[1] - t, 0) + i // c) * c + i % c],
                            0,
                        )
                    if (bx == 0) & (lane == 0):
                        tail_pos[0] = pos[0] + T.max(pos[1] - t, 0)
                        tail_pos[1] = T.min(pos[1], t)

        return kernel
    if kind == "add_rms":
        r, c, eps = p["r"], p["c"], p["eps"]

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            mixed: T.Tensor((r * c,), "float32"),
            residual: T.Tensor((r * c,), "float32"),
            w: T.Tensor((c,), "float32"),
            out: T.Tensor((r * c,), "float32"),
        ):
            with T.Kernel(r, threads=256) as row:
                square = T.alloc_fragment((c,), "float32")
                total = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(c):
                    square[i] = (x[row * c + i] + mixed[row * c + i]) * (
                        x[row * c + i] + mixed[row * c + i]
                    )
                T.reduce_sum(square, total, dim=0)
                for i in T.Parallel(c):
                    residual[row * c + i] = x[row * c + i] + mixed[row * c + i]
                    out[row * c + i] = (
                        (x[row * c + i] + mixed[row * c + i]) * T.rsqrt(total[0] / c + eps) * w[i]
                    )

        return kernel
    return baseline_kernel(kind, p)
