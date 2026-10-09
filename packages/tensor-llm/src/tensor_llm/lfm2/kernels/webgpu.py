"""TileLang DSL for packed portable and standard-subgroup LFM2 schedules."""

import math
from .baseline import make_kernel as baseline_kernel, weight_decoder, weight_storage


def source(kind, parameters):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "make_kernel",
        kind,
        parameters,
        dependencies=(
            "tensor_llm.lfm2.kernels.baseline",
            "tensor_llm.common.gguf",
            "tensor.compiler.entry",
            "tensor.compiler.webgpu_templates",
        ),
    )


def half_bits():
    import tilelang.language as T

    @T.macro
    def unpack(bits):
        return T.call_extern("float32", "tensor_unpack_f16", T.cast(bits, "uint32"))

    return unpack


def packed_loads():
    import tilelang.language as T

    unpack = half_bits()

    @T.macro
    def byte(w, address):
        return (w[address // 4] >> (address % 4 * 8)) & T.uint32(255)

    @T.macro
    def half(w, address):
        return unpack((w[address // 4] >> (address % 4 * 8)) & T.uint32(65535))

    @T.macro
    def word(w, address):
        # The 18-byte GGML Q4 blocks alternate between 2- and 4-byte alignment.
        return T.call_extern(
            "uint32",
            "tensor_byte_align_u32",
            w[address // 4],
            w[(address + 3) // 4],
            T.cast(address % 4, "uint32"),
        )

    return byte, half, word


def round_half():
    """Exact ties-to-even FP16 rounding, including FP16 subnormals.

    Signed exponent comparisons avoid costly unsigned division in WGSL lowering.
    This operation is shared by prefill operand staging and cache stores.
    """
    import tilelang.language as T

    @T.macro
    def rounded(value):
        bits = T.reinterpret("uint32", value)
        exponent = T.cast((bits >> 23) & 255, "int32")
        shift = T.min(T.max(126 - exponent, 1), 24)
        mantissa = (bits & 8388607) | 8388608
        rounding = (
            mantissa + (T.uint32(1) << (shift - 1)) - 1 + ((mantissa >> shift) & 1)
        ) >> shift
        quantum = T.reinterpret("float32", T.uint32(103 << 23))
        small = T.if_then_else(
            exponent < 102, 0.0, T.cast(rounding, "float32") * quantum
        ) * T.if_then_else((bits >> 31) != 0, -1.0, 1.0)
        normal = T.reinterpret("float32", (bits + 4095 + ((bits >> 13) & 1)) & T.uint32(4294959104))
        return T.if_then_else(exponent < 113, small, normal)

    return rounded


def tree_reduce(size, operation="sum"):
    import tilelang.language as T

    def reduce(scratch, tx):
        for stride in (size >> i for i in range(1, size.bit_length())):
            step(scratch, tx, stride)

    @T.macro
    def step(scratch, tx, stride):
        if tx < stride:
            if operation == "max":
                scratch[tx] = T.max(scratch[tx], scratch[tx + stride])
            else:
                scratch[tx] = scratch[tx] + scratch[tx + stride]
        T.sync_threads()

    return reduce


def subgroup_reduce(size, operation="sum"):
    import tilelang.language as T

    builtin = "subgroupMax" if operation == "max" else "subgroupAdd"

    @T.macro
    def reduce(scratch, tx, value: T.Ref):
        scratch[tx] = T.call_extern("float32", builtin, value)
        T.sync_threads()
        if tx == 0:
            value = -T.infinity("float32") if operation == "max" else 0
            for group in T.serial(
                T.ceildiv(size, T.cast(T.call_extern("uint32", "tensor_subgroup_size"), "int32"))
            ):
                i = group * T.cast(T.call_extern("uint32", "tensor_subgroup_size"), "int32")
                if operation == "max":
                    value = T.max(value, scratch[i])
                else:
                    value = value + scratch[i]
            scratch[0] = value
        T.sync_threads()

    return reduce


def rms_width(c):
    for width in (256, 128, 64):
        if c % width == 0:
            return width
    return 64


def projection_epilogue(kind):
    import tilelang.language as T

    @T.macro
    def store(out, residual, index, gate, up):
        if kind in ("ffn", "ffn_q16"):
            out[index] = gate / (1 + T.exp(-gate)) * up
        elif kind == "linear_add":
            out[index] = residual[index] + gate
        else:
            out[index] = gate

    return store


def projection_abi(kind, p, algorithm):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    r, k, o, q = p.get("r", 1), p["k"], p["o"], p["type"]
    count, dtype = weight_storage(q, k * o, packed_words=True)
    args = [("x", r * k, "float32"), ("w", count, dtype)]
    if kind == "ffn":

        @T.macro
        def call(x, w, w2, out):
            algorithm(x, w, out, w2)

        return primitive([*args, ("w2", count, dtype), ("out", r * o, "float32")], call)
    if kind == "linear_add":

        @T.macro
        def call(x, w, residual, out):
            algorithm(x, w, out, None, residual)

        return primitive([*args, ("residual", r * o, "float32"), ("out", r * o, "float32")], call)
    return primitive([*args, ("out", r * o, "float32")], algorithm)


def prefill_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.webgpu_templates import (
        register_matmul,
        partitioned_matmul,
        outer_product_matmul,
    )

    r, k, o, q = (p[n] for n in ("r", "k", "o", "type"))
    parts = 2 if kind == "ffn" else 1
    tm, tn, bk = p.get("tile", (16, 32, 64))
    rounded, decoder = round_half(), weight_decoder(q, k, packed_words=True)
    epilogue = projection_epilogue(kind)

    @T.macro
    def read_rhs(w, col, channel):
        if q == 1:
            return decoder(w, col, channel)
        else:
            return rounded(decoder(w, col, channel))

    if p.get("schedule") == "outer":
        if q not in (1, 2, 14):
            raise ValueError("outer-product prefill requires F16, Q4_0 or Q6_K weights")
        dtype = p.get("outer_shared_dtype", "float16")
        algorithm = outer_product_matmul(
            r, k, o, rounded, read_rhs, epilogue, parts=parts, dtype=dtype, **p["outer"]
        )
    elif p.get("schedule") == "partitioned":
        if q != 1:
            raise ValueError("searched partitioned profile requires F16 weights")
        config = {
            name: p[name]
            for name in ("threads", "partitions", "unroll", "dot_width", "owner_axis", "k_layout")
        }
        algorithm = partitioned_matmul(
            r,
            k,
            o,
            rounded,
            read_rhs,
            epilogue,
            parts=parts,
            tile_m=tm,
            tile_n=tn,
            explicit_unroll=p.get("explicit_unroll", False),
            **config,
        )
    else:
        algorithm = register_matmul(
            r,
            k,
            o,
            rounded,
            read_rhs,
            epilogue,
            parts=parts,
            tile_m=tm,
            tile_n=tn,
            tile_k=bk,
            threads=p.get("threads", 128),
            pad=p.get("pad", 0),
            lhs_pad=p.get("lhs_pad", 0),
            lhs_transpose=p.get("lhs_transpose", False),
            dot_width=p.get("dot_width", 4 if q == 1 or k >= 2048 else 1),
            unroll=p.get("unroll", False),
            explicit_unroll=p.get("explicit_unroll", False),
        )
    return projection_abi(kind, p, algorithm)


def decode_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.webgpu_templates import registers, shared_buffers, streamed_gemv

    k, o, q = p["k"], p["o"], p["type"]
    parts = 2 if kind == "ffn" else 1
    epilogue = projection_epilogue(kind)
    if p.get("decode_schedule") == "streamed":
        if q != 1:
            raise ValueError("streamed GEMV requires native F16 weights")
        config = {
            name: p[name]
            for name in (
                "lanes",
                "threads",
                "micro_rows",
                "dot_width",
                "unroll",
                "accumulators",
                "k_layout",
                "shared_input",
            )
        }
        return projection_abi(kind, p, streamed_gemv(k, o, epilogue, parts=parts, **config))
    byte, half_value, word_value = packed_loads()
    decoder = weight_decoder(q, k, packed_words=True)
    wide = q == 2 and p.get("sg") and k % 256 == 0 and min(k, o) >= 2048
    lanes, threads, chains = (16 if q == 2 else 32), 128, 1
    fast_q4 = q == 2 and k % 128 == 0
    if fast_q4:
        lanes, threads, chains = (
            p.get("gemv_lanes", 32 if wide else 16),
            p.get("gemv_threads", 128),
            p.get("gemv_accumulators", 1),
        )
        if (
            lanes not in (8, 16, 32, 64)
            or threads not in (64, 128, 256, 512)
            or threads % lanes
            or chains not in (1, 4)
            or k % (lanes * 8)
        ):
            raise ValueError("unsupported packed Q4 decode schedule")
    if q == 14:
        threads = p.get("gemv_threads", 128)
        if threads not in (64, 128, 256, 512):
            raise ValueError("invalid Q6 decode workgroup")
    row_count = threads // lanes
    tiles = (
        k // (lanes * 8)
        if fast_q4
        else k // 256
        if q == 14
        else k // 128
        if q in (0, 1) and k % 128 == 0
        else k // 32
    )
    unroll, independent = p.get("gemv_unroll", 1), p.get("gemv_chains", 1)
    dot_q4, dot_q6 = fast_q4 and p.get("gemv_dot", wide), q == 14 and p.get("gemv_q6_dot")
    if dot_q4 and chains != 1:
        raise ValueError("packed dot schedule uses one accumulator")
    if unroll != 1 or independent != 1:
        if (
            q not in (2, 14)
            or type(unroll) is not int
            or unroll not in (1, 2, 4, 8, 16)
            or type(independent) is not int
            or independent not in (1, 2, 4, 8)
            or independent > unroll
            or chains != 1
            or tiles % unroll
        ):
            raise ValueError("invalid packed decode unroll or accumulator chains")
        chains = independent

    def q4_terms(acc, x, index, scale, packed, slot):
        if dot_q4:
            for component in range(2):
                vector_term(acc[slot][0], x, index, scale, packed, component)
        else:
            for component in range(4):
                chain = component if p.get("gemv_accumulators", 1) == 4 else slot
                scalar_term(acc[chain][0], x, index + component, scale, packed, component * 8, 0)
                scalar_term(
                    acc[chain][0], x, index + component, scale, packed, component * 8 + 4, 16
                )

    def q6_terms(acc, x, w, base, tile, lane, slot, scale):
        if dot_q6:
            for half in range(2):
                q6_vector(acc[slot][0], x, w, base, tile, lane, half, scale)
        else:
            for group in range(8):
                q6_scalar(acc[slot][0], x, w, base, tile, lane, group, scale)

    def compute_slots(acc, x, weights, chunk, lane, row, bx):
        for slot in range(unroll):
            for part in range(parts):
                compute(
                    {j: acc[part, j] for j in range(chains)},
                    x,
                    weights[part],
                    chunk * unroll + slot,
                    lane,
                    row,
                    bx,
                    slot % independent,
                )

    def combine(acc, totals):
        for part in range(parts):
            if chains == 4:
                combine4(totals[part,][0], *(acc[part, j][0] for j in range(4)))
            else:
                for j in range(chains):
                    add(totals[part,][0], acc[part, j][0])

    def shuffles(totals):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            for part in range(parts):
                shuffle(totals[part,][0], stride)

    def spills(totals, scratch, tx):
        for part in range(parts):
            spill(scratch[part], totals[part,][0], tx)

    def steps(scratch, tx, stride):
        for part in range(parts):
            shared_step(scratch[part], tx, stride)

    def reduction(scratch, tx, lane):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            reduction_step(scratch, tx, lane, stride)

    @T.macro
    def scalar_term(acc: T.Ref, x, index, scale, packed, shift, offset):
        acc = acc + x[index + offset] * (scale * (T.cast((packed >> shift) & 15, "float32") - 8))

    @T.macro
    def vector_term(acc: T.Ref, x, index, scale, packed, component):
        left = T.call_extern(
            "float32x4",
            "vec4<f32>",
            x[index + component * 16],
            x[index + component * 16 + 1],
            x[index + component * 16 + 2],
            x[index + component * 16 + 3],
        )
        right = T.call_extern(
            "float32x4",
            "vec4<f32>",
            T.cast((packed >> (component * 4)) & 15, "float32") - 8,
            T.cast((packed >> (8 + component * 4)) & 15, "float32") - 8,
            T.cast((packed >> (16 + component * 4)) & 15, "float32") - 8,
            T.cast((packed >> (24 + component * 4)) & 15, "float32") - 8,
        )
        acc = acc + scale * T.call_extern("float32", "dot", left, right)

    @T.macro
    def q6_scalar(acc: T.Ref, x, w, base, tile, lane, group, scale):
        low = byte(w, base + (group // 4 * 2 + group % 2) * 32 + lane)
        high = byte(w, base + 128 + group // 4 * 32 + lane)
        bits = byte(w, base + 192 + group * 2 + lane // 16)
        signed_scale = T.cast(
            T.cast(bits, "int32") - T.if_then_else(T.cast(bits, "int32") >= 128, 256, 0), "float32"
        )
        value = ((low >> (4 * (group % 4 // 2))) & 15) | (((high >> (2 * (group % 4))) & 3) << 4)
        acc = acc + x[tile * 256 + group * 32 + lane] * (
            scale * signed_scale * (T.cast(value, "float32") - 32)
        )

    @T.macro
    def q6_vector(acc: T.Ref, x, w, base, tile, lane, half, scale):
        low = word_value(w, base + half * 64 + (lane // 8 % 2) * 32 + lane % 8 * 4)
        high = word_value(w, base + 128 + half * 32 + lane % 8 * 4)
        j = tile * 256 + half * 128 + lane * 4
        left = T.call_extern("float32x4", "vec4<f32>", x[j], x[j + 1], x[j + 2], x[j + 3])
        right = T.call_extern(
            "float32x4",
            "vec4<f32>",
            T.cast(
                ((low >> (lane // 16 * 4)) & 15) | (((high >> (lane // 8 * 2)) & 3) << 4), "float32"
            )
            - 32,
            T.cast(
                ((low >> (8 + lane // 16 * 4)) & 15) | (((high >> (8 + lane // 8 * 2)) & 3) << 4),
                "float32",
            )
            - 32,
            T.cast(
                ((low >> (16 + lane // 16 * 4)) & 15) | (((high >> (16 + lane // 8 * 2)) & 3) << 4),
                "float32",
            )
            - 32,
            T.cast(
                ((low >> (24 + lane // 16 * 4)) & 15) | (((high >> (24 + lane // 8 * 2)) & 3) << 4),
                "float32",
            )
            - 32,
        )
        bits = byte(w, base + 192 + half * 8 + lane // 4)
        signed_scale = T.cast(
            T.cast(bits, "int32") - T.if_then_else(T.cast(bits, "int32") >= 128, 256, 0), "float32"
        )
        acc = acc + scale * signed_scale * T.call_extern("float32", "dot", left, right)

    @T.macro
    def update_native(acc: T.Ref, x, w, tile, lane, row, bx):
        index = tile * 128 + lane * 4
        left = T.call_extern(
            "float32x4", "vec4<f32>", x[index], x[index + 1], x[index + 2], x[index + 3]
        )
        base = (bx * 4 + row) * k + index
        right = T.call_extern(
            "float32x4",
            "vec4<f32>",
            T.cast(w[base], "float32"),
            T.cast(w[base + 1], "float32"),
            T.cast(w[base + 2], "float32"),
            T.cast(w[base + 3], "float32"),
        )
        acc = acc + T.call_extern("float32", "dot", left, right)

    @T.macro
    def update_scalar(acc: T.Ref, x, w, tile, lane, row, bx):
        acc = acc + x[tile * 32 + lane] * decoder(w, bx * row_count + row, tile * 32 + lane)

    @T.macro
    def compute(acc, x, w, tile, lane, row, bx, slot):
        if fast_q4:
            base = ((bx * row_count + row) * (k // 32) + tile * (lanes // 4) + lane // 4) * 18
            scale = half_value(w, base)
            packed = word_value(w, base + 2 + lane % 4 * 4)
            index = tile * (lanes * 8) + lane // 4 * 32 + lane % 4 * 4
            q4_terms(acc, x, index, scale, packed, slot)
        elif q == 14:
            base = ((bx * row_count + row) * (k // 256) + tile) * 210
            scale = half_value(w, base + 208)
            q6_terms(acc, x, w, base, tile, lane, slot, scale)
        elif q in (0, 1) and k % 128 == 0:
            update_native(acc[slot][0], x, w, tile, lane, row, bx)
        elif q == 2:
            base = ((bx * row_count + row) * (k // 32) + tile) * 18
            scale = half_value(w, base)
            packed = byte(w, base + 2 + lane)
            scalar_term(acc[slot][0], x, tile * 32 + lane, scale, packed, 0, 0)
            scalar_term(acc[slot][0], x, tile * 32 + lane, scale, packed, 4, 16)
        else:
            update_scalar(acc[slot][0], x, w, tile, lane, row, bx)

    @T.macro
    def combine4(out: T.Ref, a, b, c, d):
        out = (a + b) + (c + d)

    @T.macro
    def add(out: T.Ref, value):
        out = out + value

    @T.macro
    def shuffle(acc: T.Ref, stride):
        acc = acc + T.call_extern("float32", "subgroupShuffleXor", acc, T.uint32(stride))

    @T.macro
    def spill(scratch, value, tx):
        scratch[tx] = value

    @T.macro
    def shared_step(scratch, tx, stride):
        scratch[tx] = scratch[tx] + scratch[tx + stride]

    @T.macro
    def reduction_step(scratch, tx, lane, stride):
        if lane < stride:
            steps(scratch, tx, stride)
        T.sync_threads()

    @T.macro
    def gemv(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(o, row_count), threads=threads) as bx:
            tx = T.get_thread_binding()
            row = tx // lanes
            lane = tx % lanes
            accum = registers((parts, chains))
            totals = registers((parts,))
            scratch = shared_buffers((threads,), "float32", parts)
            if bx * row_count + row < o:
                for chunk in T.serial(tiles // unroll):
                    compute_slots(accum, x, (w, w2), chunk, lane, row, bx)
            combine(accum, totals)
            if p.get("sg"):
                if T.call_extern("uint32", "tensor_subgroup_size") >= lanes:
                    shuffles(totals)
                    spills(totals, scratch, tx)
                else:
                    spills(totals, scratch, tx)
                    T.sync_threads()
                    reduction(scratch, tx, lane)
            else:
                spills(totals, scratch, tx)
                T.sync_threads()
                reduction(scratch, tx, lane)
            if (lane == 0) & (bx * row_count + row < o):
                epilogue(
                    out,
                    residual,
                    bx * row_count + row,
                    scratch[0][tx],
                    scratch[1][tx] if parts == 2 else 0,
                )

    return projection_abi(kind, p, gemv)


def quantize_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    r, k = (p.get("r", 1) if kind == "quantize_q16" else 1), p["k"]
    if k % 32:
        raise ValueError("activation blocks require depth divisible by 32")
    blocks, words, components = r * k // 32, r * k // 4, 2 if kind == "quantize_q16" else 1
    maximum, total_reduce, rounded = tree_reduce(32, "max"), tree_reduce(32), round_half()

    def shared_components(packed, scales, sums, quants, values, scratch, factor, block, tx):
        for component in range(components):
            component_shared(
                packed, scales, sums, quants, values, scratch, factor, block, tx, component
            )

    def subgroup_components(packed, scales, sums, value, factor, block, tx):
        for component in range(components):
            component_subgroup(packed, scales, sums, value, factor, block, tx, component)

    @T.macro
    def component_shared(
        packed, scales, sums, quants, values, scratch, factor: T.Ref, block, tx, component
    ):
        scratch[tx] = T.abs(values[tx])
        T.sync_threads()
        maximum(scratch, tx)
        if component == 1 and p.get("fixed_residual"):
            factor = scales[block] / 254
        else:
            factor = T.max(scratch[0] / 127, 1.0e-20)
        quants[component, tx] = T.cast(T.round(values[tx] / factor), "int32")
        T.sync_threads()
        if tx < 8:
            packed[block * 8 + tx + component * words] = (
                (T.reinterpret("uint32", quants[component, tx * 4]) & 255)
                | ((T.reinterpret("uint32", quants[component, tx * 4 + 1]) & 255) << 8)
                | ((T.reinterpret("uint32", quants[component, tx * 4 + 2]) & 255) << 16)
                | ((T.reinterpret("uint32", quants[component, tx * 4 + 3]) & 255) << 24)
            )
        if components == 2:
            values[tx] = values[tx] - T.cast(quants[component, tx], "float32") * factor
        scratch[tx] = T.cast(quants[component, tx], "float32")
        T.sync_threads()
        total_reduce(scratch, tx)
        if tx == 0:
            scales[block + component * blocks] = factor
            sums[block + component * blocks] = T.cast(scratch[0], "int32")
        if components == 2:
            T.sync_threads()

    @T.macro
    def component_subgroup(packed, scales, sums, value: T.Ref, factor: T.Ref, block, tx, component):
        if component == 1 and p.get("fixed_residual"):
            factor = factor / 254
        else:
            factor = T.max(T.call_extern("float32", "subgroupMax", T.abs(value)) / 127, 1.0e-20)
        quant = T.cast(T.round(value / factor), "int32")
        bits = (
            (
                T.reinterpret(
                    "uint32",
                    T.call_extern("int32", "subgroupShuffle", quant, T.uint32((tx // 4) * 4)),
                )
                & 255
            )
            | (
                (
                    T.reinterpret(
                        "uint32",
                        T.call_extern(
                            "int32", "subgroupShuffle", quant, T.uint32((tx // 4) * 4 + 1)
                        ),
                    )
                    & 255
                )
                << 8
            )
            | (
                (
                    T.reinterpret(
                        "uint32",
                        T.call_extern(
                            "int32", "subgroupShuffle", quant, T.uint32((tx // 4) * 4 + 2)
                        ),
                    )
                    & 255
                )
                << 16
            )
            | (
                (
                    T.reinterpret(
                        "uint32",
                        T.call_extern(
                            "int32", "subgroupShuffle", quant, T.uint32((tx // 4) * 4 + 3)
                        ),
                    )
                    & 255
                )
                << 24
            )
        )
        total = T.call_extern("int32", "subgroupAdd", quant)
        if tx % 4 == 0:
            packed[block * 8 + tx // 4 + component * words] = bits
        if tx == 0:
            scales[block + component * blocks] = factor
            sums[block + component * blocks] = total
        value = value - T.cast(quant, "float32") * factor

    @T.macro
    def quantize(x, packed, scales, sums):
        with T.Kernel(blocks, threads=32) as block:
            tx = T.get_thread_binding()
            scratch = T.alloc_shared((32,), "float32")
            quants = T.alloc_shared((components, 32), "int32")
            values = T.alloc_shared((32,), "float32")
            factor = T.alloc_var("float32")
            if components == 2 and p.get("sg"):
                value = T.alloc_var("float32")
                value = rounded(x[block * 32 + tx])
                if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
                    subgroup_components(packed, scales, sums, value, factor, block, tx)
                else:
                    values[tx] = rounded(x[block * 32 + tx])
                    shared_components(
                        packed, scales, sums, quants, values, scratch, factor, block, tx
                    )
            else:
                values[tx] = rounded(x[block * 32 + tx]) if components == 2 else x[block * 32 + tx]
                shared_components(packed, scales, sums, quants, values, scratch, factor, block, tx)

    return primitive(
        [
            ("x", r * k, "float32"),
            ("packed" if components == 2 else "out", components * words, "uint32"),
            ("scales", components * blocks, "float32"),
            ("sums", components * blocks, "int32"),
        ],
        quantize,
    )


def integer_prefill_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    from tensor.compiler.webgpu_templates import packed_integer_matmul

    r, k, o = p.get("r", 1), p["k"], p["o"]
    parts = 2 if kind == "ffn_q16" else 1
    byte, half_value, word_value = packed_loads()
    prepacked = p.get("q4_prepacked", False)
    count = k * o // 32 * (9 if prepacked else 18) // (1 if prepacked else 4)

    @T.macro
    def read_word(w, col, block, word):
        if prepacked:
            return w[(col * (k // 32) + block) * 9 + word]
        else:
            return word_value(w, (col * (k // 32) + block) * 18 + 2 + word * 4)

    @T.macro
    def read_scale(w, col, block):
        if prepacked:
            return T.reinterpret("float32", w[(col * (k // 32) + block) * 9 + 8])
        else:
            return half_value(w, (col * (k // 32) + block) * 18)

    config = dict(p.get("integer", {}))
    if prepacked:
        config["signed_rhs"] = True
    algorithm = packed_integer_matmul(
        r, k, o, read_word, read_scale, projection_epilogue(kind), parts=parts, **config
    )
    args = [
        ("packed", r * k // 2, "uint32"),
        ("scales", r * k // 16, "float32"),
        ("sums", r * k // 16, "int32"),
        ("w", count, "uint32"),
    ]
    if parts == 2:

        @T.macro
        def call(packed, scales, sums, w, w2, out):
            algorithm(packed, scales, sums, w, out, w2)

        return primitive([*args, ("w2", count, "uint32"), ("out", r * o, "float32")], call)
    return primitive([*args, ("out", r * o, "float32")], algorithm)


def q8_kernel(p):
    import tilelang.language as T

    k, o, lanes = p["k"], p["o"], 16
    segmented = segmented_reduce(lanes)
    byte, half_value, word_value = packed_loads()

    @T.prim_func
    def kernel(
        x: T.Tensor((k // 4,), "uint32"),
        scales: T.Tensor((k // 32,), "float32"),
        sums: T.Tensor((k // 32,), "int32"),
        w: T.Tensor((k * o // 32 * 18 // 4,), "uint32"),
        out: T.Tensor((o,), "float32"),
    ):
        with T.Kernel(T.ceildiv(o, 8), threads=128) as bx:
            tx = T.get_thread_binding()
            row = tx // lanes
            lane = tx % lanes
            accum = T.alloc_var("float32")
            scratch = T.alloc_shared((128,), "float32")
            accum = 0
            if bx * 8 + row < o:
                for tile in T.serial(k // 128):
                    block = tile * 4 + lane // 4
                    base = ((bx * 8 + row) * (k // 32) + block) * 18
                    packed = word_value(w, base + 2 + lane % 4 * 4)
                    low = T.call_extern(
                        "int32",
                        "dot4I8Packed",
                        packed & T.uint32(252645135),
                        x[block * 8 + lane % 4],
                    )
                    high = T.call_extern(
                        "int32",
                        "dot4I8Packed",
                        (packed >> 4) & T.uint32(252645135),
                        x[block * 8 + lane % 4 + 4],
                    )
                    accum = accum + T.cast(low + high - 2 * sums[block], "float32") * scales[
                        block
                    ] * half_value(w, base)
            segmented(accum, scratch, tx, lane)
            if (lane == 0) & (bx * 8 + row < o):
                out[bx * 8 + row] = scratch[tx]

    return kernel


def segmented_reduce(lanes):
    import tilelang.language as T

    def shuffles(acc):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            shuffle(acc, stride)

    def steps(scratch, tx, lane):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            step(scratch, tx, lane, stride)

    @T.macro
    def shuffle(acc: T.Ref, stride):
        acc = acc + T.call_extern("float32", "subgroupShuffleXor", acc, T.uint32(stride))

    @T.macro
    def step(scratch, tx, lane, stride):
        if lane < stride:
            scratch[tx] = scratch[tx] + scratch[tx + stride]
        T.sync_threads()

    @T.macro
    def segmented(acc: T.Ref, scratch, tx, lane):
        if T.call_extern("uint32", "tensor_subgroup_size") >= lanes:
            shuffles(acc)
            scratch[tx] = acc
        else:
            scratch[tx] = acc
            T.sync_threads()
            steps(scratch, tx, lane)

    return segmented


def rms_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    r, c, eps = p.get("r", 1), p["c"], p["eps"]
    threads, fused = rms_width(c), kind == "add_rms"
    reduce = subgroup_reduce(threads) if p.get("sg") else tree_reduce(threads)

    @T.macro
    def value(x, mixed, index):
        if fused:
            return x[index] + mixed[index]
        else:
            return x[index]

    @T.macro
    def normalize(x, w, out, mixed=None, residual=None):
        with T.Kernel(r, threads=threads) as row:
            tx = T.get_thread_binding()
            total = T.alloc_var("float32")
            scratch = T.alloc_shared((threads,), "float32")
            total = 0
            for i in T.serial(c // threads):
                total = total + value(x, mixed, row * c + i * threads + tx) * value(
                    x, mixed, row * c + i * threads + tx
                )
            if p.get("sg"):
                reduce(scratch, tx, total)
            else:
                scratch[tx] = total
                T.sync_threads()
                reduce(scratch, tx)
            for i in T.serial(c // threads):
                if fused:
                    residual[row * c + i * threads + tx] = (
                        x[row * c + i * threads + tx] + mixed[row * c + i * threads + tx]
                    )
                out[row * c + i * threads + tx] = (
                    value(x, mixed, row * c + i * threads + tx)
                    * T.rsqrt(scratch[0] / c + eps)
                    * w[i * threads + tx]
                )

    if fused:

        @T.macro
        def call(x, mixed, residual, w, out):
            normalize(x, w, out, mixed, residual)

        return primitive(
            [
                ("x", r * c, "float32"),
                ("mixed", r * c, "float32"),
                ("residual", r * c, "float32"),
                ("w", c, "float32"),
                ("out", r * c, "float32"),
            ],
            call,
        )
    return primitive(
        [("x", r * c, "float32"), ("w", c, "float32"), ("out", r * c, "float32")], normalize
    )


def normalization_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    r, h, kh, d, cap = p.get("r", 1), p["h"], p["kh"], p["d"], p["cap"]
    query = kind == "qnorm"
    heads, eps, angle_scale = h if query else kh, p["eps"], math.log(p["theta"]) * 2 / d
    rounded = round_half()

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
            if query:
                for i in T.Parallel(d // 2):
                    angle = T.cast(pos[0] + row, "float32") * T.exp(-angle_scale * i)
                    out[(row * heads + head) * d + i] = norm[i] * T.cos(angle) - norm[
                        i + d // 2
                    ] * T.sin(angle)
                    out[(row * heads + head) * d + i + d // 2] = norm[i] * T.sin(angle) + norm[
                        i + d // 2
                    ] * T.cos(angle)
            else:
                if row < pos[1]:
                    for i in T.Parallel(d // 2):
                        angle = T.cast(pos[0] + row, "float32") * T.exp(-angle_scale * i)
                        out[((pos[0] + row) * heads + head) * d + i] = rounded(
                            norm[i] * T.cos(angle) - norm[i + d // 2] * T.sin(angle)
                        )
                        out[((pos[0] + row) * heads + head) * d + i + d // 2] = rounded(
                            norm[i] * T.sin(angle) + norm[i + d // 2] * T.cos(angle)
                        )
                    for i in T.Parallel(d):
                        vc[((pos[0] + row) * heads + head) * d + i] = rounded(
                            v[(row * heads + head) * d + i]
                        )

    @T.macro
    def call(k, v, kw, kc, vc, pos):
        normalize(k, kw, kc, pos, v, vc)

    if query:
        return primitive(
            [
                ("q", r * h * d, "float32"),
                ("qw", d, "float32"),
                ("qo", r * h * d, "float32"),
                ("pos", 2, "int32"),
            ],
            normalize,
        )
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


def score_shuffle():
    import tilelang.language as T

    def reduce(dot):
        for stride in (16, 8, 4, 2, 1):
            step(dot, stride)

    @T.macro
    def step(dot: T.Ref, stride):
        dot = dot + T.call_extern("float32", "subgroupShuffleXor", dot, T.uint32(stride))

    return reduce


def attention_scores_kernel(p):
    import tilelang.language as T

    h, kh, d, cap = (p[n] for n in ("h", "kh", "d", "cap"))
    shuffle = score_shuffle()

    @T.prim_func
    def kernel(
        q: T.Tensor((h * d,), "float32"),
        kc: T.Tensor((cap * kh * d,), "float16"),
        out: T.Tensor((h * cap,), "float32"),
        pos: T.Tensor((2,), "int32"),
    ):
        with T.Kernel(h, T.ceildiv(cap, 32), threads=128) as (head, block):
            tx = T.get_thread_binding()
            dot = T.alloc_var("float32")
            if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
                lane = tx % 32
                for tile in T.serial(8):
                    token = block * 32 + tile * 4 + tx // 32
                    dot = 0
                    if token <= pos[0]:
                        for j in T.serial(d // 32):
                            channel = j * 32 + lane
                            dot = dot + q[head * d + channel] * T.cast(
                                kc[(token * kh + head // (h // kh)) * d + channel], "float32"
                            )
                    shuffle(dot)
                    if (lane == 0) & (token < cap):
                        out[head * cap + token] = T.if_then_else(
                            token <= pos[0], dot * d**-0.5, -T.infinity("float32")
                        )
            else:
                if tx < 32:
                    token = block * 32 + tx
                    if token < cap:
                        dot = -T.infinity("float32")
                        if token <= pos[0]:
                            dot = 0
                            for j in T.serial(d):
                                dot = dot + q[head * d + j] * T.cast(
                                    kc[(token * kh + head // (h // kh)) * d + j], "float32"
                                )
                            dot = dot * d**-0.5
                        out[head * cap + token] = dot

    return kernel


def attention_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    r, h, kh, d, cap = p.get("r", 1), p["h"], p["kh"], p["d"], p["cap"]
    partitioned = r == 1 and p.get("attention_schedule") == "partitioned_values"
    channels, parts = (p["channels"], p["value_parts"]) if partitioned else (d, 1)
    threads = channels * parts if partitioned else 128
    if partitioned and (
        channels not in (16, 32, 64)
        or d % channels
        or parts not in (1, 2, 4, 8, 16)
        or threads > 1024
    ):
        raise ValueError("unsupported decode attention distribution")
    fused_scores = partitioned and p.get("fused_scores", False)
    if fused_scores and (threads % 32 or d % 32):
        raise ValueError("fused attention requires complete 32-lane reductions")
    split = partitioned and not fused_scores or not partitioned and p.get("sg") and r == 1
    sg = partitioned or p.get("sg", False)
    max_reduce = subgroup_reduce(threads, "max") if sg else tree_reduce(threads, "max")
    sum_reduce = subgroup_reduce(threads) if sg else tree_reduce(threads)
    shuffle = score_shuffle()
    score_loop = T.unroll if partitioned else T.serial

    @T.macro
    def scores_serial(q, kc, scores, maximum: T.Ref, dot: T.Ref, head, row, pos, tx):
        for tile in T.serial(T.ceildiv(cap, threads)):
            token = tile * threads + tx
            if token < cap:
                dot = -T.infinity("float32")
                if token <= pos[0] + row:
                    dot = 0
                    for j in T.serial(d):
                        dot = dot + q[(row * h + head) * d + j] * T.cast(
                            kc[(token * kh + head // (h // kh)) * d + j], "float32"
                        )
                    dot = dot * d**-0.5
                scores[token] = dot
                maximum = T.max(maximum, dot)

    @T.macro
    def scores_subgroup(q, kc, scores, maximum: T.Ref, dot: T.Ref, head, row, pos, tx):
        if not partitioned:
            for token in T.Parallel(cap):
                scores[token] = -T.infinity("float32")
            T.sync_threads()
        if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
            lane = tx % 32
            for tile in T.serial(T.ceildiv(pos[0] + row + 1, threads // 32)):
                token = tile * (threads // 32) + tx // 32
                dot = 0
                if token <= pos[0] + row:
                    for j in score_loop(d // 32):
                        channel = j * 32 + lane
                        dot = dot + q[(row * h + head) * d + channel] * T.cast(
                            kc[(token * kh + head // (h // kh)) * d + channel], "float32"
                        )
                shuffle(dot)
                if (lane == 0) & (token <= pos[0] + row):
                    scores[token] = dot * d**-0.5
        else:
            if partitioned:
                for tile in T.serial(T.ceildiv(pos[0] + 1, threads)):
                    token = tile * threads + tx
                    if token <= pos[0]:
                        dot = 0
                        for j in T.serial(d):
                            dot = dot + q[head * d + j] * T.cast(
                                kc[(token * kh + head // (h // kh)) * d + j], "float32"
                            )
                        scores[token] = dot * d**-0.5
            else:
                scores_serial(q, kc, scores, maximum, dot, head, row, pos, tx)
        T.sync_threads()
        for tile in T.serial(T.ceildiv(pos[0] + row + 1, threads)):
            token = tile * threads + tx
            if token < cap:
                if not partitioned or token <= pos[0]:
                    maximum = T.max(maximum, scores[token])

    @T.macro
    def attention(q, vc, out, pos, kc=None):
        with T.Kernel(h, d // channels if partitioned else r, threads=threads) as (head, block):
            row = 0 if partitioned else block
            tx = T.get_thread_binding()
            dot = T.alloc_var("float32")
            maximum = T.alloc_var("float32")
            total = T.alloc_var("float32")
            result = T.alloc_var("float32")
            scores = T.alloc_shared((cap,), "float32")
            scratch = T.alloc_shared((threads,), "float32")
            if partitioned:
                partials = T.alloc_shared((threads,), "float32")
                channel = tx % channels
                part = tx // channels
            maximum = -T.infinity("float32")
            if split:
                for tile in T.serial(T.ceildiv(pos[0] + row + 1 if sg else cap, threads)):
                    token = tile * threads + tx
                    if token < cap:
                        if not partitioned or token <= pos[0]:
                            scores[token] = q[head * cap + token]
                            maximum = T.max(maximum, scores[token])
            elif sg:
                scores_subgroup(q, kc, scores, maximum, dot, head, row, pos, tx)
            else:
                scores_serial(q, kc, scores, maximum, dot, head, row, pos, tx)
            if sg:
                max_reduce(scratch, tx, maximum)
            else:
                scratch[tx] = maximum
                T.sync_threads()
                max_reduce(scratch, tx)
            total = 0
            for tile in T.serial(T.ceildiv(pos[0] + row + 1 if sg else cap, threads)):
                token = tile * threads + tx
                if token < cap:
                    if not partitioned or token <= pos[0]:
                        scores[token] = T.exp(scores[token] - scratch[0])
                        total = total + scores[token]
            T.sync_threads()
            if sg:
                sum_reduce(scratch, tx, total)
            else:
                scratch[tx] = total
                T.sync_threads()
                sum_reduce(scratch, tx)
            result = 0
            if partitioned:
                for tile in T.serial(T.ceildiv(pos[0] + 1, parts)):
                    token = tile * parts + part
                    if token <= pos[0]:
                        result = result + scores[token] * T.cast(
                            vc[(token * kh + head // (h // kh)) * d + block * channels + channel],
                            "float32",
                        )
                partials[tx] = result
                T.sync_threads()
                if part == 0:
                    result = 0
                    for other in T.unroll(parts):
                        result = result + partials[other * channels + channel]
                    out[head * d + block * channels + channel] = result / scratch[0]
            else:
                if tx < d:
                    if sg:
                        for token in T.serial(pos[0] + row + 1):
                            result = result + scores[token] * T.cast(
                                vc[(token * kh + head // (h // kh)) * d + tx], "float32"
                            )
                    else:
                        for token in T.serial(cap):
                            if token <= pos[0] + row:
                                result = result + scores[token] * T.cast(
                                    vc[(token * kh + head // (h // kh)) * d + tx], "float32"
                                )
                    out[(row * h + head) * d + tx] = result / scratch[0]

    @T.macro
    def call(q, kc, vc, out, pos):
        attention(q, vc, out, pos, kc)

    args = [("q", h * cap if split else r * h * d, "float32")]
    if not split:
        args += [("kc", cap * kh * d, "float16")]
    args += [("vc", cap * kh * d, "float16"), ("out", r * h * d, "float32"), ("pos", 2, "int32")]
    if split:
        return primitive(args, attention)
    return primitive(args, call)


def make_kernel(kind, p):
    import tilelang.language as T

    r = p.get("r", 1)
    if kind in ("linear", "ffn") and p.get("q16"):
        return integer_prefill_kernel(kind + "_q16", p)
    if kind in ("linear_q16", "ffn_q16"):
        return integer_prefill_kernel(kind, p)
    if kind in ("linear", "ffn", "linear_add"):
        if kind == "linear_add" and r != 1:
            raise ValueError("residual projection fusion requires decode rows")
        return decode_kernel(kind, p) if r == 1 else prefill_kernel(kind, p)
    if kind in ("quantize_q8", "quantize_q16"):
        return quantize_kernel(kind, p)
    if kind == "linear_q8":
        return q8_kernel(p)
    if kind == "rms" and (r == 1 or p.get("parallel_rows")) or kind == "add_rms":
        return rms_kernel(kind, p)
    if kind in ("qnorm", "kvnorm"):
        return normalization_kernel(kind, p)
    if kind == "attention_scores":
        return attention_scores_kernel(p)
    if kind == "attention":
        return attention_kernel(p)
    if kind == "embedding":
        c, q, v = p["c"], p["type"], p["v"]
        count, dtype = weight_storage(q, v * c, packed_words=True)
        decoder = weight_decoder(q, c, packed_words=True)

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
                        out[row * c + col] = decoder(w, tokens[row], col)

        return kernel
    if kind == "argmax":
        # This shared-memory reduction has identical CUDA and WebGPU semantics.
        from .baseline import argmax_kernel

        return argmax_kernel(p, explicit_unroll=True)
    if kind == "prefill_tail":
        c, t = p["c"], p["t"]

        @T.prim_func
        def kernel(
            x: T.Tensor((r * c,), "float32"),
            out: T.Tensor((t * c,), "float32"),
            control: T.Tensor((2,), "int32"),
            tail_control: T.Tensor((2,), "int32"),
        ):
            with T.Kernel(T.ceildiv(t * c, 256), threads=256) as block:
                tx = T.get_thread_binding()
                index = block * 256 + tx
                start = T.max(control[1] - t, 0)
                count = T.min(control[1], t)
                if index < t * c:
                    out[index] = T.if_then_else(index // c < count, x[start * c + index], 0)
                if index == 0:
                    tail_control[0] = control[0] + start
                    tail_control[1] = count

        return kernel
    if kind == "qkv":
        raise ValueError("WebGPU uses separate query/cache normalization kernels")
    return baseline_kernel(kind, p)
