"""Source-text helpers for historical benchmark experiments only.

Production packages define kernels in TileLang DSL. These helpers retain the
independent text controls used by FP16 and packed-load research experiments.
"""

from tensor_llm.common.gguf import TYPES


def emit(args, body):
    declarations = [f'{name}: T.Tensor(({n},), "{dtype}")' for name, n, dtype in args]
    return (
        "import tilelang.language as T\n\n@T.prim_func\ndef kernel("
        + ", ".join(declarations)
        + "):\n"
        + "\n".join(("    " + line for line in body.splitlines()))
        + '\n\ndef tensor_export():\n    return {"kernel": kernel}\n'
    )


def weight(kind, row, col, k, buffer="w", *, packed_words=False):
    """Expression reading one exact GGML value; packed buffers are unsigned bytes."""
    if kind in (0, 1):
        return f'T.cast({buffer}[({row}) * {k} + ({col})], "float32")'
    _, block, size = TYPES[kind]
    base = f"((({row}) * {k} + ({col})) // {block} * {size})"
    j = f"(({col}) % {block})"
    u = lambda offset: f'T.cast({buffer}[{base} + ({offset})], "uint32")'
    half = lambda offset: (
        f"""T.cast(T.reinterpret("float16", T.cast({u(offset)} | ({u(str(offset) + " + 1")} << 8), "uint16")), "float32")"""
    )
    if packed_words:
        u = lambda offset: (
            f"(({buffer}[({base} + ({offset})) // 4] >> (({base} + ({offset})) % 4 * 8)) & T.uint32(255))"
        )

        def half(offset):
            bits = f"({u(offset)} | ({u(str(offset) + ' + 1')} << 8))"
            return f'T.call_extern("float32", "tensor_unpack_f16", {bits})'

    signed = lambda expr: (
        f'T.cast(T.reinterpret("int8", T.cast({expr}, "uint8")), "float32")'
        if not packed_words
        else f'T.cast(T.cast({expr}, "int32") - T.if_then_else({expr} >= 128, 256, 0), "float32")'
    )
    if kind == 2:
        return f"""{half(0)} * (T.cast(({u("2 + " + j + " % 16")} >> (4 * ({j} // 16))) & 15, "float32") - 8)"""
    if kind == 8:
        return f"{half(0)} * {signed(u('2 + ' + j))}"
    if kind == 12:
        group = f"({j} // 32)"
        scale = f"T.if_then_else({group} < 4, {u('4 + ' + group)} & 63, ({u('8 + ' + group)} & 15) | (({u(group)} >> 6) << 4))"
        minimum = f"T.if_then_else({group} < 4, {u('8 + ' + group)} & 63, ({u('8 + ' + group)} >> 4) | (({u('4 + ' + group)} >> 6) << 4))"
        q = f"(({u('16 + ' + j + ' // 64 * 32 + ' + j + ' % 32')} >> (4 * ({group} % 2))) & 15)"
        return f'{half(0)} * T.cast({scale}, "float32") * T.cast({q}, "float32") - {half(2)} * T.cast({minimum}, "float32")'
    if kind == 14:
        group = f"({j} % 128 // 32)"
        low = u(j + " // 128 * 64 + " + group + " % 2 * 32 + " + j + " % 32")
        high = u("128 + " + j + " // 128 * 32 + " + j + " % 32")
        scale = u("192 + " + j + " // 16")
        q = f"((({low} >> (4 * ({group} // 2))) & 15) | ((({high} >> (2 * {group})) & 3) << 4))"
        return f'{half(208)} * {signed(scale)} * (T.cast({q}, "float32") - 32)'
    raise ValueError("unsupported GGML encoding")


def half_bits(bits):
    return f'T.call_extern("float32", "tensor_unpack_f16", T.cast({bits}, "uint32"))'


def round_half(value):
    """Round FP32 to an exactly representable FP16 value before native casting.

    Native f16 conversion need not match NumPy's ties-to-even rounding. Integer
    rounding also handles FP16 subnormals, retaining the stated prefill contract.

    The exponent range tests compare a signed cast. Lowering the equivalent
    unsigned compare emits `exponent / 113u < 1u`, and an unsigned divide costs
    far more than the branch it replaces. Every staged prefill operand passes
    through this routine, so those two divisions dominated the staging loop.
    """
    bits = f'T.reinterpret("uint32", {value})'
    exponent = f"(({bits} >> 23) & 255)"
    signed = f'T.cast({exponent}, "int32")'
    shift = f"T.min(T.max(126 - {signed}, 1), 24)"
    mantissa = f"(({bits} & 8388607) | 8388608)"
    rounded = f"(({mantissa} + (T.uint32(1) << ({shift} - 1)) - 1 + (({mantissa} >> {shift}) & 1)) >> {shift})"
    quantum = f'T.reinterpret("float32", T.uint32({103 << 23}))'
    small = f'T.if_then_else({signed} < 102, 0.0, T.cast({rounded}, "float32") * {quantum}) * T.if_then_else(({bits} >> 31) != 0, -1.0, 1.0)'
    normal = (
        f'T.reinterpret("float32", ({bits} + 4095 + (({bits} >> 13) & 1)) & T.uint32(4294959104))'
    )
    return f"T.if_then_else({signed} < 113, {small}, {normal})"


def reference_attention_source(kind, p):
    r = p.get("r", 1)
    a = lambda name, count, dtype="float32": (name, count, dtype)
    if kind == "attention":
        h, kh, d, cap = (p["h"], p["kh"], p["d"], p["cap"])
        args = [
            a("q", r * h * d),
            a("kc", cap * kh * d, "float16"),
            a("vc", cap * kh * d, "float16"),
            a("out", r * h * d),
            a("pos", 2, "int32"),
        ]
        if r == 1:
            return emit(
                args,
                f'with T.Kernel({h}, threads=128) as head:\n    dots = T.alloc_fragment((64, {d}), "float32")\n    scores = T.alloc_fragment((64,), "float32")\n    products = T.alloc_fragment((64, {d}), "float32")\n    partial = T.alloc_fragment(({d},), "float32")\n    result = T.alloc_fragment(({d},), "float32")\n    maximum = T.alloc_fragment((1,), "float32")\n    previous = T.alloc_fragment((1,), "float32")\n    normalizer = T.alloc_fragment((1,), "float32")\n    total = T.alloc_fragment((1,), "float32")\n    T.fill(maximum, -T.infinity("float32"))\n    T.clear(result)\n    T.clear(normalizer)\n    for tile in T.serial(T.ceildiv(pos[0] + 1, 64)):\n        for i, j in T.Parallel(64, {d}):\n            dots[i, j] = q[head * {d} + j] * T.cast(kc[((tile * 64 + i) * {kh} + head // {h // kh}) * {d} + j], "float32")\n        T.reduce_sum(dots, scores, dim=1)\n        T.copy(maximum, previous)\n        for i in T.Parallel(64):\n            scores[i] = T.if_then_else(tile * 64 + i <= pos[0], scores[i] * {d ** (-0.5)}, -T.infinity("float32"))\n        T.reduce_max(scores, maximum, dim=0, clear=False)\n        for i in T.Parallel(64):\n            scores[i] = T.exp(scores[i] - maximum[0])\n        T.reduce_sum(scores, total, dim=0)\n        normalizer[0] = normalizer[0] * T.exp(previous[0] - maximum[0]) + total[0]\n        for i, j in T.Parallel(64, {d}):\n            products[i, j] = scores[i] * T.cast(vc[((tile * 64 + i) * {kh} + head // {h // kh}) * {d} + j], "float32")\n        T.reduce_sum(products, partial, dim=0)\n        for j in T.Parallel({d}):\n            result[j] = result[j] * T.exp(previous[0] - maximum[0]) + partial[j]\n    for j in T.Parallel({d}):\n        out[head * {d} + j] = result[j] / normalizer[0]',
            )
        return emit(
            args,
            f'with T.Kernel(T.ceildiv({r}, 32), {h}, threads=128) as (bx, head):\n    query = T.alloc_shared((32, {d}), "float16")\n    key = T.alloc_shared((64, {d}), "float16")\n    value = T.alloc_shared((64, {d}), "float16")\n    prob = T.alloc_shared((32, 64), "float16")\n    scores = T.alloc_fragment((32, 64), "float32")\n    result = T.alloc_fragment((32, {d}), "float32")\n    maximum = T.alloc_fragment((32,), "float32")\n    previous = T.alloc_fragment((32,), "float32")\n    factor = T.alloc_fragment((32,), "float32")\n    normalizer = T.alloc_fragment((32,), "float32")\n    total = T.alloc_fragment((32,), "float32")\n    for i, j in T.Parallel(32, {d}):\n        query[i, j] = T.if_then_else(bx * 32 + i < {r}, q[((bx * 32 + i) * {h} + head) * {d} + j], 0)\n    T.clear(result)\n    T.clear(normalizer)\n    T.fill(maximum, -T.infinity("float32"))\n    for tile in T.serial(T.ceildiv(pos[0] + T.min({r}, (bx + 1) * 32), 64)):\n        for i, j in T.Parallel(64, {d}):\n            key[i, j] = kc[((tile * 64 + i) * {kh} + head // {h // kh}) * {d} + j]\n            value[i, j] = vc[((tile * 64 + i) * {kh} + head // {h // kh}) * {d} + j]\n        T.gemm(query, key, scores, transpose_B=True, clear_accum=True)\n        T.copy(maximum, previous)\n        for i, j in T.Parallel(32, 64):\n            scores[i, j] = T.if_then_else(tile * 64 + j <= pos[0] + bx * 32 + i, scores[i, j] * {d ** (-0.5)}, -T.infinity("float32"))\n        T.reduce_max(scores, maximum, dim=1, clear=False)\n        for i in T.Parallel(32):\n            factor[i] = T.exp(previous[i] - maximum[i])\n        for i, j in T.Parallel(32, 64):\n            scores[i, j] = T.exp(scores[i, j] - maximum[i])\n        T.reduce_sum(scores, total, dim=1)\n        for i in T.Parallel(32):\n            normalizer[i] = normalizer[i] * factor[i] + total[i]\n        for i, j in T.Parallel(32, {d}):\n            result[i, j] *= factor[i]\n        T.copy(scores, prob)\n        T.gemm(prob, value, result)\n    for i, j in T.Parallel(32, {d}):\n        if bx * 32 + i < {r}:\n            out[((bx * 32 + i) * {h} + head) * {d} + j] = T.if_then_else(bx * 32 + i < pos[1], result[i, j] / normalizer[i], 0)',
        )
    raise ValueError(f"unknown LFM2 kernel {kind}")
