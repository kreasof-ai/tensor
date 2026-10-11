"""Independent source controls for generic WebGPU compiler tests."""

def emit(args, body):
    declarations = [f'{name}: T.Tensor(({n},), "{dtype}")' for name, n, dtype in args]
    return (
        "import tilelang.language as T\n\n@T.prim_func\ndef kernel("
        + ", ".join(declarations)
        + "):\n"
        + "\n".join(("    " + line for line in body.splitlines()))
        + '\n\ndef tensor_export():\n    return {"kernel": kernel}\n'
    )

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
