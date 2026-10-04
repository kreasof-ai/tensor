"""TileLang DSL for FX pointwise regions and GEMM epilogues.

The graph descriptor contains operations and operands, not Python expressions.
Python specializes that descriptor into frontend arithmetic during IR building.
"""

import math
import operator


def make_kernel(specification):
    import tilelang.language as T
    from tensor.compiler.entry import primitive

    dtype, shape = specification["dtype"], specification["shape"]
    input_shapes, gemm = specification["inputs"], specification.get("gemm")
    binary = {
        "add": operator.add,
        "sub": operator.sub,
        "mul": operator.mul,
        "div": operator.truediv,
    }

    def pointwise(operands, coordinates, index=None, accum=None, row=None, col=None):
        expressions = {}
        for i, (buffer, buffer_shape) in enumerate(zip(operands, input_shapes)):
            if gemm and i in (gemm["a"], gemm["b"]):
                continue
            if index is not None:
                indices = (
                    coordinates
                    if tuple(buffer_shape) == tuple(shape)
                    else (index % shape[-1],)
                    if math.prod(buffer_shape) > 1
                    else tuple(0 for _ in buffer_shape)
                )
            else:
                indices = (col,) if len(buffer_shape) == 1 else (row, col)
            expressions["arg" + str(i)] = buffer[indices]
        if gemm:
            value = accum
            if "bias" in gemm:
                value = value + expressions["arg" + str(gemm["bias"])]
            expressions[gemm["name"]] = T.cast(value, "float16")
        for operation in specification["operations"]:
            args = [
                expressions[arg["node"]] if "node" in arg else arg["scalar"]
                for arg in operation["args"]
            ]
            kind, alpha = operation["kind"], operation["alpha"]
            if kind in binary:
                rhs = args[1] * alpha if alpha != 1 else args[1]
                value = binary[kind](args[0], rhs)
            elif kind == "neg":
                value = -args[0]
            elif kind == "relu":
                value = T.if_then_else(T.isnan(args[0]), args[0], T.max(args[0], 0))
            elif kind == "sigmoid":
                value = 1 / (1 + T.exp(-T.cast(args[0], "float32")))
            elif kind == "tanh":
                value = T.tanh(T.cast(args[0], "float32"))
            else:
                raise ValueError("unsupported epilogue")
            expressions[operation["name"]] = T.cast(value, dtype)
        return expressions[specification["root"]]

    def coordinates(index):
        return tuple((index // math.prod(shape[i + 1 :])) % d for i, d in enumerate(shape))

    size = math.prod(shape)

    @T.macro
    def elementwise(operands, out):
        with T.Kernel(T.ceildiv(size, 256), threads=256) as block:
            for lane in T.Parallel(256):
                index = block * 256 + lane
                if index < size:
                    out[coordinates(index)] = pointwise(operands, coordinates(index), index=index)

    if gemm:
        m, k, n, transpose = gemm["m"], gemm["k"], gemm["n"], gemm["transpose"]
        rhs_shape = (64, 32) if transpose else (32, 64)

        @T.macro
        def matmul(operands, out):
            with T.Kernel(T.ceildiv(n, 64), T.ceildiv(m, 32), threads=128) as (bx, by):
                lhs = T.alloc_shared((32, 32), "float16")
                rhs = T.alloc_shared(rhs_shape, "float16")
                accum = T.alloc_fragment((32, 64), "float32")
                T.clear(accum)
                for tile in T.Pipelined(T.ceildiv(k, 32), num_stages=3):
                    T.copy(operands[gemm["a"]][by * 32, tile * 32], lhs)
                    if transpose:
                        T.copy(operands[gemm["b"]][bx * 64, tile * 32], rhs)
                    else:
                        T.copy(operands[gemm["b"]][tile * 32, bx * 64], rhs)
                    T.gemm(lhs, rhs, accum, transpose_B=transpose)
                for row, col in T.Parallel(32, 64):
                    if (by * 32 + row < m) & (bx * 64 + col < n):
                        out[by * 32 + row, bx * 64 + col] = pointwise(
                            operands,
                            (),
                            accum=accum[row, col],
                            row=by * 32 + row,
                            col=bx * 64 + col,
                        )

    def algorithm(*buffers):
        if gemm:
            matmul(buffers[:-1], buffers[-1])
        else:
            elementwise(buffers[:-1], buffers[-1])

    return primitive(
        [
            *[(f"arg{i}", tuple(s), dtype) for i, s in enumerate(input_shapes)],
            ("out", tuple(shape), dtype),
        ],
        algorithm,
    )
