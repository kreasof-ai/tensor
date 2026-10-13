"""Packed integer implementation for WebGPU kernels."""

from itertools import product

from tensor.compiler.webgpu.schedules.registers import registers


def packed_integer_matmul_schedule(
    rows,
    depth,
    columns,
    rhs_words,
    rhs_scales,
    *,
    tile_m=16,
    tile_n=32,
    micro_m=2,
    micro_n=2,
    threads=128,
    owner_axis="column",
    group_order="row",
    tile_k=32,
    fixed_residual=False,
    signed_rhs=False,
):
    """Two-component signed-byte activations, unsigned nibble RHS, F32 sums.

    Producers provide packed RHS word/scale expressions with {column}, {block}
    and {word} placeholders. One or two RHS matrices select linear/SwiGLU output.
    Input planes use packed[2*M*K/4], scales/sums[2*M*K/32]. No GPU imports.
    """
    dims = (rows, depth, columns, tile_m, tile_n, micro_m, micro_n, threads, tile_k)
    if any(type(v) is not int or v <= 0 for v in dims) or tile_k % 32 or depth % tile_k:
        raise ValueError("invalid packed integer matmul dimensions")
    if (
        threads not in (64, 128, 256, 512)
        or tile_m % micro_m
        or tile_n % micro_n
        or (tile_m // micro_m) * (tile_n // micro_n) != threads
    ):
        raise ValueError("invalid packed integer matmul ownership")
    parts = len(rhs_words)
    if (
        parts not in (1, 2)
        or len(rhs_scales) != parts
        or any(type(v) is not str or not v.strip() for v in (*rhs_words, *rhs_scales))
    ):
        raise ValueError("invalid packed integer matmul RHS expressions")
    if (
        micro_m * micro_n > 32
        or owner_axis not in ("column", "row")
        or group_order not in ("column", "row")
    ):
        raise ValueError("invalid packed integer matmul layout")
    if type(fixed_residual) is not bool:
        raise ValueError("invalid fixed residual mode")
    if type(signed_rhs) is not bool:
        raise ValueError("invalid signed RHS mode")
    blocks = tile_k // 32
    storage = (tile_m * 80 + parts * tile_n * 36) * blocks
    if storage > 32768:
        raise ValueError("packed integer matmul exceeds 32 KiB")
    rm, rn = tile_m // micro_m, tile_n // micro_n
    mr, nr = (
        (f"tx // {rn}", f"tx % {rn}")
        if owner_axis == "column"
        else (f"tx % {rm}", f"tx // {rm}")
    )
    grid = (
        (f"T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n})", "(by, bx)")
        if group_order == "row"
        else (
            f"T.ceildiv({columns}, {tile_n}), T.ceildiv({rows}, {tile_m})",
            "(bx, by)",
        )
    )
    lines = [
        'T.func_attr({"tensor.webgpu.loop_unroll":"explicit"})',
        f"with T.Kernel({grid[0]}, threads={threads}) as {grid[1]}:",
        "    tx = T.get_thread_binding()",
        f"    mr = {mr}",
        f"    nr = {nr}",
        f'    lhs = T.alloc_shared((2, {tile_m}, {8 * blocks}), "uint32")',
        f'    factors = T.alloc_shared((2, {tile_m}, {blocks}), "float32")',
        f'    offsets = T.alloc_shared((2, {tile_m}, {blocks}), "int32")',
        f'    rhs = T.alloc_shared(({parts}, {8 * blocks}, {tile_n}), "uint32")',
        f'    weight_scale = T.alloc_shared(({parts}, {blocks}, {tile_n}), "float32")',
    ]
    for part in range(parts):
        for i in range(micro_m):
            for j in range(micro_n):
                lines += [
                    f'    acc{part}_{i}_{j} = T.alloc_var("float32")',
                    f"    acc{part}_{i}_{j} = 0",
                ]
                for component in range(2):
                    lines += [
                        f'    dot{part}_{component}_{i}_{j} = T.alloc_var("int32")'
                    ]
    lines += [
        f"    for tile in T.serial({depth // tile_k}):",
        f"        for load_a in T.serial(T.ceildiv({tile_m * 8 * blocks}, {threads})):",
        f"            flat = load_a * {threads} + tx",
        f"            ar = flat // {8 * blocks}",
        f"            word = flat % {8 * blocks}",
        f"            if ar < {tile_m}:",
    ]
    for component in range(2):
        row = f"by * {tile_m} + ar"
        index = f"({row}) * {depth // 4} + tile * {8 * blocks} + word + {component * rows * depth // 4}"
        scaleindex = f"({row}) * {depth // 32} + tile * {blocks} + word // 8 + {component * rows * depth // 32}"
        lines += [
            f"                lhs[{component}, ar, word] = T.if_then_else({row} < {rows}, packed[{index}], T.uint32(0))",
            "                if word % 8 == 0:",
            f"                    factors[{component}, ar, word // 8] = T.if_then_else({row} < {rows}, scales[{scaleindex}], 0)",
            f"                    offsets[{component}, ar, word // 8] = T.if_then_else({row} < {rows}, sums[{scaleindex}], 0)",
        ]
    rhs_words_per_block = 8 if signed_rhs else 4
    lines += [
        f"        for load_b in T.serial(T.ceildiv({tile_n * rhs_words_per_block * blocks}, {threads})):",
        f"            flat = load_b * {threads} + tx",
        f"            br = flat // {rhs_words_per_block * blocks}",
        f"            word = flat % {rhs_words_per_block * blocks}",
        f"            if br < {tile_n}:",
    ]
    for part in range(parts):
        column = f"bx * {tile_n} + br"
        values = dict(
            column=column,
            block=f"tile * {blocks} + word // {rhs_words_per_block}",
            word=f"word % {rhs_words_per_block}",
        )
        lines += [
            f"                bits{part} = T.if_then_else({column} < {columns}, {rhs_words[part].format(**values)}, T.uint32(0))"
        ]
        if signed_rhs:
            lines += [f"                rhs[{part}, word, br] = bits{part}"]
        else:
            lines += [
                f"                rhs[{part}, (word // 4) * 8 + word % 4, br] = bits{part} & T.uint32(252645135)",
                f"                rhs[{part}, (word // 4) * 8 + word % 4 + 4, br] = (bits{part} >> 4) & T.uint32(252645135)",
            ]
        lines += [
            f"                if word % {rhs_words_per_block} == 0:",
            f"                    weight_scale[{part}, word // {rhs_words_per_block}, br] = T.if_then_else({column} < {columns}, {rhs_scales[part].format(**values)}, 0)",
        ]
    lines += ["        T.sync_threads()", f"        for block in T.serial({blocks}):"]
    for part in range(parts):
        for component in range(2):
            for i in range(micro_m):
                for j in range(micro_n):
                    lines += [f"            dot{part}_{component}_{i}_{j} = 0"]
    lines += ["            for word in T.unroll(8):"]
    for component in range(2):
        for i in range(micro_m):
            lines += [
                f"                left{component}_{i} = lhs[{component}, mr + {i * rm}, block * 8 + word]"
            ]
    for part in range(parts):
        for j in range(micro_n):
            lines += [
                f"                right{part}_{j} = rhs[{part}, block * 8 + word, nr + {j * rn}]"
            ]
        for component in range(2):
            for i in range(micro_m):
                for j in range(micro_n):
                    lines += [
                        f'                dot{part}_{component}_{i}_{j} = dot{part}_{component}_{i}_{j} + T.call_extern("int32", "dot4I8Packed", left{component}_{i}, right{part}_{j})'
                    ]
    for part in range(parts):
        for i in range(micro_m):
            for j in range(micro_n):
                row = f"mr + {i * rm}"
                col = f"nr + {j * rn}"
                values = [
                    f"T.cast(dot{part}_{component}_{i}_{j}"
                    + (
                        ""
                        if signed_rhs
                        else f" - 8 * offsets[{component}, {row}, block]"
                    )
                    + f', "float32") * factors[{component}, {row}, block]'
                    for component in range(2)
                ]
                if fixed_residual:
                    combined = (
                        f"dot{part}_0_{i}_{j} * 254 + dot{part}_1_{i}_{j}"
                        if signed_rhs
                        else f"(dot{part}_0_{i}_{j} - 8 * offsets[0, {row}, block]) * 254 + dot{part}_1_{i}_{j} - 8 * offsets[1, {row}, block]"
                    )
                    lines += [
                        f'            acc{part}_{i}_{j} = acc{part}_{i}_{j} + T.cast({combined}, "float32") * factors[1, {row}, block] * weight_scale[{part}, block, {col}]'
                    ]
                else:
                    lines += [
                        f"            acc{part}_{i}_{j} = acc{part}_{i}_{j} + ({values[0]} + {values[1]}) * weight_scale[{part}, block, {col}]"
                    ]
    lines += ["        T.sync_threads()"]
    for i in range(micro_m):
        for j in range(micro_n):
            row = f"by * {tile_m} + mr + {i * rm}"
            col = f"bx * {tile_n} + nr + {j * rn}"
            value = f"acc0_{i}_{j}"
            if parts == 2:
                value = f"({value}) / (1 + T.exp(-({value}))) * acc1_{i}_{j}"
            lines += [
                f"    if ({row} < {rows}) & ({col} < {columns}):",
                f"        out[({row}) * {columns} + {col}] = {value}",
            ]
    return "\n".join(lines)


def packed_integer_matmul(
    rows,
    depth,
    columns,
    read_word,
    read_scale,
    epilogue,
    *,
    parts=1,
    tile_m=16,
    tile_n=32,
    micro_m=2,
    micro_n=2,
    threads=128,
    owner_axis="column",
    group_order="row",
    tile_k=32,
    fixed_residual=False,
    signed_rhs=False,
):
    import tilelang.language as T

    dims = (rows, depth, columns, tile_m, tile_n, micro_m, micro_n, threads, tile_k)
    if (
        any((type(v) is not int or v <= 0 for v in dims))
        or tile_k % 32
        or depth % tile_k
    ):
        raise ValueError("invalid packed integer matmul dimensions")
    if (
        threads not in (64, 128, 256, 512)
        or tile_m % micro_m
        or tile_n % micro_n
        or (tile_m // micro_m * (tile_n // micro_n) != threads)
    ):
        raise ValueError("invalid packed integer matmul ownership")
    if (
        micro_m * micro_n > 32
        or owner_axis not in ("column", "row")
        or group_order not in ("column", "row")
    ):
        raise ValueError("invalid packed integer matmul layout")
    if type(fixed_residual) is not bool or type(signed_rhs) is not bool:
        raise ValueError("invalid packed integer matmul arithmetic")
    blocks = tile_k // 32
    if (tile_m * 80 + parts * tile_n * 36) * blocks > 32768:
        raise ValueError("packed integer matmul exceeds 32 KiB")
    rm, rn, rhs_words = (tile_m // micro_m, tile_n // micro_n, 8 if signed_rhs else 4)

    def stage_components(
        packed, scales, sums, lhs, factors, offsets, ar, word, by, tile
    ):
        for component in range(2):
            stage_component(
                packed,
                scales,
                sums,
                lhs,
                factors,
                offsets,
                component,
                ar,
                word,
                by,
                tile,
            )

    def stage_weights(weights, rhs, weight_scale, bx, br, tile, word):
        for part in range(parts):
            stage_rhs(weights[part], rhs, weight_scale, part, bx, br, tile, word)

    def clear_dots(dots):
        for value in dots.values():
            clear(value[0])

    def multiply(dots, lhs, rhs, mr, nr, block, word):
        left = {
            (component, i): lhs[component, mr + i * rm, block * 8 + word]
            for component, i in product(range(2), range(micro_m))
        }
        for part in range(parts):
            right = [rhs[part, block * 8 + word, nr + j * rn] for j in range(micro_n)]
            for component, i, j in product(range(2), range(micro_m), range(micro_n)):
                update(dots[part, component, i, j][0], left[component, i], right[j])

    def accumulate_dots(acc, dots, factors, offsets, weight_scale, block, mr, nr):
        for part, i, j in product(range(parts), range(micro_m), range(micro_n)):
            accumulate(
                acc[part, i, j][0],
                dots[part, 0, i, j][0],
                dots[part, 1, i, j][0],
                factors,
                offsets,
                weight_scale,
                part,
                block,
                mr + i * rm,
                nr + j * rn,
            )

    def stores(out, acc, by, bx, mr, nr):
        for i, j in product(range(micro_m), range(micro_n)):
            store(
                out,
                acc[0, i, j][0],
                acc[1, i, j][0] if parts == 2 else 0,
                by,
                bx,
                mr,
                nr,
                i,
                j,
            )

    @T.macro
    def stage_component(
        packed, scales, sums, lhs, factors, offsets, component, ar, word, by, tile
    ):
        row = by * tile_m + ar
        lhs[component, ar, word] = T.if_then_else(
            row < rows,
            packed[
                row * (depth // 4)
                + tile * (8 * blocks)
                + word
                + component * rows * depth // 4
            ],
            T.uint32(0),
        )
        if word % 8 == 0:
            factors[component, ar, word // 8] = T.if_then_else(
                row < rows,
                scales[
                    row * (depth // 32)
                    + tile * blocks
                    + word // 8
                    + component * rows * depth // 32
                ],
                0,
            )
            offsets[component, ar, word // 8] = T.if_then_else(
                row < rows,
                sums[
                    row * (depth // 32)
                    + tile * blocks
                    + word // 8
                    + component * rows * depth // 32
                ],
                0,
            )

    @T.macro
    def stage_rhs(w, rhs, weight_scale, part, bx, br, tile, word):
        bits = T.alloc_var("uint32")
        if bx * tile_n + br < columns:
            bits = read_word(
                w, bx * tile_n + br, tile * blocks + word // rhs_words, word % rhs_words
            )
        else:
            bits = T.uint32(0)
        if signed_rhs:
            rhs[part, word, br] = bits
        else:
            rhs[part, word // 4 * 8 + word % 4, br] = bits & T.uint32(252645135)
            rhs[part, word // 4 * 8 + word % 4 + 4, br] = bits >> 4 & T.uint32(
                252645135
            )
        if word % rhs_words == 0:
            if bx * tile_n + br < columns:
                weight_scale[part, word // rhs_words, br] = read_scale(
                    w, bx * tile_n + br, tile * blocks + word // rhs_words
                )
            else:
                weight_scale[part, word // rhs_words, br] = 0

    @T.macro
    def clear(dot: T.Ref):
        dot = 0

    @T.macro
    def update(dot: T.Ref, left, right):
        dot = dot + T.call_extern("int32", "dot4I8Packed", left, right)

    @T.macro
    def accumulate(
        acc: T.Ref, dot0, dot1, factors, offsets, weight_scale, part, block, row, col
    ):
        if fixed_residual:
            if signed_rhs:
                acc = (
                    acc
                    + T.cast(dot0 * 254 + dot1, "float32")
                    * factors[1, row, block]
                    * weight_scale[part, block, col]
                )
            else:
                acc = (
                    acc
                    + T.cast(
                        (dot0 - 8 * offsets[0, row, block]) * 254
                        + dot1
                        - 8 * offsets[1, row, block],
                        "float32",
                    )
                    * factors[1, row, block]
                    * weight_scale[part, block, col]
                )
        elif signed_rhs:
            acc = (
                acc
                + (
                    T.cast(dot0, "float32") * factors[0, row, block]
                    + T.cast(dot1, "float32") * factors[1, row, block]
                )
                * weight_scale[part, block, col]
            )
        else:
            acc = (
                acc
                + (
                    T.cast(dot0 - 8 * offsets[0, row, block], "float32")
                    * factors[0, row, block]
                    + T.cast(dot1 - 8 * offsets[1, row, block], "float32")
                    * factors[1, row, block]
                )
                * weight_scale[part, block, col]
            )

    @T.macro
    def store(out, value, up, by, bx, mr, nr, i, j):
        if (by * tile_m + mr + i * rm < rows) & (bx * tile_n + nr + j * rn < columns):
            epilogue(
                out,
                None,
                (by * tile_m + mr + i * rm) * columns + bx * tile_n + nr + j * rn,
                value,
                up,
            )

    @T.macro
    def matmul(packed, scales, sums, w, out, w2=None):
        T.func_attr({"tensor.webgpu.loop_unroll": "explicit"})
        with T.Kernel(
            T.ceildiv(
                rows if group_order == "row" else columns,
                tile_m if group_order == "row" else tile_n,
            ),
            T.ceildiv(
                columns if group_order == "row" else rows,
                tile_n if group_order == "row" else tile_m,
            ),
            threads=threads,
        ) as (axis0, axis1):
            by = axis0 if group_order == "row" else axis1
            bx = axis1 if group_order == "row" else axis0
            tx = T.get_thread_binding()
            mr = tx // rn if owner_axis == "column" else tx % rm
            nr = tx % rn if owner_axis == "column" else tx // rm
            lhs = T.alloc_shared((2, tile_m, 8 * blocks), "uint32")
            factors = T.alloc_shared((2, tile_m, blocks), "float32")
            offsets = T.alloc_shared((2, tile_m, blocks), "int32")
            rhs = T.alloc_shared((parts, 8 * blocks, tile_n), "uint32")
            weight_scale = T.alloc_shared((parts, blocks, tile_n), "float32")
            accum = registers((parts, micro_m, micro_n))
            dots = registers((parts, 2, micro_m, micro_n), "int32", initialize=False)
            for tile in T.serial(depth // tile_k):
                for load_a in T.serial(T.ceildiv(tile_m * 8 * blocks, threads)):
                    flat = load_a * threads + tx
                    ar = flat // (8 * blocks)
                    word = flat % (8 * blocks)
                    if ar < tile_m:
                        stage_components(
                            packed,
                            scales,
                            sums,
                            lhs,
                            factors,
                            offsets,
                            ar,
                            word,
                            by,
                            tile,
                        )
                for load_b in T.serial(T.ceildiv(tile_n * rhs_words * blocks, threads)):
                    flat = load_b * threads + tx
                    br = flat // (rhs_words * blocks)
                    word = flat % (rhs_words * blocks)
                    if br < tile_n:
                        stage_weights((w, w2), rhs, weight_scale, bx, br, tile, word)
                T.sync_threads()
                for block in T.serial(blocks):
                    clear_dots(dots)
                    for word in T.unroll(8):
                        multiply(dots, lhs, rhs, mr, nr, block, word)
                    accumulate_dots(
                        accum, dots, factors, offsets, weight_scale, block, mr, nr
                    )
                T.sync_threads()
            stores(out, accum, by, bx, mr, nr)

    return matmul
