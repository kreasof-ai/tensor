"""Packed outer product implementation for WebGPU kernels."""

from __future__ import annotations

from itertools import product

from tensor.compiler.webgpu.schedules.registers import registers
from tensor.compiler.webgpu.schedules.registers import shared_buffers


def packed_outer_product_schedule(
    rows,
    depth,
    columns,
    *,
    tile_m,
    tile_n,
    tile_k,
    micro_m,
    micro_n,
    threads,
    lhs_pad,
    rhs_pad,
    owner_axis,
    unroll,
    explicit_unroll,
    epilogue,
    lhs_value,
    rhs_value,
    rhs_transform,
    half_accum=False,
    group_order="column",
):
    """Adjacent K halves in shared u32; optional short private F16 FMA chains.

    Even/odd K lanes round independently in F16 for ``unroll`` terms before
    contributing to the complete-K F32 total. This changes accumulation
    precision and requires a producer's model-level acceptance, not just timing.
    """
    pk = tile_k // 2
    rm, rn = tile_m // micro_m, tile_n // micro_n
    if (pk * (tile_m + lhs_pad + tile_n + rhs_pad) * 4) > 32768:
        raise ValueError("packed outer-product storage exceeds 32 KiB")
    mr, nr = (
        (f"tx // {rn}", f"tx % {rn}")
        if owner_axis == "column"
        else (f"tx % {rm}", f"tx // {rm}")
    )
    lines = (
        ['T.func_attr({"tensor.webgpu.loop_unroll":"explicit"})']
        if explicit_unroll
        else []
    )
    grid = (
        f"T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n})"
        if group_order == "row"
        else f"T.ceildiv({columns}, {tile_n}), T.ceildiv({rows}, {tile_m})"
    )
    axes = "by, bx" if group_order == "row" else "bx, by"
    lines += [
        f"with T.Kernel({grid}, threads={threads}) as ({axes}):",
        "    tx = T.get_thread_binding()",
        f"    mr = {mr}",
        f"    nr = {nr}",
    ]
    for name, width in (("lhs", tile_m + lhs_pad), ("rhs", tile_n + rhs_pad)):
        lines += [f'    {name} = T.alloc_shared(({pk}, {width}), "uint32")']
    lines += [
        '    value0 = T.alloc_var("float32")',
        '    value1 = T.alloc_var("float32")',
    ]
    for i in range(micro_m):
        for j in range(micro_n):
            lines += [f'    acc{i}_{j} = T.alloc_var("float32")', f"    acc{i}_{j} = 0"]
            if half_accum:
                lines += [f'    partial{i}_{j} = T.alloc_var("float16x2")']
    lines += [
        f"    for tile in T.serial(T.ceildiv({depth}, {tile_k})):",
        f"        for load_a in T.serial(T.ceildiv({tile_m * pk}, {threads})):",
        f"            flat = load_a * {threads} + tx",
        f"            ar = flat // {pk}",
        f"            pair = flat % {pk}",
        f"            if ar < {tile_m}:",
    ]
    av = []
    for lane in range(2):
        row = f"by * {tile_m} + ar"
        k = f"tile * {tile_k} + pair * 2 + {lane}"
        expr = (lhs_value or f"x[({{row}}) * {depth} + ({{k}})]").format(row=row, k=k)
        av.append(
            f'T.if_then_else(({row} < {rows}) & ({k} < {depth}), T.cast({expr}, "float32"), 0)'
        )
    lines += [
        f'                lhs[pair, ar] = T.call_extern("uint32", "pack2x16float", T.call_extern("float32x2", "vec2<f32>", {av[0]}, {av[1]}))'
    ]
    lines += [
        f"        for load_b in T.serial(T.ceildiv({tile_n * pk}, {threads})):",
        f"            flat = load_b * {threads} + tx",
        f"            br = flat // {pk}",
        f"            pair = flat % {pk}",
        f"            if br < {tile_n}:",
    ]
    for lane in range(2):
        column = f"bx * {tile_n} + br"
        k = f"tile * {tile_k} + pair * 2 + {lane}"
        expr = (rhs_value or f"w[({{column}}) * {depth} + ({{k}})]").format(
            column=column, k=k, row="0"
        )
        lines += [
            f'                value{lane} = T.if_then_else(bx * {tile_n} + br < {columns}, T.if_then_else({k} < {depth}, T.cast({expr}, "float32"), 0), 0)'
        ]
    bv = [
        rhs_transform.format(value=f"value{lane}") if rhs_transform else f"value{lane}"
        for lane in range(2)
    ]
    lines += [
        f'                rhs[pair, br] = T.call_extern("uint32", "pack2x16float", T.call_extern("float32x2", "vec2<f32>", {bv[0]}, {bv[1]}))'
    ]
    lines += [
        "        T.sync_threads()",
        f"        for chunk in T.serial({pk // unroll}):",
    ]
    if half_accum:
        for i in range(micro_m):
            for j in range(micro_n):
                lines += [
                    f'            partial{i}_{j} = T.call_extern("float16x2", "vec2<f16>", T.cast(0, "float16"))'
                ]
    lines += [
        f"            for u in T.unroll({unroll}):",
        f"                kk = chunk * {unroll} + u",
    ]
    for name, width, stride, count in (
        ("left", "lhs", rm, micro_m),
        ("right", "rhs", rn, micro_n),
    ):
        index = "mr" if name == "left" else "nr"
        for i in range(count):
            expr = f"{width}[kk, {index} + {i * stride}]"
            lines += [f"                {name}{i} = {expr}"]
    for i in range(micro_m):
        for j in range(micro_n):
            if half_accum:
                lines += [
                    f'                partial{i}_{j} = T.call_extern("float16x2", "tensor_fma2_f16_vec", left{i}, right{j}, partial{i}_{j})'
                ]
            else:
                lines += [
                    f'                acc{i}_{j} = T.call_extern("float32", "tensor_dot2_f16_add", left{i}, right{j}, acc{i}_{j})'
                ]
    if half_accum:
        for i in range(micro_m):
            for j in range(micro_n):
                lines += [
                    f'            acc{i}_{j} = acc{i}_{j} + T.call_extern("float32", "tensor_sum2_f16_vec", partial{i}_{j})'
                ]
    lines += ["        T.sync_threads()"]
    for i in range(micro_m):
        for j in range(micro_n):
            row = f"by * {tile_m} + mr + {i * rm}"
            col = f"bx * {tile_n} + nr + {j * rn}"
            value = (
                f'T.max(acc{i}_{j} + T.cast(bias[{col}], "float32"), 0)'
                if epilogue == "linear"
                else f"acc{i}_{j}"
            )
            lines += [
                f"    if ({row} < {rows}) & ({col} < {columns}):",
                f"        out[({row}) * {columns} + {col}] = {value}",
            ]
    return "\n".join(lines)


def packed_outer_product(
    rows,
    depth,
    columns,
    read_lhs,
    read_rhs,
    epilogue,
    *,
    parts,
    tile_m,
    tile_n,
    tile_k,
    micro_m,
    micro_n,
    threads,
    lhs_pad,
    rhs_pad,
    owner_axis,
    unroll,
    explicit_unroll,
    half_accum=False,
    group_order="column",
):
    import tilelang.language as T

    pk, rm, rn = (tile_k // 2, tile_m // micro_m, tile_n // micro_n)
    if pk * (tile_m + lhs_pad + parts * (tile_n + rhs_pad)) * 4 > 32768:
        raise ValueError("packed outer-product storage exceeds 32 KiB")

    def stage(weights, rhs, bx, tile, tx):
        for part in range(parts):
            stage_b(weights[part], rhs[part], bx, tile, tx)

    def multiply(acc, lhs, rhs, mr, nr, kk):
        left = [lhs[kk, mr + i * rm] for i in range(micro_m)]
        for part in range(parts):
            right = [rhs[part][kk, nr + j * rn] for j in range(micro_n)]
            for i, j in product(range(micro_m), range(micro_n)):
                update(acc[part, i, j][0], left[i], right[j])

    def clear_partials(partials):
        for value in partials.values():
            clear_half(value[0])

    def add_partials(acc, partials):
        for key in acc:
            add_half(acc[key][0], partials[key][0])

    def stores(out, residual, acc, by, bx, mr, nr):
        for i, j in product(range(micro_m), range(micro_n)):
            store(
                out,
                residual,
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
    def stage_a(x, lhs, by, tile, tx):
        for load_a in T.serial(T.ceildiv(tile_m * pk, threads)):
            flat = load_a * threads + tx
            ar = flat // pk
            pair = flat % pk
            if ar < tile_m:
                value0 = T.alloc_var("float32")
                value1 = T.alloc_var("float32")
                if (by * tile_m + ar < rows) & (tile * tile_k + pair * 2 < depth):
                    value0 = T.cast(
                        read_lhs(
                            x[(by * tile_m + ar) * depth + tile * tile_k + pair * 2]
                        ),
                        "float32",
                    )
                else:
                    value0 = 0
                if (by * tile_m + ar < rows) & (tile * tile_k + pair * 2 + 1 < depth):
                    value1 = T.cast(
                        read_lhs(
                            x[(by * tile_m + ar) * depth + tile * tile_k + pair * 2 + 1]
                        ),
                        "float32",
                    )
                else:
                    value1 = 0
                lhs[pair, ar] = T.call_extern(
                    "uint32",
                    "pack2x16float",
                    T.call_extern("float32x2", "vec2<f32>", value0, value1),
                )

    @T.macro
    def stage_b(w, rhs, bx, tile, tx):
        for load_b in T.serial(T.ceildiv(tile_n * pk, threads)):
            flat = load_b * threads + tx
            br = flat // pk
            pair = flat % pk
            if br < tile_n:
                value0 = T.alloc_var("float32")
                value1 = T.alloc_var("float32")
                if bx * tile_n + br < columns:
                    if tile * tile_k + pair * 2 < depth:
                        value0 = T.cast(
                            read_rhs(w, bx * tile_n + br, tile * tile_k + pair * 2),
                            "float32",
                        )
                    else:
                        value0 = 0
                else:
                    value0 = 0
                if bx * tile_n + br < columns:
                    if tile * tile_k + pair * 2 + 1 < depth:
                        value1 = T.cast(
                            read_rhs(w, bx * tile_n + br, tile * tile_k + pair * 2 + 1),
                            "float32",
                        )
                    else:
                        value1 = 0
                else:
                    value1 = 0
                rhs[pair, br] = T.call_extern(
                    "uint32",
                    "pack2x16float",
                    T.call_extern("float32x2", "vec2<f32>", value0, value1),
                )

    @T.macro
    def update(acc: T.Ref, left, right):
        if half_accum:
            acc = T.call_extern("float16x2", "tensor_fma2_f16_vec", left, right, acc)
        else:
            acc = T.call_extern("float32", "tensor_dot2_f16_add", left, right, acc)

    @T.macro
    def clear_half(acc: T.Ref):
        acc = T.call_extern("float16x2", "vec2<f16>", T.cast(0, "float16"))

    @T.macro
    def add_half(acc: T.Ref, partial):
        acc = acc + T.call_extern("float32", "tensor_sum2_f16_vec", partial)

    @T.macro
    def store(out, residual, value, up, by, bx, mr, nr, i, j):
        if (by * tile_m + mr + i * rm < rows) & (bx * tile_n + nr + j * rn < columns):
            epilogue(
                out,
                residual,
                (by * tile_m + mr + i * rm) * columns + bx * tile_n + nr + j * rn,
                value,
                up,
            )

    @T.macro
    def matmul(x, w, out, w2=None, residual=None):
        if explicit_unroll:
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
            lhs = T.alloc_shared((pk, tile_m + lhs_pad), "uint32")
            rhs = shared_buffers((pk, tile_n + rhs_pad), "uint32", parts)
            accum = registers((parts, micro_m, micro_n))
            if half_accum:
                partials = registers(
                    (parts, micro_m, micro_n), "float16x2", initialize=False
                )
            for tile in T.serial(T.ceildiv(depth, tile_k)):
                stage_a(x, lhs, by, tile, tx)
                stage((w, w2), rhs, bx, tile, tx)
                T.sync_threads()
                for chunk in T.serial(pk // unroll):
                    if half_accum:
                        clear_partials(partials)
                    for u in T.unroll(unroll):
                        kk = chunk * unroll + u
                        multiply(
                            partials if half_accum else accum, lhs, rhs, mr, nr, kk
                        )
                    if half_accum:
                        add_partials(accum, partials)
                T.sync_threads()
            stores(out, residual, accum, by, bx, mr, nr)

    return matmul
