"""Partitioned implementation for WebGPU kernels."""

from __future__ import annotations

from itertools import product

from tensor.compiler.webgpu.schedules.registers import registers
from tensor.compiler.webgpu.schedules.registers import shared_buffers
from tensor.compiler.webgpu.schedules.registers import vector


def partitioned_matmul_schedule(
    rows,
    depth,
    columns,
    lhs_value,
    rhs_value,
    *,
    tile_m=8,
    tile_n=16,
    threads=128,
    partitions=8,
    unroll=4,
    dot_width=1,
    owner_axis="column",
    k_layout="blocked",
):
    """Direct-load SIMT microtiles with one workgroup-local K reduction.

    Load expressions use {row}, {column}, and {k} placeholders. The producer
    owns operand precision. Partitions never require a second dispatch or atomics.
    """
    dimensions = (rows, depth, columns, tile_m, tile_n, threads, partitions, unroll)
    if any(type(v) is not int or v <= 0 for v in dimensions):
        raise ValueError("partitioned matmul dimensions must be positive integers")
    if (
        partitions & (partitions - 1)
        or threads % partitions
        or depth % (partitions * unroll)
    ):
        raise ValueError("invalid K partition or unroll")
    if dot_width not in (1, 2, 4) or unroll % dot_width:
        raise ValueError("invalid partitioned matmul dot width")
    if owner_axis not in ("row", "column") or k_layout not in ("blocked", "striped"):
        raise ValueError("invalid partitioned matmul distribution")
    owners = threads // partitions
    nc = (
        min(tile_n, owners) if owner_axis == "column" else owners // min(tile_m, owners)
    )
    if owners % nc or tile_m % (owners // nc) or tile_n % nc:
        raise ValueError("invalid partitioned output tile")
    rm = owners // nc
    mm = tile_m // rm
    nn = tile_n // nc
    if mm * nn > 64 or tile_m * tile_n * partitions * 4 > 32768:
        raise ValueError("partitioned matmul resource limit")
    lines = [
        f"with T.Kernel(T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n}), threads={threads}) as (by, bx):",
        "    tx = T.get_thread_binding()",
        f"    lane = tx % {partitions}",
        f"    mr = tx // {partitions * nc}",
        f"    nr = (tx // {partitions}) % {nc}",
        f'    scratch = T.alloc_shared(({tile_m * tile_n * partitions},), "float32")',
    ]
    for i in range(mm):
        for j in range(nn):
            lines += [f'    acc{i}_{j} = T.alloc_var("float32")', f"    acc{i}_{j} = 0"]
    lines += [
        f"    for tile in T.serial({depth // (partitions * unroll)}):",
        f"        for u in T.unroll({unroll // dot_width}):",
    ]
    kval = lambda v: (
        f"tile * {partitions * unroll} + lane * {unroll} + u * {dot_width} + {v}"
        if k_layout == "blocked"
        else f"tile * {partitions * unroll} + lane * {dot_width} + u * {partitions * dot_width} + {v}"
    )
    for i in range(mm):
        row = f"by * {tile_m} + mr + {i * rm}"
        exprs = [
            f"T.if_then_else({row} < {rows}, {lhs_value.format(row=row, column='0', k=kval(v))}, 0)"
            for v in range(dot_width)
        ]
        expr = (
            exprs[0]
            if dot_width == 1
            else f'T.call_extern("float32x{dot_width}", "vec{dot_width}<f32>", '
            + ", ".join(exprs)
            + ")"
        )
        lines += [f"            left{i} = {expr}"]
    for j in range(nn):
        col = f"bx * {tile_n} + nr + {j * nc}"
        exprs = [
            f"T.if_then_else({col} < {columns}, {rhs_value.format(row='0', column=col, k=kval(v))}, 0)"
            for v in range(dot_width)
        ]
        expr = (
            exprs[0]
            if dot_width == 1
            else f'T.call_extern("float32x{dot_width}", "vec{dot_width}<f32>", '
            + ", ".join(exprs)
            + ")"
        )
        lines += [f"            right{j} = {expr}"]
    for i in range(mm):
        for j in range(nn):
            product = (
                f"left{i} * right{j}"
                if dot_width == 1
                else f'T.call_extern("float32", "dot", left{i}, right{j})'
            )
            lines += [f"            acc{i}_{j} = acc{i}_{j} + {product}"]
    for i in range(mm):
        for j in range(nn):
            lines += [
                f"    scratch[((mr + {i * rm}) * {tile_n} + nr + {j * nc}) * {partitions} + lane] = acc{i}_{j}"
            ]
    lines += [
        "    T.sync_threads()",
        '    total = T.alloc_var("float32")',
        f"    for index in T.serial(T.ceildiv({tile_m * tile_n}, {threads})):",
        f"        slot = index * {threads} + tx",
        f"        if slot < {tile_m * tile_n}:",
        "            total = 0",
        f"            for part in T.unroll({partitions}):",
        f"                total = total + scratch[slot * {partitions} + part]",
        f"            if (by * {tile_m} + slot // {tile_n} < {rows}) & (bx * {tile_n} + slot % {tile_n} < {columns}):",
        f"                out[(by * {tile_m} + slot // {tile_n}) * {columns} + bx * {tile_n} + slot % {tile_n}] = total",
    ]
    return "\n".join(lines)


def partitioned_matmul(
    rows,
    depth,
    columns,
    read_lhs,
    read_rhs,
    epilogue,
    *,
    parts=1,
    tile_m=8,
    tile_n=16,
    threads=128,
    partitions=8,
    unroll=4,
    dot_width=1,
    owner_axis="column",
    k_layout="blocked",
    explicit_unroll=False,
):
    import tilelang.language as T

    if any(
        (
            type(v) is not int or v <= 0
            for v in (rows, depth, columns, tile_m, tile_n, threads, partitions, unroll)
        )
    ):
        raise ValueError("partitioned matmul dimensions must be positive integers")
    if (
        partitions & partitions - 1
        or threads % partitions
        or depth % (partitions * unroll)
        or (dot_width not in (1, 2, 4))
        or unroll % dot_width
    ):
        raise ValueError("invalid K partition or unroll")
    if owner_axis not in ("row", "column") or k_layout not in ("blocked", "striped"):
        raise ValueError("invalid partitioned matmul distribution")
    owners = threads // partitions
    nc = (
        min(tile_n, owners) if owner_axis == "column" else owners // min(tile_m, owners)
    )
    if not nc or owners % nc or tile_m % (owners // nc) or tile_n % nc:
        raise ValueError("invalid partitioned output tile")
    rm, mm, nn = (owners // nc, tile_m // (owners // nc), tile_n // nc)
    if mm * nn > 64 or parts * tile_m * tile_n * partitions * 4 > 32768:
        raise ValueError("partitioned matmul resource limit")

    def depth_index(tile, lane, u, v):
        return (
            tile * (partitions * unroll) + lane * unroll + u * dot_width + v
            if k_layout == "blocked"
            else tile * (partitions * unroll)
            + lane * dot_width
            + u * (partitions * dot_width)
            + v
        )

    def multiply(acc, x, weights, by, bx, mr, nr, tile, lane, u):
        left = [
            vector(
                [
                    lhs_value(
                        x, by * tile_m + mr + i * rm, depth_index(tile, lane, u, v)
                    )
                    for v in range(dot_width)
                ]
            )
            for i in range(mm)
        ]
        for part in range(parts):
            right = [
                vector(
                    [
                        rhs_value(
                            weights[part],
                            bx * tile_n + nr + j * nc,
                            depth_index(tile, lane, u, v),
                        )
                        for v in range(dot_width)
                    ]
                )
                for j in range(nn)
            ]
            for i, j in product(range(mm), range(nn)):
                update(acc[part, i, j][0], left[i], right[j])

    def spills(acc, scratch, mr, nr, lane):
        for part, i, j in product(range(parts), range(mm), range(nn)):
            spill(scratch[part], acc[part, i, j][0], mr, nr, lane, i, j)

    def merges(scratch, totals, slot):
        for part in range(parts):
            merge(scratch[part], totals[part,][0], slot)

    @T.macro
    def lhs_value(x, row, kk):
        result = T.alloc_var("float32")
        if row < rows:
            result = read_lhs(x[row * depth + kk])
        else:
            result = 0
        return result

    @T.macro
    def rhs_value(w, col, kk):
        result = T.alloc_var("float32")
        if col < columns:
            result = read_rhs(w, col, kk)
        else:
            result = 0
        return result

    @T.macro
    def update(acc: T.Ref, left, right):
        if dot_width == 1:
            acc = acc + left * right
        else:
            acc = acc + T.call_extern("float32", "dot", left, right)

    @T.macro
    def spill(scratch, value, mr, nr, lane, i, j):
        scratch[((mr + i * rm) * tile_n + nr + j * nc) * partitions + lane] = value

    @T.macro
    def merge(scratch, total: T.Ref, slot):
        total = 0
        for part in T.unroll(partitions):
            total = total + scratch[slot * partitions + part]

    @T.macro
    def matmul(x, w, out, w2=None, residual=None):
        if explicit_unroll:
            T.func_attr({"tensor.webgpu.loop_unroll": "explicit"})
        with T.Kernel(
            T.ceildiv(rows, tile_m), T.ceildiv(columns, tile_n), threads=threads
        ) as (
            by,
            bx,
        ):
            tx = T.get_thread_binding()
            lane = tx % partitions
            mr = tx // (partitions * nc)
            nr = tx // partitions % nc
            scratch = shared_buffers((tile_m * tile_n * partitions,), "float32", parts)
            accum = registers((parts, mm, nn))
            for tile in T.serial(depth // (partitions * unroll)):
                for u in T.unroll(unroll // dot_width):
                    multiply(accum, x, (w, w2), by, bx, mr, nr, tile, lane, u)
            spills(accum, scratch, mr, nr, lane)
            T.sync_threads()
            totals = registers((parts,), initialize=False)
            for index in T.serial(T.ceildiv(tile_m * tile_n, threads)):
                slot = index * threads + tx
                if slot < tile_m * tile_n:
                    merges(scratch, totals, slot)
                    if (by * tile_m + slot // tile_n < rows) & (
                        bx * tile_n + slot % tile_n < columns
                    ):
                        epilogue(
                            out,
                            residual,
                            (by * tile_m + slot // tile_n) * columns
                            + bx * tile_n
                            + slot % tile_n,
                            totals[0,][0],
                            totals[1,][0] if parts == 2 else 0,
                        )

    return matmul
