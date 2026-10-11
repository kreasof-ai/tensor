"""Register matmul implementation for WebGPU kernels."""

from __future__ import annotations

from itertools import product

from tensor.compiler.webgpu.schedules.registers import registers
from tensor.compiler.webgpu.schedules.registers import shared_buffers
from tensor.compiler.webgpu.schedules.registers import vector


def register_matmul_schedule(
    rows,
    depth,
    columns,
    lhs_value,
    rhs_value,
    *,
    tile_m=16,
    tile_n=32,
    tile_k=32,
    threads=128,
    pad=0,
    lhs_pad=0,
    lhs_transpose=False,
    dot_width=1,
    unroll=False,
):
    """Reusable SIMT schedule with accumulators retained across all K tiles.

    Producers supply load/rounding expressions; this schedule owns distribution,
    shared layout, barriers and register microtiles. No GPU/compiler imports.
    """
    if (
        any(
            type(v) is not int or v <= 0
            for v in (rows, depth, columns, tile_m, tile_n, tile_k, threads)
        )
        or type(pad) is not int
        or pad < 0
    ):
        raise ValueError("register matmul dimensions must be positive integers")
    if (
        type(lhs_pad) is not int
        or lhs_pad < 0
        or type(lhs_transpose) is not bool
        or type(unroll) is not bool
        or dot_width not in (1, 4)
        or tile_k % dot_width
    ):
        raise ValueError("invalid register matmul shared layout or dot width")
    nr = tile_n // 2
    if nr == 0:
        raise ValueError("invalid register matmul tile")
    if tile_n % 2 or threads % nr or tile_m % (threads // nr) or depth % tile_k:
        raise ValueError("invalid register matmul tile")
    row_stride = threads // nr
    micro_m = tile_m // row_stride
    initialize = "\n".join(
        f'    acc{i}{j} = T.alloc_var("float32")\n    acc{i}{j} = 0'
        for i in range(micro_m)
        for j in range(2)
    )

    def lhs_index(row, k):
        return f"[{k}, {row}]" if lhs_transpose else f"[{row}, {k}]"

    if dot_width == 1:
        loads = "\n".join(
            f'            left{i} = T.cast(lhs{lhs_index(f"mr + {i * row_stride}", "kk")}, "float32")'
            for i in range(micro_m)
        )
        loads += "\n" + "\n".join(
            f'            right{j} = T.cast(rhs[kk, nr + {j * nr}], "float32")'
            for j in range(2)
        )
        multiply = "\n".join(
            f"            acc{i}{j} = acc{i}{j} + left{i} * right{j}"
            for i in range(micro_m)
            for j in range(2)
        )
    else:
        loads = "\n".join(
            f'            left{i} = T.call_extern("float32x4", "vec4<f32>", '
            + ", ".join(
                f'T.cast(lhs{lhs_index(f"mr + {i * row_stride}", f"kk * 4 + {lane}")}, "float32")'
                for lane in range(4)
            )
            + ")"
            for i in range(micro_m)
        )
        loads += "\n" + "\n".join(
            f'            right{j} = T.call_extern("float32x4", "vec4<f32>", '
            + ", ".join(
                f'T.cast(rhs[kk * 4 + {lane}, nr + {j * nr}], "float32")'
                for lane in range(4)
            )
            + ")"
            for j in range(2)
        )
        multiply = "\n".join(
            f'            acc{i}{j} = acc{i}{j} + T.call_extern("float32", "dot", left{i}, right{j})'
            for i in range(micro_m)
            for j in range(2)
        )
    lhs_shape = (
        (tile_k, tile_m + lhs_pad) if lhs_transpose else (tile_m, tile_k + lhs_pad)
    )
    stores = "\n".join(
        f"""    if by * {tile_m} + mr + {i * row_stride} < {rows}:
        if bx * {tile_n} + nr + {j * nr} < {columns}:
            out[(by * {tile_m} + mr + {i * row_stride}) * {columns} + bx * {tile_n} + nr + {j * nr}] = acc{i}{j}"""
        for i in range(micro_m)
        for j in range(2)
    )
    return f"""with T.Kernel(T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n}), threads={threads}) as (by, bx):
    tx = T.get_thread_binding()
    mr = tx // {nr}
    nr = tx % {nr}
    lhs = T.alloc_shared({lhs_shape}, "float16")
    rhs = T.alloc_shared(({tile_k}, {tile_n + pad}), "float16")
    value = T.alloc_var("float32")
{initialize}
    for tile in T.serial({depth // tile_k}):
        for i, j in T.Parallel({tile_m}, {tile_k}):
            value = T.if_then_else(by * {tile_m} + i < {rows}, x[(by * {tile_m} + i) * {depth} + tile * {tile_k} + j], 0)
            lhs{lhs_index("i", "j")} = {lhs_value}
        for i, j in T.Parallel({tile_n}, {tile_k}):
            value = {rhs_value}
            rhs[j, i] = value
        T.sync_threads()
        for kk in T.{"unroll" if unroll else "serial"}({tile_k // dot_width}):
{loads}
{multiply}
        T.sync_threads()
{stores}"""


def register_matmul(
    rows,
    depth,
    columns,
    read_lhs,
    read_rhs,
    epilogue,
    *,
    parts=1,
    tile_m=16,
    tile_n=32,
    tile_k=32,
    threads=128,
    pad=0,
    lhs_pad=0,
    lhs_transpose=False,
    dot_width=1,
    unroll=False,
    explicit_unroll=False,
):
    import tilelang.language as T

    if (
        any(
            (
                type(v) is not int or v <= 0
                for v in (rows, depth, columns, tile_m, tile_n, tile_k, threads)
            )
        )
        or type(pad) is not int
        or pad < 0
    ):
        raise ValueError("register matmul dimensions must be positive integers")
    if (
        type(lhs_pad) is not int
        or lhs_pad < 0
        or type(lhs_transpose) is not bool
        or (type(unroll) is not bool)
        or (dot_width not in (1, 4))
        or tile_k % dot_width
    ):
        raise ValueError("invalid register matmul shared layout or dot width")
    nr = tile_n // 2
    if (
        not nr
        or tile_n % 2
        or threads % nr
        or tile_m % (threads // nr)
        or depth % tile_k
    ):
        raise ValueError("invalid register matmul tile")
    rm, mm = (threads // nr, tile_m // (threads // nr))
    lhs_shape = (
        (tile_k, tile_m + lhs_pad) if lhs_transpose else (tile_m, tile_k + lhs_pad)
    )
    loop = T.unroll if unroll else T.serial

    def multiply(acc, lhs, rhs, mr, nc, kk):
        left = [
            vector(
                [
                    T.cast(
                        lhs[kk * dot_width + lane, mr + i * rm]
                        if lhs_transpose
                        else lhs[mr + i * rm, kk * dot_width + lane],
                        "float32",
                    )
                    for lane in range(dot_width)
                ]
            )
            for i in range(mm)
        ]
        for part in range(parts):
            right = [
                vector(
                    [
                        T.cast(rhs[part][kk * dot_width + lane, nc + j * nr], "float32")
                        for lane in range(dot_width)
                    ]
                )
                for j in range(2)
            ]
            for i, j in product(range(mm), range(2)):
                update(acc[part, i, j][0], left[i], right[j])

    def stage(weights, rhs, bx, tile):
        for part in range(parts):
            stage_rhs(weights[part], rhs[part], bx, tile)

    def stores(out, residual, acc, by, bx, mr, nc):
        for i, j in product(range(mm), range(2)):
            store(
                out,
                residual,
                acc[0, i, j][0],
                acc[1, i, j][0] if parts == 2 else 0,
                by,
                bx,
                mr,
                nc,
                i,
                j,
            )

    @T.macro
    def stage_lhs(x, lhs, by, tile):
        for i, j in T.Parallel(tile_m, tile_k):
            value = T.if_then_else(
                by * tile_m + i < rows,
                x[(by * tile_m + i) * depth + tile * tile_k + j],
                0,
            )
            if lhs_transpose:
                lhs[j, i] = read_lhs(value)
            else:
                lhs[i, j] = read_lhs(value)

    @T.macro
    def stage_rhs(w, rhs, bx, tile):
        for i, j in T.Parallel(tile_n, tile_k):
            if bx * tile_n + i < columns:
                rhs[j, i] = read_rhs(w, bx * tile_n + i, tile * tile_k + j)
            else:
                rhs[j, i] = 0

    @T.macro
    def update(acc: T.Ref, left, right):
        if dot_width == 1:
            acc = acc + left * right
        else:
            acc = acc + T.call_extern("float32", "dot", left, right)

    @T.macro
    def store(out, residual, acc, up, by, bx, mr, nc, i, j):
        if by * tile_m + mr + i * rm < rows:
            if bx * tile_n + nc + j * nr < columns:
                epilogue(
                    out,
                    residual,
                    (by * tile_m + mr + i * rm) * columns + bx * tile_n + nc + j * nr,
                    acc,
                    up,
                )

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
            mr = tx // nr
            nc = tx % nr
            lhs = T.alloc_shared(lhs_shape, "float16")
            rhs = shared_buffers((tile_k, tile_n + pad), "float16", parts)
            accum = registers((parts, mm, 2))
            for tile in T.serial(depth // tile_k):
                stage_lhs(x, lhs, by, tile)
                stage((w, w2), rhs, bx, tile)
                T.sync_threads()
                for kk in loop(tile_k // dot_width):
                    multiply(accum, lhs, rhs, mr, nc, kk)
                T.sync_threads()
            stores(out, residual, accum, by, bx, mr, nc)

    return matmul
