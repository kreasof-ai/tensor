"""Outer product implementation for WebGPU kernels."""

from __future__ import annotations

from itertools import product

from tensor.compiler.webgpu.schedules.packed_outer_product import (
    packed_outer_product_schedule,
)
from tensor.compiler.webgpu.schedules.packed_outer_product import packed_outer_product
from tensor.compiler.webgpu.schedules.registers import registers
from tensor.compiler.webgpu.schedules.registers import shared_buffers
from tensor.compiler.webgpu.schedules.registers import vector


def outer_product_matmul_schedule(
    rows,
    depth,
    columns,
    *,
    tile_m=64,
    tile_n=64,
    tile_k=16,
    micro_m=4,
    micro_n=4,
    threads=256,
    lhs_layout="km",
    lhs_pad=0,
    rhs_pad=0,
    owner_axis="column",
    unroll=4,
    fma=True,
    dtype="float32",
    epilogue="gemm",
    explicit_unroll=False,
    lhs_value=None,
    rhs_value=None,
    rhs_transform=None,
    dot_width=1,
    packed_pairs=False,
    half_accum=False,
    group_order="column",
):
    """Staged SIMT outer products with complete-K private FP32 accumulators.

    Cooperative loads preserve row-major A and transposed row-major B storage.
    Shared layouts and output ownership are independent scheduling choices.
    The generated source uses x[M*K], w[N*K], optional bias[N], and out[M*N].
    """

    dims = (
        rows,
        depth,
        columns,
        tile_m,
        tile_n,
        tile_k,
        micro_m,
        micro_n,
        threads,
        unroll,
    )
    if any(type(v) is not int or v <= 0 for v in dims):
        raise ValueError("invalid outer-product dimensions")
    if (
        threads not in (64, 128, 256, 512)
        or tile_m % micro_m
        or tile_n % micro_n
        or (tile_m // micro_m) * (tile_n // micro_n) != threads
    ):
        raise ValueError("invalid outer-product ownership")
    if micro_m * micro_n > 64 or tile_k % unroll or unroll > 16:
        raise ValueError("invalid outer-product register footprint or unroll")
    if (
        type(dot_width) is not int
        or dot_width not in (1, 2, 4)
        or tile_k % (unroll * dot_width)
    ):
        raise ValueError("invalid outer-product dot width")
    if type(packed_pairs) is not bool or (
        packed_pairs and (dtype != "float16" or dot_width != 2 or lhs_layout != "km")
    ):
        raise ValueError(
            "packed outer-product pairs require F16, K-major layout and dot width two"
        )
    if type(half_accum) is not bool or (half_accum and not packed_pairs):
        raise ValueError("half accumulation requires packed F16 pairs")
    if group_order not in ("row", "column"):
        raise ValueError("invalid outer-product group order")
    if (
        lhs_layout not in ("mk", "km")
        or owner_axis not in ("row", "column")
        or dtype not in ("float32", "float16")
        or epilogue not in ("gemm", "linear")
        or type(fma) is not bool
        or type(explicit_unroll) is not bool
    ):
        raise ValueError("invalid outer-product layout or arithmetic")
    if any(type(v) is not int or v < 0 for v in (lhs_pad, rhs_pad)):
        raise ValueError("invalid outer-product padding")
    if lhs_value is not None and (type(lhs_value) is not str or not lhs_value.strip()):
        raise ValueError("invalid outer-product activation expression")
    if any(
        v is not None and (type(v) is not str or not v.strip())
        for v in (rhs_value, rhs_transform)
    ):
        raise ValueError("invalid outer-product weight expression")
    if packed_pairs:
        return packed_outer_product_schedule(
            rows,
            depth,
            columns,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            micro_m=micro_m,
            micro_n=micro_n,
            threads=threads,
            lhs_pad=lhs_pad,
            rhs_pad=rhs_pad,
            owner_axis=owner_axis,
            unroll=unroll,
            explicit_unroll=explicit_unroll,
            epilogue=epilogue,
            lhs_value=lhs_value,
            rhs_value=rhs_value,
            rhs_transform=rhs_transform,
            half_accum=half_accum,
            group_order=group_order,
        )
    activation = (lhs_value or f"x[({{row}}) * {depth} + ({{k}})]").format(
        row=f"by * {tile_m} + ar", k=f"tile * {tile_k} + ak"
    )
    weight = (rhs_value or f"w[({{column}}) * {depth} + ({{k}})]").format(
        column=f"bx * {tile_n} + br", row="0", k=f"tile * {tile_k} + bk"
    )
    lhs_shape = (
        (tile_k, tile_m + lhs_pad) if lhs_layout == "km" else (tile_m, tile_k + lhs_pad)
    )
    rhs_shape = (tile_k, tile_n + rhs_pad)
    size = (lhs_shape[0] * lhs_shape[1] + rhs_shape[0] * rhs_shape[1]) * (
        4 if dtype == "float32" else 2
    )
    if size > 32768:
        raise ValueError("outer-product shared storage exceeds 32 KiB")
    rm, rn = tile_m // micro_m, tile_n // micro_n
    mr, nr = (
        ("tx // " + str(rn), "tx % " + str(rn))
        if owner_axis == "column"
        else ("tx % " + str(rm), "tx // " + str(rm))
    )
    idx = lambda row, k: f"[{k}, {row}]" if lhs_layout == "km" else f"[{row}, {k}]"
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
        f'    lhs = T.alloc_shared({lhs_shape}, "{dtype}")',
        f'    rhs = T.alloc_shared({rhs_shape}, "{dtype}")',
    ]
    if rhs_transform is not None:
        lines += ['    value = T.alloc_var("float32")']
    for i in range(micro_m):
        for j in range(micro_n):
            lines += [f'    acc{i}_{j} = T.alloc_var("float32")', f"    acc{i}_{j} = 0"]
    lines += [
        f"    for tile in T.serial(T.ceildiv({depth}, {tile_k})):",
        f"        for load_a in T.serial(T.ceildiv({tile_m * tile_k}, {threads})):",
        f"            flat_a = load_a * {threads} + tx",
        f"            ar = flat_a // {tile_k}",
        f"            ak = flat_a % {tile_k}",
        f"            if ar < {tile_m}:",
        f'                lhs{idx("ar", "ak")} = T.if_then_else((by * {tile_m} + ar < {rows}) & (tile * {tile_k} + ak < {depth}), T.cast({activation}, "{dtype}"), T.cast(0, "{dtype}"))',
        f"        for load_b in T.serial(T.ceildiv({tile_n * tile_k}, {threads})):",
        f"            flat_b = load_b * {threads} + tx",
        f"            br = flat_b // {tile_k}",
        f"            bk = flat_b % {tile_k}",
        f"            if br < {tile_n}:",
    ]
    if rhs_transform is None:
        operand = weight if rhs_value is None else f'T.cast({weight}, "{dtype}")'
        lines += [
            f'                rhs[bk, br] = T.if_then_else((bx * {tile_n} + br < {columns}) & (tile * {tile_k} + bk < {depth}), {operand}, T.cast(0, "{dtype}"))'
        ]
    else:
        lines += [
            f"                value = T.if_then_else(bx * {tile_n} + br < {columns}, T.if_then_else(tile * {tile_k} + bk < {depth}, {weight}, 0), 0)",
            f'                rhs[bk, br] = T.cast({rhs_transform.format(value="value")}, "{dtype}")',
        ]
    lines += [
        "        T.sync_threads()",
        f"        for chunk in T.serial({tile_k // (unroll * dot_width)}):",
        f"            for u in T.unroll({unroll}):",
        f"                kk = chunk * {unroll} + u"
        if dot_width == 1
        else f"                kk = (chunk * {unroll} + u) * {dot_width}",
    ]

    def vector(values):
        return (
            values[0]
            if dot_width == 1
            else f'T.call_extern("float32x{dot_width}", "vec{dot_width}<f32>", '
            + ", ".join(values)
            + ")"
        )

    for i in range(micro_m):
        values = [
            f'T.cast(lhs{idx(f"mr + {i * rm}", "kk" if dot_width == 1 else f"kk + {lane}")}, "float32")'
            for lane in range(dot_width)
        ]
        lines += [f"                left{i} = {vector(values)}"]
    for j in range(micro_n):
        values = [
            f'T.cast(rhs[{"kk" if dot_width == 1 else f"kk + {lane}"}, nr + {j * rn}], "float32")'
            for lane in range(dot_width)
        ]
        lines += [f"                right{j} = {vector(values)}"]
    for i in range(micro_m):
        for j in range(micro_n):
            value = (
                f'acc{i}_{j} + T.call_extern("float32", "dot", left{i}, right{j})'
                if dot_width > 1
                else f'T.call_extern("float32", "fma", left{i}, right{j}, acc{i}_{j})'
                if fma
                else f"acc{i}_{j} + left{i} * right{j}"
            )
            lines += [f"                acc{i}_{j} = {value}"]
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


def outer_product_matmul(
    rows,
    depth,
    columns,
    read_lhs,
    read_rhs,
    epilogue,
    *,
    parts=1,
    tile_m=64,
    tile_n=64,
    tile_k=16,
    micro_m=4,
    micro_n=4,
    threads=256,
    lhs_layout="km",
    lhs_pad=0,
    rhs_pad=0,
    owner_axis="column",
    unroll=4,
    fma=True,
    dtype="float32",
    explicit_unroll=False,
    dot_width=1,
    packed_pairs=False,
    half_accum=False,
    group_order="column",
):
    import tilelang.language as T

    if any(
        (
            type(v) is not int or v <= 0
            for v in (
                rows,
                depth,
                columns,
                tile_m,
                tile_n,
                tile_k,
                micro_m,
                micro_n,
                threads,
                unroll,
            )
        )
    ):
        raise ValueError("invalid outer-product dimensions")
    if (
        threads not in (64, 128, 256, 512)
        or tile_m % micro_m
        or tile_n % micro_n
        or (tile_m // micro_m * (tile_n // micro_n) != threads)
    ):
        raise ValueError("invalid outer-product ownership")
    if (
        micro_m * micro_n > 64
        or tile_k % unroll
        or unroll > 16
        or (dot_width not in (1, 2, 4))
        or tile_k % (unroll * dot_width)
    ):
        raise ValueError("invalid outer-product register footprint or unroll")
    if (
        lhs_layout not in ("mk", "km")
        or owner_axis not in ("row", "column")
        or dtype not in ("float32", "float16")
        or (group_order not in ("row", "column"))
    ):
        raise ValueError("invalid outer-product layout or arithmetic")
    if any((type(v) is not int or v < 0 for v in (lhs_pad, rhs_pad))):
        raise ValueError("invalid outer-product padding")
    if packed_pairs:
        if dtype != "float16" or dot_width != 2 or lhs_layout != "km":
            raise ValueError(
                "packed outer-product pairs require F16, K-major layout and dot width two"
            )
        return packed_outer_product(
            rows,
            depth,
            columns,
            read_lhs,
            read_rhs,
            epilogue,
            parts=parts,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            micro_m=micro_m,
            micro_n=micro_n,
            threads=threads,
            lhs_pad=lhs_pad,
            rhs_pad=rhs_pad,
            owner_axis=owner_axis,
            unroll=unroll,
            explicit_unroll=explicit_unroll,
            half_accum=half_accum,
            group_order=group_order,
        )
    if half_accum:
        raise ValueError("half accumulation requires packed F16 pairs")
    lhs_shape = (
        (tile_k, tile_m + lhs_pad) if lhs_layout == "km" else (tile_m, tile_k + lhs_pad)
    )
    rhs_shape = (tile_k, tile_n + rhs_pad)
    if (lhs_shape[0] * lhs_shape[1] + parts * rhs_shape[0] * rhs_shape[1]) * (
        4 if dtype == "float32" else 2
    ) > 32768:
        raise ValueError("outer-product shared storage exceeds 32 KiB")
    rm, rn = (tile_m // micro_m, tile_n // micro_n)

    def multiply(acc, lhs, rhs, mr, nr, kk):
        left = [
            vector(
                [
                    T.cast(
                        lhs[kk + lane, mr + i * rm]
                        if lhs_layout == "km"
                        else lhs[mr + i * rm, kk + lane],
                        "float32",
                    )
                    for lane in range(dot_width)
                ]
            )
            for i in range(micro_m)
        ]
        for part in range(parts):
            right = [
                vector(
                    [
                        T.cast(rhs[part][kk + lane, nr + j * rn], "float32")
                        for lane in range(dot_width)
                    ]
                )
                for j in range(micro_n)
            ]
            for i, j in product(range(micro_m), range(micro_n)):
                update(acc[part, i, j][0], left[i], right[j])

    def stage(weights, rhs, bx, tile, tx):
        for part in range(parts):
            stage_b(weights[part], rhs[part], bx, tile, tx)

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
        for load_a in T.serial(T.ceildiv(tile_m * tile_k, threads)):
            flat_a = load_a * threads + tx
            ar = flat_a // tile_k
            ak = flat_a % tile_k
            if ar < tile_m:
                if lhs_layout == "km":
                    if (by * tile_m + ar < rows) & (tile * tile_k + ak < depth):
                        lhs[ak, ar] = T.cast(
                            read_lhs(
                                x[(by * tile_m + ar) * depth + tile * tile_k + ak]
                            ),
                            dtype,
                        )
                    else:
                        lhs[ak, ar] = T.cast(0, dtype)
                elif (by * tile_m + ar < rows) & (tile * tile_k + ak < depth):
                    lhs[ar, ak] = T.cast(
                        read_lhs(x[(by * tile_m + ar) * depth + tile * tile_k + ak]),
                        dtype,
                    )
                else:
                    lhs[ar, ak] = T.cast(0, dtype)

    @T.macro
    def stage_b(w, rhs, bx, tile, tx):
        for load_b in T.serial(T.ceildiv(tile_n * tile_k, threads)):
            flat_b = load_b * threads + tx
            br = flat_b // tile_k
            bk = flat_b % tile_k
            if br < tile_n:
                if (bx * tile_n + br < columns) & (tile * tile_k + bk < depth):
                    rhs[bk, br] = T.cast(
                        read_rhs(w, bx * tile_n + br, tile * tile_k + bk), dtype
                    )
                else:
                    rhs[bk, br] = T.cast(0, dtype)

    @T.macro
    def update(acc: T.Ref, left, right):
        if dot_width > 1:
            acc = acc + T.call_extern("float32", "dot", left, right)
        elif fma:
            acc = T.call_extern("float32", "fma", left, right, acc)
        else:
            acc = acc + left * right

    @T.macro
    def store(out, residual, acc, up, by, bx, mr, nr, i, j):
        if (by * tile_m + mr + i * rm < rows) & (bx * tile_n + nr + j * rn < columns):
            epilogue(
                out,
                residual,
                (by * tile_m + mr + i * rm) * columns + bx * tile_n + nr + j * rn,
                acc,
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
            lhs = T.alloc_shared(lhs_shape, dtype)
            rhs = shared_buffers(rhs_shape, dtype, parts)
            accum = registers((parts, micro_m, micro_n))
            for tile in T.serial(T.ceildiv(depth, tile_k)):
                stage_a(x, lhs, by, tile, tx)
                stage((w, w2), rhs, bx, tile, tx)
                T.sync_threads()
                for chunk in T.serial(tile_k // (unroll * dot_width)):
                    for u in T.unroll(unroll):
                        kk = (chunk * unroll + u) * dot_width
                        multiply(accum, lhs, rhs, mr, nr, kk)
                T.sync_threads()
            stores(out, residual, accum, by, bx, mr, nr)

    return matmul
