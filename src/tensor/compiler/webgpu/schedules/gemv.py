"""Gemv implementation for WebGPU kernels."""

from __future__ import annotations

from itertools import product

from tensor.compiler.webgpu.schedules.registers import registers
from tensor.compiler.webgpu.schedules.registers import shared_buffers
from tensor.compiler.webgpu.schedules.registers import vector


def streamed_gemv_schedule(
    depth,
    columns,
    *,
    lanes=32,
    threads=128,
    micro_rows=1,
    dot_width=4,
    unroll=1,
    accumulators=1,
    k_layout="striped",
    shared_input=False,
):
    """F16-weight/F32-input GEMV with reusable input and independent K chains.

    Keep activation precision intact. Output tails are guarded and subgroup
    reductions have a shared-memory fallback for smaller physical subgroups.
    """
    if any(
        type(v) is not int or v <= 0
        for v in (
            depth,
            columns,
            lanes,
            threads,
            micro_rows,
            dot_width,
            unroll,
            accumulators,
        )
    ):
        raise ValueError("invalid GEMV dimensions")
    if (
        lanes not in (8, 16, 32, 64, 128)
        or threads not in (64, 128, 256, 512)
        or threads % lanes
    ):
        raise ValueError("invalid GEMV distribution")
    if (
        dot_width not in (1, 2, 4)
        or accumulators not in (1, 2, 4, 8)
        or unroll % accumulators
    ):
        raise ValueError("invalid GEMV arithmetic")
    if (
        depth % (lanes * dot_width * unroll)
        or micro_rows > 8
        or micro_rows * accumulators > 32
    ):
        raise ValueError("invalid GEMV tile")
    if k_layout not in ("blocked", "striped") or type(shared_input) is not bool:
        raise ValueError("invalid GEMV input layout")
    if (depth * 4 if shared_input else 0) + micro_rows * threads * 4 > 32768:
        raise ValueError("GEMV scratch exceeds portable limit")
    rows = threads // lanes
    tile_rows = rows * micro_rows
    step = lanes * dot_width * unroll
    lines = [
        f"with T.Kernel(T.ceildiv({columns}, {tile_rows}), threads={threads}) as bx:",
        "    tx = T.get_thread_binding()",
        f"    lane = tx % {lanes}",
        f"    row = tx // {lanes}",
        f'    scratch = T.alloc_shared(({micro_rows * threads},), "float32")',
    ]
    if shared_input:
        lines += [
            f'    lhs = T.alloc_shared(({depth},), "float32")',
            f"    for i in T.Parallel({depth}):",
            "        lhs[i] = x[i]",
            "    T.sync_threads()",
        ]
    for i in range(micro_rows):
        for j in range(accumulators):
            lines += [f'    acc{i}_{j} = T.alloc_var("float32")', f"    acc{i}_{j} = 0"]
    lines += [f"    for tile in T.serial({depth // step}):"]
    # Generate unroll slots explicitly to distribute independent accumulators.
    for u in range(unroll):
        kval = lambda v: (
            f"tile * {step} + lane * {unroll * dot_width} + {u * dot_width + v}"
            if k_layout == "blocked"
            else f"tile * {step} + {u * lanes * dot_width} + lane * {dot_width} + {v}"
        )
        lhs = [f"{'lhs' if shared_input else 'x'}[{kval(v)}]" for v in range(dot_width)]
        vector = lambda terms: (
            terms[0]
            if dot_width == 1
            else f'T.call_extern("float32x{dot_width}", "vec{dot_width}<f32>", '
            + ", ".join(terms)
            + ")"
        )
        lines += [f"        left{u} = {vector(lhs)}"]
        for i in range(micro_rows):
            output_row = f"bx * {tile_rows} + row + {i * rows}"
            rhs = [
                f'T.cast(w[({output_row}) * {depth} + {kval(v)}], "float32")'
                for v in range(dot_width)
            ]
            lines += [
                f"        if {output_row} < {columns}:",
                f"            right{u} = {vector(rhs)}",
            ]
            prod = (
                f"left{u} * right{u}"
                if dot_width == 1
                else f'T.call_extern("float32", "dot", left{u}, right{u})'
            )
            lines += [
                f"            acc{i}_{u % accumulators} = acc{i}_{u % accumulators} + {prod}"
            ]
    for i in range(micro_rows):
        for j in range(1, accumulators):
            lines += [f"    acc{i}_0 = acc{i}_0 + acc{i}_{j}"]
    lines += [f'    if T.call_extern("uint32", "tensor_subgroup_size") >= {lanes}:']
    for stride in (lanes >> i for i in range(1, lanes.bit_length())):
        for i in range(micro_rows):
            lines += [
                f'        acc{i}_0 = acc{i}_0 + T.call_extern("float32", "subgroupShuffleXor", acc{i}_0, T.uint32({stride}))'
            ]
    lines += ["    else:"]
    for i in range(micro_rows):
        lines += [f"        scratch[{i * threads} + tx] = acc{i}_0"]
    lines += ["        T.sync_threads()"]
    for stride in (lanes >> i for i in range(1, lanes.bit_length())):
        lines += [f"        if lane < {stride}:"]
        for i in range(micro_rows):
            lines += [
                f"            scratch[{i * threads} + tx] = scratch[{i * threads} + tx] + scratch[{i * threads + stride} + tx]"
            ]
        lines += ["        T.sync_threads()"]
    for i in range(micro_rows):
        lines += [f"        acc{i}_0 = scratch[{i * threads} + tx]"]
    for i in range(micro_rows):
        output_row = f"bx * {tile_rows} + row + {i * rows}"
        lines += [
            f"    if (lane == 0) & ({output_row} < {columns}):",
            f"        out[{output_row}] = acc{i}_0",
        ]
    return "\n".join(lines)


def streamed_gemv(
    depth,
    columns,
    epilogue,
    *,
    parts=1,
    lanes=32,
    threads=128,
    micro_rows=1,
    dot_width=4,
    unroll=1,
    accumulators=1,
    k_layout="striped",
    shared_input=False,
):
    import tilelang.language as T

    if any(
        (
            type(v) is not int or v <= 0
            for v in (
                depth,
                columns,
                lanes,
                threads,
                micro_rows,
                dot_width,
                unroll,
                accumulators,
            )
        )
    ):
        raise ValueError("invalid GEMV dimensions")
    if (
        lanes not in (8, 16, 32, 64, 128)
        or threads not in (64, 128, 256, 512)
        or threads % lanes
    ):
        raise ValueError("invalid GEMV distribution")
    if (
        dot_width not in (1, 2, 4)
        or accumulators not in (1, 2, 4, 8)
        or unroll % accumulators
    ):
        raise ValueError("invalid GEMV arithmetic")
    if (
        depth % (lanes * dot_width * unroll)
        or micro_rows > 8
        or micro_rows * accumulators > 32
    ):
        raise ValueError("invalid GEMV tile")
    if k_layout not in ("blocked", "striped") or type(shared_input) is not bool:
        raise ValueError("invalid GEMV input layout")
    if (depth * 4 if shared_input else 0) + parts * micro_rows * threads * 4 > 32768:
        raise ValueError("GEMV scratch exceeds portable limit")
    rows, tile_rows, step = (
        threads // lanes,
        threads // lanes * micro_rows,
        lanes * dot_width * unroll,
    )

    def multiply(acc, x, weights, tile, lane, bx, row):
        for u in range(unroll):
            index = tile * step + (
                lane * (unroll * dot_width) + u * dot_width
                if k_layout == "blocked"
                else u * lanes * dot_width + lane * dot_width
            )
            left = vector([x[index + v] for v in range(dot_width)])
            for part, i in product(range(parts), range(micro_rows)):
                update(
                    acc[part, i, u % accumulators][0],
                    weights[part],
                    left,
                    index,
                    bx * tile_rows + row + i * rows,
                )

    def combine(acc):
        for part, i, j in product(
            range(parts), range(micro_rows), range(1, accumulators)
        ):
            add(acc[part, i, 0][0], acc[part, i, j][0])

    def shuffles(acc, stride):
        for part, i in product(range(parts), range(micro_rows)):
            shuffle(acc[part, i, 0][0], stride)

    def spills(acc, scratch, tx):
        for part, i in product(range(parts), range(micro_rows)):
            spill(scratch[part], acc[part, i, 0][0], i * threads, tx)

    def shared_adds(scratch, tx, stride):
        for part, i in product(range(parts), range(micro_rows)):
            add_shared(scratch[part], i * threads, tx, stride)

    def reloads(acc, scratch, tx):
        for part, i in product(range(parts), range(micro_rows)):
            reload(acc[part, i, 0][0], scratch[part], i * threads + tx)

    def stores(out, residual, acc, bx, row, lane):
        for i in range(micro_rows):
            store(
                out,
                residual,
                acc[0, i, 0][0],
                acc[1, i, 0][0] if parts == 2 else 0,
                bx,
                row,
                lane,
                i,
            )

    def reduce_shuffles(acc):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            shuffles(acc, stride)

    def reduce_shared(scratch, tx, lane):
        for stride in (lanes >> i for i in range(1, lanes.bit_length())):
            reduce_step(scratch, tx, lane, stride)

    @T.macro
    def update(acc: T.Ref, w, left, index, output_row):
        if output_row < columns:
            if dot_width == 1:
                acc = acc + left * T.cast(w[output_row * depth + index], "float32")
            elif dot_width == 2:
                right = T.call_extern(
                    "float32x2",
                    "vec2<f32>",
                    T.cast(w[output_row * depth + index], "float32"),
                    T.cast(w[output_row * depth + index + 1], "float32"),
                )
                acc = acc + T.call_extern("float32", "dot", left, right)
            else:
                right = T.call_extern(
                    "float32x4",
                    "vec4<f32>",
                    T.cast(w[output_row * depth + index], "float32"),
                    T.cast(w[output_row * depth + index + 1], "float32"),
                    T.cast(w[output_row * depth + index + 2], "float32"),
                    T.cast(w[output_row * depth + index + 3], "float32"),
                )
                acc = acc + T.call_extern("float32", "dot", left, right)

    @T.macro
    def add(acc: T.Ref, other):
        acc = acc + other

    @T.macro
    def shuffle(acc: T.Ref, stride):
        acc = acc + T.call_extern(
            "float32", "subgroupShuffleXor", acc, T.uint32(stride)
        )

    @T.macro
    def spill(scratch, value, offset, tx):
        scratch[offset + tx] = value

    @T.macro
    def add_shared(scratch, offset, tx, stride):
        scratch[offset + tx] = scratch[offset + tx] + scratch[offset + tx + stride]

    @T.macro
    def reload(acc: T.Ref, scratch, index):
        acc = scratch[index]

    @T.macro
    def store(out, residual, value, up, bx, row, lane, i):
        if (lane == 0) & (bx * tile_rows + row + i * rows < columns):
            epilogue(out, residual, bx * tile_rows + row + i * rows, value, up)

    @T.macro
    def gemv(x, w, out, w2=None, residual=None):
        with T.Kernel(T.ceildiv(columns, tile_rows), threads=threads) as bx:
            tx = T.get_thread_binding()
            lane = tx % lanes
            row = tx // lanes
            scratch = shared_buffers((micro_rows * threads,), "float32", parts)
            if shared_input:
                lhs = T.alloc_shared((depth,), "float32")
                for i in T.Parallel(depth):
                    lhs[i] = x[i]
                T.sync_threads()
            accum = registers((parts, micro_rows, accumulators))
            for tile in T.serial(depth // step):
                multiply(
                    accum, lhs if shared_input else x, (w, w2), tile, lane, bx, row
                )
            combine(accum)
            if T.call_extern("uint32", "tensor_subgroup_size") >= lanes:
                reduce_shuffles(accum)
            else:
                spills(accum, scratch, tx)
                T.sync_threads()
                reduce_shared(scratch, tx, lane)
                reloads(accum, scratch, tx)
            stores(out, residual, accum, bx, row, lane)

    @T.macro
    def reduce_step(scratch, tx, lane, stride):
        if lane < stride:
            shared_adds(scratch, tx, stride)
        T.sync_threads()

    return gemv
