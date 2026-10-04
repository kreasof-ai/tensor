"""Inspectable TileLang DSL macros for portable SIMT projection schedules.

Python expands register microtiles at specialization time. Each accumulator is
an independent scalar, preserving the measured shader layouts and K chains.
Operand decoders and epilogues are DSL macros supplied by the producer.
"""

from itertools import product


def registers(shape, dtype="float32", *, initialize=True):
    """Allocate separate scalar registers, rather than a shader private array."""
    import tilelang.language as T

    return {
        index: T.alloc_var(dtype, init=0) if initialize else T.alloc_var(dtype)
        for index in product(*(range(size) for size in shape))
    }


def shared_buffers(shape, dtype, count):
    import tilelang.language as T

    return [T.alloc_shared(shape, dtype) for _ in range(count)]


def vector(values):
    import tilelang.language as T

    if len(values) == 1:
        return values[0]
    return T.call_extern(f"float32x{len(values)}", f"vec{len(values)}<f32>", *values)


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
    if not nr or tile_n % 2 or threads % nr or tile_m % (threads // nr) or depth % tile_k:
        raise ValueError("invalid register matmul tile")
    rm, mm = (threads // nr, tile_m // (threads // nr))
    lhs_shape = (tile_k, tile_m + lhs_pad) if lhs_transpose else (tile_m, tile_k + lhs_pad)
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
                by * tile_m + i < rows, x[(by * tile_m + i) * depth + tile * tile_k + j], 0
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
        with T.Kernel(T.ceildiv(rows, tile_m), T.ceildiv(columns, tile_n), threads=threads) as (
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
    lhs_shape = (tile_k, tile_m + lhs_pad) if lhs_layout == "km" else (tile_m, tile_k + lhs_pad)
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
                            read_lhs(x[(by * tile_m + ar) * depth + tile * tile_k + ak]), dtype
                        )
                    else:
                        lhs[ak, ar] = T.cast(0, dtype)
                elif (by * tile_m + ar < rows) & (tile * tile_k + ak < depth):
                    lhs[ar, ak] = T.cast(
                        read_lhs(x[(by * tile_m + ar) * depth + tile * tile_k + ak]), dtype
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
                    rhs[bk, br] = T.cast(read_rhs(w, bx * tile_n + br, tile * tile_k + bk), dtype)
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
    nc = min(tile_n, owners) if owner_axis == "column" else owners // min(tile_m, owners)
    if not nc or owners % nc or tile_m % (owners // nc) or tile_n % nc:
        raise ValueError("invalid partitioned output tile")
    rm, mm, nn = (owners // nc, tile_m // (owners // nc), tile_n // nc)
    if mm * nn > 64 or parts * tile_m * tile_n * partitions * 4 > 32768:
        raise ValueError("partitioned matmul resource limit")

    def depth_index(tile, lane, u, v):
        return (
            tile * (partitions * unroll) + lane * unroll + u * dot_width + v
            if k_layout == "blocked"
            else tile * (partitions * unroll) + lane * dot_width + u * (partitions * dot_width) + v
        )

    def multiply(acc, x, weights, by, bx, mr, nr, tile, lane, u):
        left = [
            vector(
                [
                    lhs_value(x, by * tile_m + mr + i * rm, depth_index(tile, lane, u, v))
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
                            weights[part], bx * tile_n + nr + j * nc, depth_index(tile, lane, u, v)
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
        with T.Kernel(T.ceildiv(rows, tile_m), T.ceildiv(columns, tile_n), threads=threads) as (
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
                            (by * tile_m + slot // tile_n) * columns + bx * tile_n + slot % tile_n,
                            totals[0,][0],
                            totals[1,][0] if parts == 2 else 0,
                        )

    return matmul


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
            for v in (depth, columns, lanes, threads, micro_rows, dot_width, unroll, accumulators)
        )
    ):
        raise ValueError("invalid GEMV dimensions")
    if lanes not in (8, 16, 32, 64, 128) or threads not in (64, 128, 256, 512) or threads % lanes:
        raise ValueError("invalid GEMV distribution")
    if dot_width not in (1, 2, 4) or accumulators not in (1, 2, 4, 8) or unroll % accumulators:
        raise ValueError("invalid GEMV arithmetic")
    if depth % (lanes * dot_width * unroll) or micro_rows > 8 or micro_rows * accumulators > 32:
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
        for part, i, j in product(range(parts), range(micro_rows), range(1, accumulators)):
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
        acc = acc + T.call_extern("float32", "subgroupShuffleXor", acc, T.uint32(stride))

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
                multiply(accum, lhs if shared_input else x, (w, w2), tile, lane, bx, row)
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
                        read_lhs(x[(by * tile_m + ar) * depth + tile * tile_k + pair * 2]),
                        "float32",
                    )
                else:
                    value0 = 0
                if (by * tile_m + ar < rows) & (tile * tile_k + pair * 2 + 1 < depth):
                    value1 = T.cast(
                        read_lhs(x[(by * tile_m + ar) * depth + tile * tile_k + pair * 2 + 1]),
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
                            read_rhs(w, bx * tile_n + br, tile * tile_k + pair * 2), "float32"
                        )
                    else:
                        value0 = 0
                else:
                    value0 = 0
                if bx * tile_n + br < columns:
                    if tile * tile_k + pair * 2 + 1 < depth:
                        value1 = T.cast(
                            read_rhs(w, bx * tile_n + br, tile * tile_k + pair * 2 + 1), "float32"
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
                partials = registers((parts, micro_m, micro_n), "float16x2", initialize=False)
            for tile in T.serial(T.ceildiv(depth, tile_k)):
                stage_a(x, lhs, by, tile, tx)
                stage((w, w2), rhs, bx, tile, tx)
                T.sync_threads()
                for chunk in T.serial(pk // unroll):
                    if half_accum:
                        clear_partials(partials)
                    for u in T.unroll(unroll):
                        kk = chunk * unroll + u
                        multiply(partials if half_accum else accum, lhs, rhs, mr, nr, kk)
                    if half_accum:
                        add_partials(accum, partials)
                T.sync_threads()
            stores(out, residual, accum, by, bx, mr, nr)

    return matmul


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
    if any((type(v) is not int or v <= 0 for v in dims)) or tile_k % 32 or depth % tile_k:
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

    def stage_components(packed, scales, sums, lhs, factors, offsets, ar, word, by, tile):
        for component in range(2):
            stage_component(
                packed, scales, sums, lhs, factors, offsets, component, ar, word, by, tile
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
            store(out, acc[0, i, j][0], acc[1, i, j][0] if parts == 2 else 0, by, bx, mr, nr, i, j)

    @T.macro
    def stage_component(packed, scales, sums, lhs, factors, offsets, component, ar, word, by, tile):
        row = by * tile_m + ar
        lhs[component, ar, word] = T.if_then_else(
            row < rows,
            packed[row * (depth // 4) + tile * (8 * blocks) + word + component * rows * depth // 4],
            T.uint32(0),
        )
        if word % 8 == 0:
            factors[component, ar, word // 8] = T.if_then_else(
                row < rows,
                scales[
                    row * (depth // 32) + tile * blocks + word // 8 + component * rows * depth // 32
                ],
                0,
            )
            offsets[component, ar, word // 8] = T.if_then_else(
                row < rows,
                sums[
                    row * (depth // 32) + tile * blocks + word // 8 + component * rows * depth // 32
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
            rhs[part, word // 4 * 8 + word % 4 + 4, br] = bits >> 4 & T.uint32(252645135)
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
    def accumulate(acc: T.Ref, dot0, dot1, factors, offsets, weight_scale, part, block, row, col):
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
                    T.cast(dot0 - 8 * offsets[0, row, block], "float32") * factors[0, row, block]
                    + T.cast(dot1 - 8 * offsets[1, row, block], "float32") * factors[1, row, block]
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
                            packed, scales, sums, lhs, factors, offsets, ar, word, by, tile
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
                    accumulate_dots(accum, dots, factors, offsets, weight_scale, block, mr, nr)
                T.sync_threads()
            stores(out, accum, by, bx, mr, nr)

    return matmul
