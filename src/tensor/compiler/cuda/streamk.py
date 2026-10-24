"""Work-centric GEMM with bounded scratch and deterministic separate reduction.

Workers own contiguous ranges of K-tile iterations across output tiles. Complete
tiles go directly to output; only boundary tiles need FP32 partials. Each worker
has at most two boundary fragments, so graph replays never need scratch clearing.
"""


def streamk_segments(tiles, k_tiles, workers):
    if (
        any(type(v) is not int or v < 1 for v in (tiles, k_tiles, workers))
        or workers > tiles * k_tiles
    ):
        raise ValueError("positive nonempty work ranges required")
    result = []
    total = tiles * k_tiles
    for worker in range(workers):
        begin, end = worker * total // workers, (worker + 1) * total // workers
        position = begin
        while position < end:
            tile = position // k_tiles
            stop = min(end, (tile + 1) * k_tiles)
            full = position % k_tiles == 0 and stop - position == k_tiles
            slot = 0 if tile == begin // k_tiles else 1
            result.append(
                (
                    worker,
                    tile,
                    position % k_tiles,
                    stop - position,
                    -1 if full else slot,
                )
            )
            position = stop
    return result


def _streamk_geometry(
    m, n, k, workers, block_m, block_n, block_k, stages=2, threads=128
):
    if any(
        type(v) is not int or v < 1
        for v in (m, n, k, workers, block_m, block_n, block_k, stages, threads)
    ):
        raise ValueError("positive static dimensions required")
    if (
        k % block_k
        or block_k % 16
        or block_m % 8
        or block_n % 16
        or threads not in (128, 256)
    ):
        raise ValueError("aligned Tensor Core tiles required")
    tiles_n = (n + block_n - 1) // block_n
    tiles = ((m + block_m - 1) // block_m) * tiles_n
    k_tiles = k // block_k
    if workers > tiles * k_tiles:
        raise ValueError("worker count exceeds nonempty work ranges")
    return tiles_n, tiles, k_tiles, tiles * k_tiles


def streamk_matmul(
    m,
    n,
    k,
    workers,
    *,
    block_m=32,
    block_n=64,
    block_k=128,
    stages=2,
    threads=128,
    out_dtype="bfloat16",
):
    import tilelang.language as T

    tiles_n, _, k_tiles, total = _streamk_geometry(
        m, n, k, workers, block_m, block_n, block_k, stages, threads
    )
    if out_dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported output dtype")

    @T.prim_func
    def product(
        x: T.Tensor((m, k), "bfloat16"),
        weights: T.Tensor((n, k), "bfloat16"),
        parts: T.Tensor((workers, 2, block_m, block_n), "float32"),
        out: T.Tensor((m, n), out_dtype),
    ):
        with T.Kernel(workers, threads=threads) as worker:
            a = T.alloc_shared((block_n, block_k), "bfloat16")
            b = T.alloc_shared((block_m, block_k), "bfloat16")
            acc = T.alloc_fragment((block_n, block_m), "float32")
            position = T.alloc_var("int32")
            position = worker * total // workers
            stop = (worker + 1) * total // workers
            while position < stop:
                tile = position // k_tiles
                bx, by = tile % tiles_n, tile // tiles_n
                first_k = position % k_tiles
                count = T.min(stop, (tile + 1) * k_tiles) - position
                T.clear(acc)
                for step in T.Pipelined(count, num_stages=stages):
                    T.copy(weights[bx * block_n, (first_k + step) * block_k], a)
                    T.copy(x[by * block_m, (first_k + step) * block_k], b)
                    T.gemm(a, b, acc, transpose_B=True)
                if first_k == 0 and count == k_tiles:
                    for i, j in T.Parallel(block_m, block_n):
                        if by * block_m + i < m and bx * block_n + j < n:
                            out[by * block_m + i, bx * block_n + j] = acc[j, i]
                else:
                    slot = T.if_then_else(
                        tile == worker * total // workers // k_tiles, 0, 1
                    )
                    for i, j in T.Parallel(block_m, block_n):
                        parts[worker, slot, i, j] = acc[j, i]
                position += count

    return product


def streamk_reduce(
    m, n, k, workers, *, block_m=32, block_n=64, block_k=128, out_dtype="bfloat16"
):
    import tilelang.language as T

    tiles_n, tiles, k_tiles, total = _streamk_geometry(
        m, n, k, workers, block_m, block_n, block_k
    )
    if out_dtype not in ("bfloat16", "float32"):
        raise ValueError("unsupported output dtype")

    @T.prim_func
    def reduce(
        parts: T.Tensor((workers, 2, block_m, block_n), "float32"),
        out: T.Tensor((m, n), out_dtype),
    ):
        with T.Kernel(tiles, threads=128) as tile:
            first_worker = ((tile * k_tiles + 1) * workers - 1) // total
            last_worker = ((tile + 1) * k_tiles * workers - 1) // total
            if first_worker != last_worker:
                acc = T.alloc_fragment((block_m, block_n), "float32")
                T.clear(acc)
                # Resolve the contributor once, then read its contiguous tile.
                # Keeping the worker loop outside Parallel prevents duplicating
                # dynamic division/slot lookup for every output element.
                # Preserve the separate floor-divided endpoint. Converting
                # it to a start+extent loop can incorrectly widen the interval
                # by one contributor when workers outnumber output tiles.
                worker = T.alloc_var("int32")
                worker = first_worker
                while worker <= last_worker:
                    slot = T.if_then_else(
                        tile == worker * total // workers // k_tiles, 0, 1
                    )
                    for i, j in T.Parallel(block_m, block_n):
                        acc[i, j] += parts[worker, slot, i, j]
                    worker += 1
                for i, j in T.Parallel(block_m, block_n):
                    if (
                        tile // tiles_n * block_m + i < m
                        and tile % tiles_n * block_n + j < n
                    ):
                        out[
                            tile // tiles_n * block_m + i, tile % tiles_n * block_n + j
                        ] = acc[i, j]

    return reduce
