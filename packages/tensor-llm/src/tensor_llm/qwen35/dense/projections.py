"""BF16 projections with deterministic Split-K and a fixed reduction order."""


def schedule(rows, k, o, width):
    # Narrow output projections need extra CTAs at low concurrency. Keep the
    # same partitioning at every capacity to preserve request arithmetic.
    m = 64 if rows >= 128 else 32 if rows >= 32 and o > width else 16
    # Measured across five row counts; K partitioning remains unchanged.
    return dict(
        r=rows, k=k, o=o, m=m, n=64, bk=128, parts=8 if o == width and k > width else 1
    )


def make_kernel(kind, p):
    import tilelang.language as T

    r, k, o, parts = (p[n] for n in ("r", "k", "o", "parts"))
    if kind == "split_linear":
        m, n, bk = p["m"], p["n"], p["bk"]
        tiles = T.ceildiv(T.ceildiv(k, bk), parts)

        @T.prim_func
        def kernel(
            x: T.Tensor((r, k), "bfloat16"),
            w: T.Tensor((o, k), "bfloat16"),
            out: T.Tensor((r, parts, o), "float32"),
        ):
            # Adjacent row tiles reuse the same weight tile in L2. Sweeping
            # the entire vocabulary for each row tile reloads the head matrix.
            with T.Kernel(
                T.ceildiv(r, m), T.ceildiv(o, n), parts, threads=p.get("threads", 128)
            ) as (by, bx, part):
                a = T.alloc_shared((m, bk), "bfloat16")
                b = T.alloc_shared((n, bk), "bfloat16")
                acc = T.alloc_fragment((m, n), "float32")
                T.clear(acc)
                for local in T.Pipelined(tiles, num_stages=2):
                    tile = part * tiles + local
                    for i, j in T.Parallel(m, bk):
                        a[i, j] = T.if_then_else(
                            (by * m + i < r) & (tile * bk + j < k),
                            x[T.min(by * m + i, r - 1), T.min(tile * bk + j, k - 1)],
                            0,
                        )
                    for i, j in T.Parallel(n, bk):
                        b[i, j] = T.if_then_else(
                            (bx * n + i < o) & (tile * bk + j < k),
                            w[T.min(bx * n + i, o - 1), T.min(tile * bk + j, k - 1)],
                            0,
                        )
                    T.gemm(a, b, acc, transpose_B=True)
                for i, j in T.Parallel(m, n):
                    if (by * m + i < r) & (bx * n + j < o):
                        out[by * m + i, part, bx * n + j] = acc[i, j]

        return kernel
    if kind == "split_merge":

        @T.prim_func
        def kernel(
            x: T.Tensor((r, parts, o), "float32"), out: T.Tensor((r, o), "float32")
        ):
            with T.Kernel(r, T.ceildiv(o, 256), threads=256) as (row, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < o:
                        value = T.alloc_var("float32")
                        value = 0
                        for part in T.serial(parts):
                            value += x[row, part, col]
                        out[row, col] = value

        return kernel
    raise ValueError("unsupported projection: " + kind)
