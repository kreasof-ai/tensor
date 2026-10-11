"""Greedy vocabulary projection with exact tile maxima instead of full logits."""


def make_kernel(kind, p):
    import tilelang.language as T

    r = p["r"]
    if kind == "head_linear":
        k, o, m, n, bk = (p[name] for name in ("k", "o", "m", "n", "bk"))
        tiles = T.ceildiv(o, n)

        @T.prim_func
        def kernel(
            x: T.Tensor((r, k), "bfloat16"),
            w: T.Tensor((o, k), "bfloat16"),
            maximum: T.Tensor((r, tiles), "float32"),
            index: T.Tensor((r, tiles), "int32"),
        ):
            with T.Kernel(T.ceildiv(r, m), tiles, threads=128) as (by, bx):
                a = T.alloc_shared((m, bk), "bfloat16")
                b = T.alloc_shared((n, bk), "bfloat16")
                acc = T.alloc_fragment((m, n), "float32")
                T.clear(acc)
                best = T.alloc_fragment((m,), "float32")
                ids = T.alloc_fragment((m, n), "int32")
                first = T.alloc_fragment((m,), "int32")
                for tile in T.Pipelined(T.ceildiv(k, bk), num_stages=2):
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
                    if bx * n + j >= o:
                        acc[i, j] = -T.infinity("float32")
                T.reduce_max(acc, best, dim=1)
                for i, j in T.Parallel(m, n):
                    ids[i, j] = T.if_then_else(
                        (acc[i, j] == best[i]) & (bx * n + j < o), bx * n + j, o
                    )
                T.reduce_min(ids, first, dim=1)
                for i in T.Parallel(m):
                    if by * m + i < r:
                        maximum[by * m + i, bx] = best[i]
                        index[by * m + i, bx] = first[i]

        return kernel
    if kind == "head_argmax":
        tiles, vocab = p["tiles"], p["vocab"]
        padded = 1 << (tiles - 1).bit_length()

        @T.prim_func
        def kernel(
            maximum: T.Tensor((r, tiles), "float32"),
            index: T.Tensor((r, tiles), "int32"),
            active: T.Tensor((r,), "int32"),
            predicted: T.Tensor((r,), "int32"),
        ):
            with T.Kernel(r, threads=256) as row:
                values = T.alloc_fragment((padded,), "float32")
                best = T.alloc_fragment((1,), "float32")
                ids = T.alloc_fragment((padded,), "int32")
                first = T.alloc_fragment((1,), "int32")
                for j in T.Parallel(padded):
                    values[j] = T.if_then_else(
                        j < tiles,
                        maximum[row, T.min(j, tiles - 1)],
                        -T.infinity("float32"),
                    )
                T.reduce_max(values, best, dim=0)
                for j in T.Parallel(padded):
                    ids[j] = T.if_then_else(
                        (j < tiles) & (values[j] == best[0]),
                        index[row, T.min(j, tiles - 1)],
                        vocab,
                    )
                T.reduce_min(ids, first, dim=0)
                for j in T.Parallel(1):
                    predicted[row] = T.if_then_else(active[row] != 0, first[0], -1)

        return kernel
    raise ValueError("unknown fused greedy head kernel")
