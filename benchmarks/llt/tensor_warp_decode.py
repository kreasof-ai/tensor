"""Single-query warp reductions without padded query rows or fragment inference."""


def decode_partial(p):
    import tilelang.language as T

    batch, heads, kv_heads, capacity, dim, dv, dtype, scale, partitions = p
    bn = 64

    @T.prim_func
    def kernel(
        q: T.Tensor((batch, heads, 1, dim), dtype),
        k: T.Tensor((batch, kv_heads, capacity, dim), dtype),
        v: T.Tensor((batch, kv_heads, capacity, dv), dtype),
        lengths: T.Tensor((batch,), "int32"),
        partial: T.Tensor((batch, heads, partitions, dv), "float32"),
        stats: T.Tensor((batch, heads, partitions, 2), "float32"),
    ):
        with T.Kernel(partitions, heads, batch, threads=128) as (part, head, b):
            tx = T.get_thread_binding()
            lane = tx % 32
            warp = tx // 32
            probability = T.alloc_shared((bn,), "float32")
            meta = T.alloc_shared((3,), "float32")
            dot = T.alloc_local((1,), "float32")
            mx = T.alloc_local((1,), "float32")
            norm = T.alloc_local((1,), "float32")
            result = T.alloc_local((1,), "float32")
            a = T.alloc_local((1,), "float32")
            z = T.alloc_local((1,), "float32")
            mx[0] = -T.infinity("float32")
            norm[0] = 0
            result[0] = 0
            tiles = T.ceildiv(lengths[b], bn)
            per = T.ceildiv(tiles, partitions)
            kh = head // (heads // kv_heads)
            for local in T.serial(T.max(0, T.min(per, tiles - part * per))):
                tile = part * per + local
                for row in T.serial(16):
                    key = tile * bn + row * 4 + warp
                    dot[0] = 0
                    for step in T.unroll(T.ceildiv(dim, 32)):
                        j = step * 32 + lane
                        if (key < lengths[b]) & (j < dim):
                            dot[0] += T.cast(q[b, head, 0, j], "float32") * T.cast(
                                k[b, kh, key, j], "float32"
                            )
                    dot[0] = T.warp_reduce_sum(dot[0])
                    if lane == 0:
                        probability[row * 4 + warp] = T.if_then_else(
                            key < lengths[b], dot[0] * scale, -T.infinity("float32")
                        )
                T.sync_threads()
                if warp == 0:
                    a[0] = T.max(probability[lane], probability[lane + 32])
                    a[0] = T.max(mx[0], T.warp_reduce_max(a[0]))
                    z[0] = T.exp(mx[0] - a[0])
                    mx[0] = a[0]
                    probability[lane] = T.exp(probability[lane] - mx[0])
                    probability[lane + 32] = T.exp(probability[lane + 32] - mx[0])
                    a[0] = T.warp_reduce_sum(probability[lane] + probability[lane + 32])
                    norm[0] = norm[0] * z[0] + a[0]
                    if lane == 0:
                        meta[0] = mx[0]
                        meta[1] = norm[0]
                        meta[2] = z[0]
                T.sync_threads()
                if tx < dv:
                    result[0] *= meta[2]
                    for row in T.unroll(bn):
                        if tile * bn + row < lengths[b]:
                            result[0] += probability[row] * T.cast(
                                v[b, kh, tile * bn + row, tx], "float32"
                            )
                T.sync_threads()
            if tx < dv:
                partial[b, head, part, tx] = result[0]
            if tx == 0:
                stats[b, head, part, 0] = mx[0]
                stats[b, head, part, 1] = norm[0]

    return kernel
