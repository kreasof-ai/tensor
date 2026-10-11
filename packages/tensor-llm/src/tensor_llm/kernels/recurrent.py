"""Exact delayed rank-one updates for speculative recurrent state.

The base matrix includes the first candidate, which greedy verification always
accepts. Remaining accepted keys, gates, and per-value corrections are held in
a small ping-pong journal.
The next verification applies that journal before computing new candidates.
"""


def make_kernel(kind, p):
    import tilelang.language as T

    slots, window, pool = (p[n] for n in ("slots", "window", "pool"))
    heads, d, tile = p.get("heads", 16), p.get("d", 128), p.get("tile", 32)
    if kind == "conv_commit":
        channels, projection = p["channels"], p["projection"]

        @T.prim_func
        def kernel(
            x: T.Tensor((slots * window, projection), "float32"),
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            state: T.Tensor((pool, channels, 3), "float32"),
        ):
            with T.Kernel(slots, T.ceildiv(channels, 256), threads=256) as (
                batch,
                tile,
            ):
                request = mapping[batch]
                if request >= 0:
                    for j in T.Parallel(256):
                        col = tile * 256 + j
                        if col < channels:
                            h0 = T.alloc_var("float32")
                            h1 = T.alloc_var("float32")
                            h2 = T.alloc_var("float32")
                            h0 = state[request, col, 0]
                            h1 = state[request, col, 1]
                            h2 = state[request, col, 2]
                            for t in T.serial(window):
                                if t < lengths[batch]:
                                    h0 = h1
                                    h1 = h2
                                    h2 = T.cast(
                                        T.cast(x[batch * window + t, col], "bfloat16"),
                                        "float32",
                                    )
                            state[request, col, 0] = h0
                            state[request, col, 1] = h1
                            state[request, col, 2] = h2

        return kernel
    if kind == "defer_accept":

        @T.prim_func
        def kernel(
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            counts: T.Tensor((pool,), "int32"),
            banks: T.Tensor((pool,), "int32"),
        ):
            with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
                for j in T.Parallel(128):
                    row = block * 128 + j
                    if row < slots:
                        request = mapping[row]
                        if request >= 0:
                            counts[request] = T.max(lengths[row] - 1, 0)
                            banks[request] = 1 - banks[request]

        return kernel
    if kind == "defer_clear":

        @T.prim_func
        def kernel(
            mapping: T.Tensor((slots,), "int32"), counts: T.Tensor((pool,), "int32")
        ):
            with T.Kernel(T.ceildiv(slots, 128), threads=128) as block:
                for j in T.Parallel(128):
                    row = block * 128 + j
                    if row < slots:
                        request = mapping[row]
                        if request >= 0:
                            counts[request] = 0

        return kernel
    if kind == "gdn_materialize":

        @T.prim_func
        def kernel(
            mapping: T.Tensor((slots,), "int32"),
            state: T.Tensor((pool, heads, d, d), "float32"),
            saved_k: T.Tensor((2, pool, window, heads, d), "float32"),
            saved_v: T.Tensor((2, pool, window, heads, d), "float32"),
            saved_g: T.Tensor((2, pool, window, heads), "float32"),
            saved_b: T.Tensor((2, pool, window, heads), "float32"),
            counts: T.Tensor((pool,), "int32"),
            banks: T.Tensor((pool,), "int32"),
        ):
            with T.Kernel(slots, heads, d // tile, threads=128) as (batch, head, part):
                request = mapping[batch]
                if request >= 0:
                    if counts[request] > 0:
                        bank = banks[request]
                        matrix = T.alloc_fragment((tile, d), "float32")
                        kk = T.alloc_shared((d,), "float32")
                        T.copy(
                            state[request, head, part * tile : part * tile + tile, :],
                            matrix,
                        )
                        for t in T.serial(window):
                            if t < counts[request]:
                                for j in T.Parallel(d):
                                    kk[j] = saved_k[bank, request, t, head, j]
                                for i, j in T.Parallel(tile, d):
                                    matrix[i, j] *= T.exp(
                                        saved_g[bank, request, t, head]
                                    )
                                    matrix[i, j] += (
                                        kk[j]
                                        * saved_b[bank, request, t, head]
                                        * saved_v[
                                            bank, request, t, head, part * tile + i
                                        ]
                                    )
                        T.copy(
                            matrix,
                            state[request, head, part * tile : part * tile + tile, :],
                        )

        return kernel
    if kind == "gdn_deferred":
        rows = slots * window

        @T.prim_func
        def kernel(
            q: T.Tensor((rows, heads, d), "float32"),
            k: T.Tensor((rows, heads, d), "float32"),
            v: T.Tensor((rows, heads, d), "float32"),
            g: T.Tensor((rows, heads), "float32"),
            beta: T.Tensor((rows, heads), "float32"),
            mapping: T.Tensor((slots,), "int32"),
            lengths: T.Tensor((slots,), "int32"),
            state: T.Tensor((pool, heads, d, d), "float32"),
            out: T.Tensor((rows, heads, d), "float32"),
            saved_k: T.Tensor((2, pool, window, heads, d), "float32"),
            saved_v: T.Tensor((2, pool, window, heads, d), "float32"),
            saved_g: T.Tensor((2, pool, window, heads), "float32"),
            saved_b: T.Tensor((2, pool, window, heads), "float32"),
            counts: T.Tensor((pool,), "int32"),
            banks: T.Tensor((pool,), "int32"),
        ):
            with T.Kernel(slots, heads, d // tile, threads=128) as (batch, head, part):
                request = mapping[batch]
                if request >= 0:
                    bank = banks[request]
                    matrix = T.alloc_fragment((tile, d), "float32")
                    prediction = T.alloc_fragment((tile, d), "float32")
                    inner = T.alloc_fragment((tile,), "float32")
                    correction = T.alloc_fragment((tile,), "float32")
                    result = T.alloc_fragment((tile,), "float32")
                    qq = T.alloc_shared((d,), "float32")
                    kk = T.alloc_shared((d,), "float32")
                    T.copy(
                        state[request, head, part * tile : part * tile + tile, :],
                        matrix,
                    )
                    for t in T.serial(window):
                        if t < counts[request]:
                            for j in T.Parallel(d):
                                kk[j] = saved_k[bank, request, t, head, j]
                            for i, j in T.Parallel(tile, d):
                                matrix[i, j] *= T.exp(saved_g[bank, request, t, head])
                                matrix[i, j] += (
                                    kk[j]
                                    * saved_b[bank, request, t, head]
                                    * saved_v[bank, request, t, head, part * tile + i]
                                )
                    for t in T.serial(window):
                        row = batch * window + t
                        if t < lengths[batch]:
                            for j in T.Parallel(d):
                                qq[j] = q[row, head, j]
                                kk[j] = k[row, head, j]
                            for i, j in T.Parallel(tile, d):
                                matrix[i, j] *= T.exp(g[row, head])
                                prediction[i, j] = matrix[i, j] * kk[j]
                            T.reduce_sum(prediction, inner, dim=1)
                            for i in T.Parallel(tile):
                                correction[i] = v[row, head, part * tile + i] - inner[i]
                                if t > 0:
                                    saved_v[
                                        1 - bank, request, t - 1, head, part * tile + i
                                    ] = correction[i]
                            for i, j in T.Parallel(tile, d):
                                matrix[i, j] += kk[j] * beta[row, head] * correction[i]
                                prediction[i, j] = matrix[i, j] * qq[j]
                            T.reduce_sum(prediction, result, dim=1)
                            for i in T.Parallel(tile):
                                out[row, head, part * tile + i] = result[i]
                            if t == 0:
                                T.copy(
                                    matrix,
                                    state[
                                        request,
                                        head,
                                        part * tile : part * tile + tile,
                                        :,
                                    ],
                                )
                            if (part == 0) & (t > 0):
                                for j in T.Parallel(d):
                                    saved_k[1 - bank, request, t - 1, head, j] = kk[j]
                                for j in T.Parallel(1):
                                    saved_g[1 - bank, request, t - 1, head] = g[
                                        row, head
                                    ]
                                    saved_b[1 - bank, request, t - 1, head] = beta[
                                        row, head
                                    ]
                        else:
                            for i in T.Parallel(tile):
                                out[row, head, part * tile + i] = 0

        return kernel
    raise ValueError("unknown recurrent journal kernel")
