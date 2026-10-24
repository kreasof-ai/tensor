"""Explicit Hopper TMA/WGMMA schedules for the native TileLang adapter.

The ordinary artifact launcher still has a pointer/scalar ABI. These kernels
require CUtensorMap arguments and therefore use TileLang's native launch adapter.
"""


def staged_matmul(
    m,
    n,
    k,
    *,
    block_m=32,
    block_n=32,
    block_k=256,
    stages=2,
    queue_depth=1,
    splits=1,
    paired=False,
    transposed_weights=True,
    out_dtype="float32",
    native_async=True,
    cluster_reduce=False,
    register_epilogue=False,
):
    """Generate a weight-major matmul with explicit bounded async ownership.

    A paired weight matrix contains interleaved BF16 hi/lo rows. Those rows
    retain separate FP32 accumulators and are combined only in the epilogue.
    K partitions are explicit FP32 outputs, consumed by a separate reducer.
    The synchronous mirror is useful for non-Hopper numerical validation.
    """
    import tilelang.language as T

    values = (m, n, k, block_m, block_n, block_k, stages, splits)
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("positive static dimensions and schedule parameters required")
    if type(queue_depth) is not int or not 0 <= queue_depth <= 7:
        raise ValueError("WGMMA queue depth must be between zero and seven")
    if stages < queue_depth + 1:
        raise ValueError("shared-memory ring must exceed the retained WGMMA queue")
    if k % (block_k * splits):
        raise ValueError("K partitions must be aligned to whole K tiles")
    if out_dtype not in ("bfloat16", "float32") or (
        splits > 1 and not cluster_reduce and out_dtype != "float32"
    ):
        raise ValueError("K partitions require FP32 outputs")
    weight_rows = (2 if paired else 1) * n
    tile_rows = (2 if paired else 1) * block_n
    if native_async and (tile_rows % 64 or block_m % 8):
        raise ValueError("Hopper WGMMA needs whole 64-row by 8-column tiles")
    if native_async and transposed_weights and weight_rows % 8:
        raise ValueError("TensorMap row strides must be aligned to 16 bytes")
    if cluster_reduce and (not native_async or splits not in (2, 4, 8)):
        raise ValueError(
            "cluster reduction needs native Hopper and 2, 4 or 8 K partitions"
        )
    if register_epilogue and (cluster_reduce or tile_rows != 64 or block_m % 8):
        raise ValueError(
            "register epilogue needs 64 weight rows, whole 8-column tiles and no cluster reduction"
        )
    shared = stages * (tile_rows + block_m) * block_k * 2 + tile_rows * block_m * 4
    if cluster_reduce:
        shared += (splits + 1) * block_m * block_n * 4
    if shared > 227328:
        raise ValueError("schedule exceeds Hopper block shared memory")
    tiles = k // (block_k * splits)
    weight_shape = (k, weight_rows) if transposed_weights else (weight_rows, k)
    storage_stages = stages if native_async else 1
    a_shape = (
        (storage_stages, block_k, tile_rows)
        if transposed_weights
        else (storage_stages, tile_rows, block_k)
    )
    output_parts = 1 if cluster_reduce else splits
    launch = T.ClusterKernel if cluster_reduce else T.Kernel
    launch_options = {"threads": 128}
    if cluster_reduce:
        launch_options["cluster_dims"] = (1, 1, splits)

    pair_source = r"""
__device__ __forceinline__ float tensor_pair_sum_xor4(float value) {
    float peer;
    asm volatile("shfl.sync.bfly.b32 %0, %1, 4, 31, 0xffffffff;"
                 : "=f"(peer) : "f"(value));
    return value + peer;
}
"""

    @T.prim_func
    def product(
        x: T.Tensor((m, k), "bfloat16"),
        weights: T.Tensor(weight_shape, "bfloat16"),
        out: T.Tensor((output_parts, m, n), out_dtype),
    ):
        with launch(
            T.ceildiv(n, block_n),
            T.ceildiv(m, block_m),
            splits,
            **launch_options,
        ) as (
            bx,
            by,
            part,
        ):
            a = T.alloc_shared(a_shape, "bfloat16")
            b = T.alloc_shared((storage_stages, block_m, block_k), "bfloat16")
            if native_async:
                ready = T.alloc_barrier([128] * stages)
            acc = T.alloc_fragment((tile_rows, block_m), "float32")
            if register_epilogue:
                # Explicit WGMMA/MMA C layout: adjacent weight rows live in
                # lanes separated by XOR 4, with identical local registers.
                T.annotate_layout(
                    {
                        acc: T.Fragment(
                            (tile_rows, block_m),
                            forward_thread_fn=lambda row, col: (
                                (row % 64 // 16) * 32 + (row % 8) * 4 + col % 8 // 2
                            ),
                            forward_index_fn=lambda row, col: (
                                row // 64 * (block_m // 2)
                                + col // 8 * 4
                                + row % 16 // 8 * 2
                                + col % 2
                            ),
                        )
                    }
                )
                if paired:
                    T.import_source(pair_source)
            else:
                epilogue = T.alloc_shared((tile_rows, block_m), "float32")
            if cluster_reduce:
                contribution = T.alloc_shared((1, block_m, block_n), "float32")
                gathered = T.alloc_shared((splits, block_m, block_n), "float32")
                completed = T.alloc_cluster_barrier(splits)
                # TileLang initializes cluster barriers and inserts the entry
                # cluster sync before any remote address can be used.
            x_view = T.reshape(x, (1, m, k))
            weight_view = T.reshape(weights, (1, *weight_shape))
            T.clear(acc)
            if native_async:
                for warm in T.serial(min(stages, tiles)):
                    if transposed_weights:
                        T.tma_copy(
                            weight_view[
                                0, (part * tiles + warm) * block_k, bx * tile_rows
                            ],
                            a[warm, :, :],
                            barrier=ready[warm],
                        )
                    else:
                        T.tma_copy(
                            weight_view[
                                0, bx * tile_rows, (part * tiles + warm) * block_k
                            ],
                            a[warm, :, :],
                            barrier=ready[warm],
                        )
                    T.tma_copy(
                        x_view[0, by * block_m, (part * tiles + warm) * block_k],
                        b[warm, :, :],
                        barrier=ready[warm],
                    )
                    T.mbarrier_arrive(ready[warm])
                for tile in T.serial(tiles):
                    T.mbarrier_wait_parity(ready[tile % stages], (tile // stages) % 2)
                    T.wgmma_gemm(
                        a[tile % stages, :, :],
                        b[tile % stages, :, :],
                        acc,
                        transpose_A=transposed_weights,
                        transpose_B=True,
                    )
                    if tile >= queue_depth:
                        T.wait_wgmma(queue_depth)
                        if tile - queue_depth + stages < tiles:
                            if transposed_weights:
                                T.tma_copy(
                                    weight_view[
                                        0,
                                        (part * tiles + tile - queue_depth + stages)
                                        * block_k,
                                        bx * tile_rows,
                                    ],
                                    a[(tile - queue_depth) % stages, :, :],
                                    barrier=ready[(tile - queue_depth) % stages],
                                )
                            else:
                                T.tma_copy(
                                    weight_view[
                                        0,
                                        bx * tile_rows,
                                        (part * tiles + tile - queue_depth + stages)
                                        * block_k,
                                    ],
                                    a[(tile - queue_depth) % stages, :, :],
                                    barrier=ready[(tile - queue_depth) % stages],
                                )
                            T.tma_copy(
                                x_view[
                                    0,
                                    by * block_m,
                                    (part * tiles + tile - queue_depth + stages)
                                    * block_k,
                                ],
                                b[(tile - queue_depth) % stages, :, :],
                                barrier=ready[(tile - queue_depth) % stages],
                            )
                            T.mbarrier_arrive(ready[(tile - queue_depth) % stages])
                T.wait_wgmma(0)
                T.warpgroup_fence_operand(acc, num_regs=tile_rows * block_m // 128)
            else:
                for tile in T.serial(tiles):
                    if transposed_weights:
                        T.copy(
                            weight_view[
                                0, (part * tiles + tile) * block_k, bx * tile_rows
                            ],
                            a[0, :, :],
                        )
                    else:
                        T.copy(
                            weight_view[
                                0, bx * tile_rows, (part * tiles + tile) * block_k
                            ],
                            a[0, :, :],
                        )
                    T.copy(
                        x_view[0, by * block_m, (part * tiles + tile) * block_k],
                        b[0, :, :],
                    )
                    T.gemm(
                        a[0, :, :],
                        b[0, :, :],
                        acc,
                        transpose_A=transposed_weights,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow
                        if register_epilogue
                        else T.GemmWarpPolicy.Square,
                    )
            if register_epilogue:
                for row, col in T.Parallel(tile_rows, block_m):
                    value = T.alloc_var("float32")
                    if paired:
                        # Every lane executes the shuffle, including peers
                        # that do not own a final output or fall in a tail.
                        value = T.call_extern(
                            "float32", "tensor_pair_sum_xor4", acc[row, col]
                        )
                    else:
                        value = acc[row, col]
                    if (
                        (not paired or row % 2 == 0)
                        and by * block_m + col < m
                        and bx * block_n + row // (2 if paired else 1) < n
                    ):
                        out[
                            part,
                            by * block_m + col,
                            bx * block_n + row // (2 if paired else 1),
                        ] = value
            else:
                T.copy(acc, epilogue)
            if cluster_reduce:
                for i, j in T.Parallel(block_m, block_n):
                    if paired:
                        contribution[0, i, j] = (
                            epilogue[2 * j, i] + epilogue[2 * j + 1, i]
                        )
                    else:
                        contribution[0, i, j] = epilogue[j, i]
                T.sync_threads()
                # The memory planner may reuse operand storage for the inbox.
                # All peers must finish their MMA reads before a remote sender
                # can write that inbox, even if the sender finishes early.
                T.call_extern("handle", "cute::cluster_sync")
                T.copy_cluster(
                    contribution,
                    gathered[part, :, :],
                    dst_block=0,
                    remote_barrier=completed[0],
                )
                if part == 0:
                    T.mbarrier_wait_parity(completed[0], 0)
                    for i, j in T.Parallel(block_m, block_n):
                        if by * block_m + i < m and bx * block_n + j < n:
                            value = T.alloc_var("float32")
                            value = 0
                            for peer in T.serial(splits):
                                value += gathered[peer, i, j]
                            out[0, by * block_m + i, bx * block_n + j] = value
                # No CTA may retire shared storage while another CTA still
                # reads its data or observes an outstanding bulk copy.
                T.call_extern("handle", "cute::cluster_sync")
            elif not register_epilogue:
                for i, j in T.Parallel(block_m, block_n):
                    if by * block_m + i < m and bx * block_n + j < n:
                        if paired:
                            out[part, by * block_m + i, bx * block_n + j] = (
                                epilogue[2 * j, i] + epilogue[2 * j + 1, i]
                            )
                        else:
                            out[part, by * block_m + i, bx * block_n + j] = epilogue[
                                j, i
                            ]

    return product


def ring_events(tiles, stages, queue_depth):
    """Reference ownership trace; no slot may be overwritten while MMA reads it."""
    if tiles < 1 or stages < queue_depth + 1 or not 0 <= queue_depth <= 7:
        raise ValueError("invalid bounded pipeline")
    events = [("load", tile, tile % stages) for tile in range(min(stages, tiles))]
    pending = []
    for tile in range(tiles):
        events.append(("issue", tile, tile % stages))
        pending.append(tile)
        if tile >= queue_depth:
            finished = pending.pop(0)
            events.append(("complete", finished, finished % stages))
            following = tile - queue_depth + stages
            if following < tiles:
                events.append(("load", following, following % stages))
    events.extend(("complete", tile, tile % stages) for tile in pending)
    return events
