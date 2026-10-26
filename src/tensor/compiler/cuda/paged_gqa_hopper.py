"""Warp-specialized paged attention with a bounded KV ring and explicit WGMMA.

Scattered pages are gathered by a producer warp group. Consumers use native FP8
QK, BF16 probabilities/values and FP32 online softmax/accumulation. The next QK
is queued before the current PV so its softmax can overlap the previous PV.
The ordinary-MMA mirror exercises the same paging, masks and barrier ownership.
"""


def make_hopper_partial(
    batch, length, hq, hkv, pages, rows, page, window, splits,
    bk=128, stages=2, native_async=True, qfactor=5.4730064863875,
):
    import tilelang.language as T

    if bk not in (64, 128, 256) or stages not in (2, 3):
        raise ValueError("attention needs 64/128/256 keys and a two/three-stage ring")
    if hq != hkv * 8 or length not in (1, 8):
        raise ValueError("head dimension 64, GQA group eight and query length one/eight required")
    group, qm = hq // hkv, 64
    qdtype = "float8_e4m3" if native_async else "bfloat16"

    copy_source = r"""
__device__ __forceinline__ void tensor_gqa_cp16(void* dst, const void* src, bool valid) {
    unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::
                 "r"(address), "l"(src), "r"(valid ? 16 : 0) : "memory");
}
"""

    @T.prim_func
    def hopper_partial(
        query: T.Tensor((batch * length, hq, 64), "float8_e4m3"),
        kc: T.Tensor((rows, hkv, 64), "float8_e4m3"),
        vc: T.Tensor((rows, hkv, 64), "float8_e4m3"),
        ks: T.Tensor((1,), "float32"),
        vs: T.Tensor((1,), "float32"),
        table: T.Tensor((batch, pages), "int32"),
        lengths: T.Tensor((batch,), "int32"),
        out: T.Tensor((batch, hkv, splits, qm, 64), "float32"),
        stats: T.Tensor((batch, hkv, splits, qm, 2), "float32"),
    ):
        with T.Kernel(batch * hkv, splits, threads=256) as (index, part):
            request, head = index // hkv, index % hkv
            seq = lengths[request]
            start = T.max(seq - window + 1, 0) if window else 0
            stop = seq + length
            span = T.ceildiv(T.ceildiv(stop - start, bk), splits)
            first = start + part * span * bk
            count = T.max(T.min(span, T.ceildiv(stop - first, bk)), 0)
            q = T.alloc_shared((qm, 64), qdtype)
            key = T.alloc_shared((stages, bk, 64), qdtype)
            value = T.alloc_shared((stages, bk, 64), "bfloat16")
            prob = T.alloc_shared((stages, qm, bk), "bfloat16")
            if native_async:
                ready = T.alloc_barrier([128] * stages)
                free = T.alloc_barrier([128] * stages)
            score = T.alloc_fragment((qm, bk), "float32")
            acc = T.alloc_fragment((qm, 64), "float32")
            maximum = T.alloc_fragment((qm,), "float32")
            previous = T.alloc_fragment((qm,), "float32")
            alpha = T.alloc_fragment((qm,), "float32")
            denom = T.alloc_fragment((qm,), "float32")
            added = T.alloc_fragment((qm,), "float32")
            for i, d in T.Parallel(qm, 64):
                q[i, d] = T.if_then_else(
                    i < length * group,
                    query[request * length + T.min(i // group, length - 1),
                          head * group + i % group, d], 0,
                )
            T.sync_threads()
            if native_async:
                T.import_source(copy_source)
            tx = T.get_thread_binding()
            if tx < 128:
                if native_async:
                    T.dec_max_nreg(96)
                for tile in T.serial(count):
                    slot = tile % stages
                    if native_async:
                        T.mbarrier_wait_parity(free[slot], ((tile // stages) % 2) ^ 1)
                    else:
                        T.sync_threads(4 + slot, 256)
                    if native_async:
                        for stripe in T.unroll(bk // 32):
                            j = stripe * 32 + tx // 4
                            d = tx % 4 * 16
                            position = first + tile * bk + j
                            physical = table[request, T.min(position // page, pages - 1)] * page + position % page
                            T.call_extern("handle", "tensor_gqa_cp16",
                                          T.access_ptr(key[slot, j, d], "w", 16),
                                          T.access_ptr(kc[T.min(T.max(physical, 0), rows - 1), head, d], "r", 16),
                                          position < stop)
                        T.ptx_commit_group()
                    for j, d in T.Parallel(bk, 64):
                        position = first + tile * bk + j
                        physical = table[request, T.min(position // page, pages - 1)] * page + position % page
                        if not native_async:
                            key[slot, j, d] = T.if_then_else(position < stop, kc[T.max(physical, 0), head, d], 0)
                        value[slot, j, d] = T.if_then_else(
                            position < stop, T.cast(vc[T.max(physical, 0), head, d], "bfloat16"), 0,
                        )
                    if native_async:
                        T.ptx_wait_group(0)
                        T.fence_proxy_async()
                    if native_async:
                        T.mbarrier_arrive(ready[slot])
                    else:
                        T.named_barrier_arrive(2 + slot, 256)
            else:
                if native_async:
                    T.inc_max_nreg(160)
                if not native_async:
                    for warm in T.unroll(stages):
                        T.named_barrier_arrive(4 + warm, 256)
                T.clear(acc)
                T.clear(denom)
                T.fill(maximum, -T.infinity("float32"))
                if count > 0:
                    if native_async:
                        T.mbarrier_wait_parity(ready[0], 0)
                    else:
                        T.sync_threads(2, 256)
                    if native_async:
                        T.wgmma_gemm(q, key[0, :, :], score, transpose_B=True, clear_accum=True)
                        T.wait_wgmma(0)
                        T.warpgroup_fence_operand(score, num_regs=qm * bk // 128)
                    else:
                        T.gemm(q, key[0, :, :], score, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                    for tile in T.serial(count):
                        slot = tile % stages
                        T.copy(maximum, previous)
                        for i, j in T.Parallel(qm, bk):
                            position = first + tile * bk + j
                            visible = (i < length * group and position <= seq + i // group
                                       and (position > seq + i // group - window if window else True))
                            score[i, j] = T.if_then_else(
                                visible, score[i, j] * (0.125 * 1.4426950408889634 * qfactor) * ks[0],
                                -T.infinity("float32"),
                            )
                        T.reduce_max(score, maximum, dim=1, clear=False)
                        for i in T.Parallel(qm):
                            alpha[i] = T.if_then_else(T.isfinite(previous[i]), T.exp2(previous[i] - maximum[i]), 0)
                        for i, j in T.Parallel(qm, bk):
                            score[i, j] = T.if_then_else(T.isfinite(score[i, j]), T.exp2(score[i, j] - maximum[i]), 0)
                        T.reduce_sum(score, added, dim=1)
                        for i in T.Parallel(qm):
                            denom[i] = denom[i] * alpha[i] + added[i]
                        # Previous PV is still in flight while the next softmax runs.
                        if tile > 0:
                            if native_async:
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc, num_regs=qm * 64 // 128)
                            if native_async:
                                T.mbarrier_arrive(free[(tile - 1) % stages])
                            else:
                                T.named_barrier_arrive(4 + (tile - 1) % stages, 256)
                        for i, d in T.Parallel(qm, 64):
                            acc[i, d] *= alpha[i]
                        T.copy(score, prob[slot, :, :])
                        T.sync_threads(1, 128)
                        if native_async:
                            T.fence_proxy_async()
                        # Queue next QK first: wait<1> below leaves the current PV
                        # outstanding, but makes the next score safe to consume.
                        if tile + 1 < count:
                            next_slot = (tile + 1) % stages
                            if native_async:
                                T.mbarrier_wait_parity(ready[next_slot], ((tile + 1) // stages) % 2)
                            else:
                                T.sync_threads(2 + next_slot, 256)
                            if native_async:
                                T.wgmma_gemm(q, key[next_slot, :, :], score, transpose_B=True, clear_accum=True)
                            else:
                                T.gemm(q, key[next_slot, :, :], score, transpose_B=True, clear_accum=True, policy=T.GemmWarpPolicy.FullRow)
                        if native_async:
                            T.wgmma_gemm(prob[slot, :, :], value[slot, :, :], acc)
                            if tile + 1 < count:
                                T.wait_wgmma(1)
                                T.warpgroup_fence_operand(score, num_regs=qm * bk // 128)
                            else:
                                T.wait_wgmma(0)
                                T.warpgroup_fence_operand(acc, num_regs=qm * 64 // 128)
                        else:
                            T.gemm(prob[slot, :, :], value[slot, :, :], acc, policy=T.GemmWarpPolicy.FullRow)
                    if native_async:
                        T.mbarrier_arrive(free[(count - 1) % stages])
                    else:
                        T.named_barrier_arrive(4 + (count - 1) % stages, 256)
                for i, d in T.Parallel(qm, 64):
                    out[request, head, part, i, d] = acc[i, d] * vs[0]
                for i in T.Parallel(qm):
                    stats[request, head, part, i, 0] = maximum[i]
                    stats[request, head, part, i, 1] = denom[i]
    return hopper_partial


def make_hopper_merge(batch, length, hq, hkv, splits):
    import tilelang.language as T

    group, width = hq // hkv, (2 * hq + 2 * hkv) * 64
    qm = 64

    @T.prim_func
    def gqa_merge(
        parts: T.Tensor((batch, hkv, splits, qm, 64), "float32"),
        stats: T.Tensor((batch, hkv, splits, qm, 2), "float32"),
        fused: T.Tensor((batch * length, width), "bfloat16"),
        out: T.Tensor((batch, length, hq * 64), "bfloat16"),
    ):
        with T.Kernel(batch * length, hq, threads=128) as (token, head):
            request, row = token // length, token % length * group + head % group
            kv = head // group
            maximum = T.alloc_fragment((splits,), "float32")
            largest = T.alloc_fragment((1,), "float32")
            weights = T.alloc_fragment((splits,), "float32")
            den = T.alloc_fragment((1,), "float32")
            acc = T.alloc_fragment((64,), "float32")
            square = T.alloc_fragment((64,), "float32")
            norm = T.alloc_fragment((1,), "float32")
            for s in T.Parallel(splits):
                maximum[s] = stats[request, kv, s, row, 0]
            T.reduce_max(maximum, largest, dim=0)
            for s in T.Parallel(splits):
                weights[s] = T.if_then_else(
                    T.isfinite(maximum[s]),
                    T.exp2(maximum[s] - largest[0]) * stats[request, kv, s, row, 1],
                    0,
                )
            T.reduce_sum(weights, den, dim=0)
            T.clear(acc)
            for s in T.serial(splits):
                for d in T.Parallel(64):
                    acc[d] += (
                        T.if_then_else(
                            T.isfinite(stats[request, kv, s, row, 0]),
                            T.exp2(stats[request, kv, s, row, 0] - largest[0]),
                            0,
                        )
                        * parts[request, kv, s, row, d]
                    )
            for d in T.Parallel(64):
                acc[d] /= den[0]
                square[d] = acc[d] * acc[d]
            T.reduce_sum(square, norm, dim=0)
            for d in T.Parallel(64):
                gate = T.cast(fused[token, (hq + 2 * hkv) * 64 + head * 64 + d], "float32")
                out[request, token % length, head * 64 + d] = (
                    acc[d] * T.rsqrt(norm[0] / 64 + 1e-5) / (1 + T.exp(-gate))
                )

    return gqa_merge


