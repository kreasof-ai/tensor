"""Inspectable CUDA primitives for Qwen's native FP8 and gated delta state.

Factories are producer-only. Consumers load the resulting Tensor artifacts.
These primitives are experimental until a full model oracle and serving run
qualify them together; kernel timings are never model throughput claims.
"""


def source(kind, parameters):
    from tensor.compiler.entry import export_source
    return export_source(__name__, 'make_kernel', kind, parameters,
        dependencies=('tensor.compiler.entry', 'tensor.compiler.cuda_lowering', 'tensor_llm.qwen35.kernels.fp8_kv'))


def make_kernel(kind, p):
    import tilelang.language as T

    if p.get('kv_dtype') == 'fp8':
        from .fp8_kv import make_kernel as fp8_kv
        return fp8_kv(kind, p)

    if kind == 'moe_groups':
        r, top = p['r'], p['top']
        @T.prim_func
        def kernel(ids: T.Tensor((r, top), 'int32'),
                   experts: T.Tensor((r * top,), 'int32'),
                   routes: T.Tensor((r * top, r), 'int32')):
            with T.Kernel(r * top, threads=32) as group:
                expert = ids[group // top, group % top]
                owner = T.alloc_var('int32')
                owner = group
                for earlier in T.serial(r * top):
                    if (earlier < group) & (ids[earlier // top, earlier % top] == expert):
                        owner = T.min(owner, earlier)
                for i in T.Parallel(1):
                    experts[group] = T.if_then_else(owner == group, expert, -1)
                for b in T.Parallel(r):
                    rank = T.alloc_var('int32')
                    rank = -1
                    for j in T.serial(top):
                        if ids[b, j] == expert:
                            rank = j
                    routes[group, b] = T.if_then_else(owner == group, rank, -1)
        return kernel

    if kind == 'fp8_experts':
        r, k, o, top, e = (p[n] for n in ('r', 'k', 'o', 'top', 'experts'))
        columns, threads = p.get('columns', 4), p.get('threads', 128)
        routed = p.get('routed_input', False)
        from tensor.compiler.entry import primitive
        arguments = [('x', (r, top, k) if routed else (r, k), 'bfloat16'),
            ('w', (e, o, k), 'uint8'), ('scales', (e, T.ceildiv(o, 128), k // 128), 'bfloat16'),
            ('experts', (r * top,), 'int32'), ('routes', (r * top, r), 'int32'),
            ('out', (r, top, o), 'float32')]
        @T.macro
        def algorithm(x, w, scales, experts, routes, out):
            with T.Kernel(o // columns, r * top, threads=threads) as (bx, group):
                expert = experts[group]
                if expert >= 0:
                    accum = T.alloc_fragment((r, columns, 32), 'float32')
                    total = T.alloc_fragment((r, columns), 'float32')
                    coefficient = T.alloc_fragment((columns, 32), 'float32')
                    activation = T.alloc_shared((r, 128), 'float32')
                    absolute = T.alloc_fragment((r, 128), 'float32')
                    maximum = T.alloc_fragment((r,), 'float32')
                    T.clear(accum)
                    for tile in T.serial(k // 128):
                        for b, j in T.Parallel(r, 128):
                            if routes[group, b] >= 0:
                                if routed:
                                    absolute[b, j] = T.abs(T.cast(x[b, routes[group, b], tile * 128 + j], 'float32'))
                                else:
                                    absolute[b, j] = T.abs(T.cast(x[b, tile * 128 + j], 'float32'))
                            else:
                                absolute[b, j] = 0
                        T.reduce_max(absolute, maximum, dim=1)
                        for b, j in T.Parallel(r, 128):
                            if routes[group, b] >= 0:
                                scale = T.max(maximum[b], T.float32(1e-12)) / T.float32(448)
                                if routed:
                                    value = T.cast(x[b, routes[group, b], tile * 128 + j], 'float32')
                                else:
                                    value = T.cast(x[b, tile * 128 + j], 'float32')
                                encoded = T.call_extern('uint32', 'tensor_encode_e4m3', value / scale)
                                activation[b, j] = T.call_extern('float32', 'tensor_decode_e4m3', encoded) * scale
                            else:
                                activation[b, j] = 0
                        for part in T.unroll(4):
                            for n, lane in T.Parallel(columns, 32):
                                coefficient[n, lane] = T.call_extern('float32', 'tensor_decode_e4m3',
                                    T.cast(w[expert, bx * columns + n, tile * 128 + part * 32 + lane], 'uint32'))
                                coefficient[n, lane] *= T.cast(scales[expert, (bx * columns + n) // 128, tile], 'float32')
                            for b, n, lane in T.Parallel(r, columns, 32):
                                accum[b, n, lane] = T.ieee_fmaf(activation[b, part * 32 + lane],
                                    coefficient[n, lane], accum[b, n, lane])
                    T.reduce_sum(accum, total, dim=2)
                    for b, n in T.Parallel(r, columns):
                        if routes[group, b] >= 0:
                            out[b, routes[group, b], bx * columns + n] = total[b, n]
        return primitive(arguments, algorithm)

    if kind == 'bf16_linear':
        r, k, o = p['r'], p['k'], p['o']
        @T.prim_func
        def kernel(x: T.Tensor((r, k), 'bfloat16'), w: T.Tensor((o, k), 'bfloat16'),
                   out: T.Tensor((r, o), 'float32')):
            with T.Kernel(T.ceildiv(o, 4), threads=128) as bx:
                accum = T.alloc_fragment((r, 4, 32), 'float32')
                total = T.alloc_fragment((r, 4), 'float32')
                T.clear(accum)
                for tile in T.serial(k // 32):
                    for b, n, lane in T.Parallel(r, 4, 32):
                        if bx * 4 + n < o:
                            accum[b, n, lane] = T.ieee_fmaf(T.cast(x[b, tile * 32 + lane], 'float32'),
                                T.cast(w[bx * 4 + n, tile * 32 + lane], 'float32'), accum[b, n, lane])
                T.reduce_sum(accum, total, dim=2)
                for b, n in T.Parallel(r, 4):
                    if bx * 4 + n < o:
                        out[b, bx * 4 + n] = total[b, n]
        return kernel

    if kind == 'embedding':
        r, c, vocab = p['r'], p['c'], p['vocab']
        @T.prim_func
        def kernel(ids: T.Tensor((r,), 'int32'), w: T.Tensor((vocab, c), 'bfloat16'),
                   out: T.Tensor((r, c), 'bfloat16')):
            with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        out[b, col] = T.if_then_else(ids[b] >= 0, w[T.max(ids[b], 0), col], 0)
        return kernel

    if kind in ('rms', 'add_rms'):
        r, c, eps = p['r'], p['c'], p['eps']
        from tensor.compiler.entry import primitive
        arguments = ([('x', (r, c), 'float32')] if kind == 'add_rms' else []) + [
            ('residual', (r, c), 'bfloat16'), ('w', (c,), 'bfloat16'), ('out', (r, c), 'bfloat16')]
        @T.macro
        def algorithm(x, residual, w, out):
            with T.Kernel(r, threads=256) as b:
                values = T.alloc_fragment((c,), 'float32')
                squares = T.alloc_fragment((c,), 'float32')
                total = T.alloc_fragment((1,), 'float32')
                for j in T.Parallel(c):
                    if kind == 'add_rms':
                        values[j] = T.cast(T.cast(T.cast(x[b, j], 'bfloat16'), 'float32')
                            + T.cast(residual[b, j], 'float32'), 'bfloat16')
                    else:
                        values[j] = T.cast(residual[b, j], 'float32')
                    squares[j] = values[j] * values[j]
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(c):
                    if kind == 'add_rms':
                        residual[b, j] = values[j]
                    out[b, j] = values[j] * T.rsqrt(total[0] / c + eps) * (1 + T.cast(w[j], 'float32'))
        @T.macro
        def normalize(residual, w, out):
            algorithm(residual, residual, w, out)
        return primitive(arguments, algorithm if kind == 'add_rms' else normalize)

    if kind == 'router':
        r, e, top = p['r'], p['experts'], p['top']
        @T.prim_func
        def kernel(logits: T.Tensor((r, e), 'float32'), ids: T.Tensor((r, top), 'int32'),
                   weights: T.Tensor((r, top), 'float32')):
            with T.Kernel(r, threads=256) as b:
                values = T.alloc_fragment((e,), 'float32')
                maximum = T.alloc_fragment((1,), 'float32')
                indices = T.alloc_fragment((e,), 'int32')
                first = T.alloc_fragment((1,), 'int32')
                selected = T.alloc_shared((top,), 'float32')
                for j in T.Parallel(e):
                    values[j] = T.cast(T.cast(logits[b, j], 'bfloat16'), 'float32')
                for rank in T.serial(top):
                    T.reduce_max(values, maximum, dim=0)
                    for j in T.Parallel(e):
                        indices[j] = T.if_then_else(values[j] == maximum[0], j, e)
                    T.reduce_min(indices, first, dim=0)
                    for j in T.Parallel(1):
                        ids[b, rank] = first[0]
                        selected[rank] = maximum[0]
                    for j in T.Parallel(e):
                        if j == first[0]:
                            values[j] = -T.infinity('float32')
                for j in T.Parallel(top):
                    denominator = T.alloc_var('float32')
                    denominator = 0
                    for rank in T.serial(top):
                        denominator += T.exp(selected[rank] - selected[0])
                    weights[b, j] = T.exp(selected[j] - selected[0]) / denominator
        return kernel

    if kind == 'swiglu':
        r, c = p['r'], p['c']
        @T.prim_func
        def kernel(g: T.Tensor((r, c), 'float32'), u: T.Tensor((r, c), 'float32'),
                   out: T.Tensor((r, c), 'bfloat16')):
            with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        gate = T.cast(T.cast(g[b, col], 'bfloat16'), 'float32')
                        up = T.cast(T.cast(u[b, col], 'bfloat16'), 'float32')
                        out[b, col] = gate / (1 + T.exp(-gate)) * up
        return kernel

    if kind == 'swiglu_experts':
        r, top, c = p['r'], p['top'], p['c']
        @T.prim_func
        def kernel(g: T.Tensor((r, top, c), 'float32'), u: T.Tensor((r, top, c), 'float32'),
                   out: T.Tensor((r, top, c), 'bfloat16')):
            with T.Kernel(r, top, T.ceildiv(c, 256), threads=256) as (b, rank, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        gate = T.cast(T.cast(g[b, rank, col], 'bfloat16'), 'float32')
                        up = T.cast(T.cast(u[b, rank, col], 'bfloat16'), 'float32')
                        out[b, rank, col] = gate / (1 + T.exp(-gate)) * up
        return kernel

    if kind == 'moe_combine':
        r, top, c = p['r'], p['top'], p['c']
        @T.prim_func
        def kernel(experts: T.Tensor((r, top, c), 'float32'), weights: T.Tensor((r, top), 'float32'),
                   shared: T.Tensor((r, c), 'float32'), gate: T.Tensor((r, 1), 'float32'),
                   out: T.Tensor((r, c), 'float32')):
            with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    if col < c:
                        total = T.alloc_var('float32')
                        total = 0
                        for rank in T.serial(top):
                            total += T.cast(T.cast(experts[b, rank, col], 'bfloat16'), 'float32') * weights[b, rank]
                        shared_value = T.cast(T.cast(shared[b, col], 'bfloat16'), 'float32')
                        shared_gate = T.cast(T.cast(gate[b, 0], 'bfloat16'), 'float32')
                        out[b, col] = total + shared_value / (1 + T.exp(-shared_gate))
        return kernel

    if kind == 'gdn_conv':
        r, channels = p['r'], 8192
        @T.prim_func
        def kernel(x: T.Tensor((r, channels), 'float32'), w: T.Tensor((channels, 1, 4), 'bfloat16'),
                   active: T.Tensor((r,), 'int32'), state: T.Tensor((r, channels, 3), 'float32'),
                   out: T.Tensor((r, channels), 'float32')):
            with T.Kernel(r, channels // 256, threads=256) as (b, tile):
                for j in T.Parallel(256):
                    col = tile * 256 + j
                    current = T.cast(T.cast(x[b, col], 'bfloat16'), 'float32')
                    value = T.alloc_var('float32')
                    value = current * T.cast(w[col, 0, 3], 'float32')
                    for t in T.unroll(3):
                        value += state[b, col, t] * T.cast(w[col, 0, t], 'float32')
                    if active[b] != 0:
                        state[b, col, 0] = state[b, col, 1]
                        state[b, col, 1] = state[b, col, 2]
                        state[b, col, 2] = current
                        out[b, col] = T.cast(value / (1 + T.exp(-value)), 'bfloat16')
                    else:
                        out[b, col] = 0
        return kernel

    if kind == 'gdn_prepare':
        r, heads, d = p['r'], 32, 128
        @T.prim_func
        def kernel(qkv: T.Tensor((r, 8192), 'float32'), a: T.Tensor((r, heads), 'float32'),
                   b: T.Tensor((r, heads), 'float32'), dt: T.Tensor((heads,), 'bfloat16'),
                   A: T.Tensor((heads,), 'float32'), q: T.Tensor((r, heads, d), 'float32'),
                   k: T.Tensor((r, heads, d), 'float32'), v: T.Tensor((r, heads, d), 'float32'),
                   g: T.Tensor((r, heads), 'float32'), beta: T.Tensor((r, heads), 'float32')):
            with T.Kernel(r, heads, threads=128) as (slot, head):
                qs = T.alloc_fragment((d,), 'float32')
                ks = T.alloc_fragment((d,), 'float32')
                qq = T.alloc_fragment((d,), 'float32')
                kk = T.alloc_fragment((d,), 'float32')
                qtotal = T.alloc_fragment((1,), 'float32')
                ktotal = T.alloc_fragment((1,), 'float32')
                for j in T.Parallel(d):
                    qq[j] = qkv[slot, head // 2 * d + j]
                    kk[j] = qkv[slot, 2048 + head // 2 * d + j]
                    qs[j] = qq[j] * qq[j]
                    ks[j] = kk[j] * kk[j]
                T.reduce_sum(qs, qtotal, dim=0)
                T.reduce_sum(ks, ktotal, dim=0)
                for j in T.Parallel(d):
                    q[slot, head, j] = qq[j] * T.rsqrt(qtotal[0] + T.float32(1e-6)) * d ** -.5
                    k[slot, head, j] = kk[j] * T.rsqrt(ktotal[0] + T.float32(1e-6))
                    v[slot, head, j] = qkv[slot, 4096 + head * d + j]
                for j in T.Parallel(1):
                    value = T.cast(T.cast(a[slot, head], 'bfloat16'), 'float32') + T.cast(dt[head], 'float32')
                    softplus = T.max(value, 0) + T.log(1 + T.exp(-T.abs(value)))
                    g[slot, head] = -T.exp(A[head]) * softplus
                    beta[slot, head] = 1 / (1 + T.exp(-T.cast(T.cast(b[slot, head], 'bfloat16'), 'float32')))
        return kernel

    if kind == 'gdn_norm':
        r, heads, d, eps = p['r'], 32, 128, p['eps']
        @T.prim_func
        def kernel(x: T.Tensor((r, heads, d), 'float32'), z: T.Tensor((r, 4096), 'float32'),
                   w: T.Tensor((d,), 'float32'), out: T.Tensor((r, 4096), 'bfloat16')):
            with T.Kernel(r, heads, threads=128) as (slot, head):
                values = T.alloc_fragment((d,), 'float32')
                squares = T.alloc_fragment((d,), 'float32')
                total = T.alloc_fragment((1,), 'float32')
                for j in T.Parallel(d):
                    values[j] = T.cast(T.cast(x[slot, head, j], 'bfloat16'), 'float32')
                    squares[j] = values[j] * values[j]
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    gate = T.cast(T.cast(z[slot, head * d + j], 'bfloat16'), 'float32')
                    out[slot, head * d + j] = values[j] * T.rsqrt(total[0] / d + eps) * w[j] * gate / (1 + T.exp(-gate))
        return kernel

    if kind in ('attention_q', 'attention_kv'):
        r, d, eps, cap = p['r'], 256, p['eps'], p['capacity']
        theta = p['theta']
        from tensor.compiler.entry import primitive
        qkind = kind == 'attention_q'
        heads = 16 if qkind else 2
        arguments = ([('projection', (r, 8192), 'float32')] if qkind else [
            ('projection', (r, 512), 'float32'), ('values', (r, 512), 'float32')]) + [
            ('w', (d,), 'bfloat16'), ('positions', (r,), 'int32'), ('active', (r,), 'int32')]
        arguments += ([('q', (r, heads, d), 'bfloat16')] if qkind else [
            ('kc', (r, heads, cap, d), 'bfloat16'), ('vc', (r, heads, cap, d), 'bfloat16')])
        @T.macro
        def query(projection, w, positions, active, q):
            with T.Kernel(r, heads, threads=256) as (slot, head):
                squares = T.alloc_fragment((d,), 'float32')
                total = T.alloc_fragment((1,), 'float32')
                normal = T.alloc_shared((d,), 'bfloat16')
                for j in T.Parallel(d):
                    value = T.cast(T.cast(projection[slot, head * 512 + j], 'bfloat16'), 'float32')
                    squares[j] = value * value
                T.reduce_sum(squares, total, dim=0)
                for j in T.Parallel(d):
                    value = T.cast(T.cast(projection[slot, head * 512 + j], 'bfloat16'), 'float32')
                    normal[j] = value * T.rsqrt(total[0] / d + eps) * (1 + T.cast(w[j], 'float32'))
                for j in T.Parallel(d):
                    value = T.alloc_var('float32')
                    value = T.cast(normal[j], 'float32')
                    if j < 64:
                        angle = positions[slot] * T.exp(T.cast(-2 * (j % 32), 'float32') / 64 * T.log(T.float32(theta)))
                        pair = T.cast(normal[(j + 32) % 64], 'float32')
                        value = value * T.cos(angle) + T.if_then_else(j < 32, -pair, pair) * T.sin(angle)
                    q[slot, head, j] = T.if_then_else(active[slot] != 0, value, 0)
        @T.macro
        def keyvalue(projection, values, w, positions, active, kc, vc):
            with T.Kernel(r, heads, threads=256) as (slot, head):
                if active[slot] != 0:
                    squares = T.alloc_fragment((d,), 'float32')
                    total = T.alloc_fragment((1,), 'float32')
                    normal = T.alloc_shared((d,), 'bfloat16')
                    for j in T.Parallel(d):
                        value = T.cast(T.cast(projection[slot, head * d + j], 'bfloat16'), 'float32')
                        squares[j] = value * value
                    T.reduce_sum(squares, total, dim=0)
                    for j in T.Parallel(d):
                        value = T.cast(T.cast(projection[slot, head * d + j], 'bfloat16'), 'float32')
                        normal[j] = value * T.rsqrt(total[0] / d + eps) * (1 + T.cast(w[j], 'float32'))
                    for j in T.Parallel(d):
                        value = T.alloc_var('float32')
                        value = T.cast(normal[j], 'float32')
                        if j < 64:
                            angle = positions[slot] * T.exp(T.cast(-2 * (j % 32), 'float32') / 64 * T.log(T.float32(theta)))
                            pair = T.cast(normal[(j + 32) % 64], 'float32')
                            value = value * T.cos(angle) + T.if_then_else(j < 32, -pair, pair) * T.sin(angle)
                        kc[slot, head, positions[slot], j] = value
                        vc[slot, head, positions[slot], j] = T.cast(values[slot, head * d + j], 'bfloat16')
        return primitive(arguments, query if qkind else keyvalue)

    if kind == 'attention_partial':
        r, cap, splits, d = p['r'], p['capacity'], p.get('splits', 16), 256
        @T.prim_func
        def kernel(q: T.Tensor((r, 16, d), 'bfloat16'),
                   kc: T.Tensor((r, 2, cap, d), 'bfloat16'), vc: T.Tensor((r, 2, cap, d), 'bfloat16'),
                   positions: T.Tensor((r,), 'int32'), active: T.Tensor((r,), 'int32'),
                   out: T.Tensor((r, 2, splits, 8, d), 'float32'),
                   stats: T.Tensor((r, 2, splits, 8, 2), 'float32')):
            with T.Kernel(r, 2, splits, threads=128) as (slot, head, part):
                query = T.alloc_shared((16, d), 'bfloat16')
                key = T.alloc_shared((64, d), 'bfloat16')
                value = T.alloc_shared((64, d), 'bfloat16')
                prob = T.alloc_shared((16, 64), 'bfloat16')
                scores = T.alloc_fragment((16, 64), 'float32')
                result = T.alloc_fragment((16, d), 'float32')
                maximum = T.alloc_fragment((16,), 'float32')
                previous = T.alloc_fragment((16,), 'float32')
                factor = T.alloc_fragment((16,), 'float32')
                normalizer = T.alloc_fragment((16,), 'float32')
                total = T.alloc_fragment((16,), 'float32')
                T.clear(result)
                T.clear(normalizer)
                T.fill(maximum, -T.infinity('float32'))
                for i, j in T.Parallel(16, d):
                    query[i, j] = T.if_then_else(i < 8, q[slot, head * 8 + i % 8, j], 0)
                count = positions[slot] + 1
                tiles = T.ceildiv(T.ceildiv(count, 64), splits)
                if (active[slot] != 0) & (part * tiles * 64 < count):
                    for tile in T.serial(tiles):
                        for i, j in T.Parallel(64, d):
                            pos = (part * tiles + tile) * 64 + i
                            key[i, j] = T.if_then_else(pos < count, kc[slot, head, T.min(pos, cap - 1), j], 0)
                            value[i, j] = T.if_then_else(pos < count, vc[slot, head, T.min(pos, cap - 1), j], 0)
                        T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                        T.copy(maximum, previous)
                        for i, j in T.Parallel(16, 64):
                            scores[i, j] = T.if_then_else((part * tiles + tile) * 64 + j < count,
                                scores[i, j] * d ** -.5, -T.infinity('float32'))
                        T.reduce_max(scores, maximum, dim=1, clear=False)
                        for i in T.Parallel(16):
                            factor[i] = T.exp(previous[i] - maximum[i])
                        for i, j in T.Parallel(16, 64):
                            scores[i, j] = T.exp(scores[i, j] - maximum[i])
                        T.reduce_sum(scores, total, dim=1)
                        for i in T.Parallel(16):
                            normalizer[i] = normalizer[i] * factor[i] + total[i]
                        for i, j in T.Parallel(16, d):
                            result[i, j] *= factor[i]
                        T.copy(scores, prob)
                        T.gemm(prob, value, result)
                for i, j in T.Parallel(8, d):
                    out[slot, head, part, i, j] = result[i, j]
                for i in T.Parallel(8):
                    stats[slot, head, part, i, 0] = maximum[i]
                    stats[slot, head, part, i, 1] = normalizer[i]
        return kernel

    if kind == 'attention_merge':
        r, splits, d = p['r'], p.get('splits', 16), 256
        @T.prim_func
        def kernel(partial: T.Tensor((r, 2, splits, 8, d), 'float32'),
                   stats: T.Tensor((r, 2, splits, 8, 2), 'float32'),
                   projection: T.Tensor((r, 8192), 'float32'), active: T.Tensor((r,), 'int32'),
                   out: T.Tensor((r, 4096), 'bfloat16')):
            with T.Kernel(r, 16, threads=256) as (slot, head):
                for j in T.Parallel(d):
                    maximum = T.alloc_var('float32')
                    numerator = T.alloc_var('float32')
                    denominator = T.alloc_var('float32')
                    maximum = -T.infinity('float32')
                    for part in T.serial(splits):
                        maximum = T.max(maximum, stats[slot, head // 8, part, head % 8, 0])
                    numerator = 0
                    denominator = 0
                    if active[slot] != 0:
                        for part in T.serial(splits):
                            factor = T.exp(stats[slot, head // 8, part, head % 8, 0] - maximum)
                            numerator += partial[slot, head // 8, part, head % 8, j] * factor
                            denominator += stats[slot, head // 8, part, head % 8, 1] * factor
                        gate = T.cast(T.cast(projection[slot, head * 512 + d + j], 'bfloat16'), 'float32')
                        # The attention output is BF16 before its gate, as in
                        # prefill and the checkpoint's BF16 inference path.
                        value = T.cast(T.cast(numerator / denominator, 'bfloat16'), 'float32')
                        out[slot, head * d + j] = value / (1 + T.exp(-gate))
                    else:
                        out[slot, head * d + j] = 0
        return kernel

    if kind == 'argmax':
        r, vocab = p['r'], p['vocab']
        @T.prim_func
        def kernel(logits: T.Tensor((r, vocab), 'float32'), tokens: T.Tensor((r,), 'int32'),
                   positions: T.Tensor((r,), 'int32'), active: T.Tensor((r,), 'int32')):
            with T.Kernel(r, threads=256) as slot:
                tx = T.get_thread_binding()
                best = T.alloc_var('float32')
                index = T.alloc_var('int32')
                values = T.alloc_shared((256,), 'float32')
                indices = T.alloc_shared((256,), 'int32')
                best = -T.infinity('float32')
                index = vocab
                for tile in T.serial(T.ceildiv(vocab, 256)):
                    i = tile * 256 + tx
                    if i < vocab:
                        if (logits[slot, i] > best) | ((logits[slot, i] == best) & (i < index)):
                            best = logits[slot, i]
                            index = i
                values[tx] = best
                indices[tx] = index
                T.sync_threads()
                for step in T.unroll(8):
                    stride = 128 >> step
                    if tx < stride:
                        if ((values[tx + stride] > values[tx])
                                | ((values[tx + stride] == values[tx]) & (indices[tx + stride] < indices[tx]))):
                            values[tx] = values[tx + stride]
                            indices[tx] = indices[tx + stride]
                    T.sync_threads()
                if tx == 0:
                    tokens[slot] = T.if_then_else(active[slot] != 0, indices[0], -1)
                    if active[slot] != 0:
                        positions[slot] += 1
        return kernel

    if kind == 'e4m3_decode':
        @T.prim_func
        def kernel(bits: T.Tensor((256,), 'uint8'), out: T.Tensor((256,), 'float32')):
            with T.Kernel(1, threads=256):
                for i in T.Parallel(256):
                    out[i] = T.call_extern('float32', 'tensor_decode_e4m3', T.cast(bits[i], 'uint32'))
        return kernel

    if kind == 'fp8_linear':
        r, k, o = p['r'], p['k'], p['o']
        columns, threads = p.get('columns', 4), p.get('threads', 128)
        if (any(type(v) is not int or v <= 0 for v in (r, k, o, columns, threads))
                or k % 128 or o % columns or threads % 32 or threads > 1024
                or r > 8 or columns * 32 % threads):
            raise ValueError('invalid native FP8 GEMV schedule')
        @T.prim_func
        def kernel(x: T.Tensor((r, k), 'bfloat16'),
                   w: T.Tensor((o, k), 'uint8'),
                   scales: T.Tensor((T.ceildiv(o, 128), k // 128), 'bfloat16'),
                   out: T.Tensor((r, o), 'float32')):
            with T.Kernel(o // columns, threads=threads) as bx:
                accum = T.alloc_fragment((r, columns, 32), 'float32')
                total = T.alloc_fragment((r, columns), 'float32')
                coefficient = T.alloc_fragment((columns, 32), 'float32')
                activation = T.alloc_shared((r, 128), 'float32')
                absolute = T.alloc_fragment((r, 128), 'float32')
                maximum = T.alloc_fragment((r,), 'float32')
                T.clear(accum)
                for tile in T.serial(k // 128):
                    for b, j in T.Parallel(r, 128):
                        absolute[b, j] = T.abs(T.cast(x[b, tile * 128 + j], 'float32'))
                    T.reduce_max(absolute, maximum, dim=1)
                    for b, j in T.Parallel(r, 128):
                        scale = T.max(maximum[b], T.float32(1e-12)) / T.float32(448)
                        encoded = T.call_extern('uint32', 'tensor_encode_e4m3',
                            T.cast(x[b, tile * 128 + j], 'float32') / scale)
                        activation[b, j] = T.call_extern('float32', 'tensor_decode_e4m3', encoded) * scale
                    for part in T.unroll(4):
                        for n, lane in T.Parallel(columns, 32):
                            coefficient[n, lane] = T.call_extern('float32', 'tensor_decode_e4m3',
                                T.cast(w[bx * columns + n, tile * 128 + part * 32 + lane], 'uint32'))
                            coefficient[n, lane] *= T.cast(scales[(bx * columns + n) // 128, tile], 'float32')
                        for b, n, lane in T.Parallel(r, columns, 32):
                            accum[b, n, lane] = T.ieee_fmaf(
                                T.cast(activation[b, part * 32 + lane], 'float32'),
                                coefficient[n, lane], accum[b, n, lane])
                T.reduce_sum(accum, total, dim=2)
                for b, n in T.Parallel(r, columns):
                    out[b, bx * columns + n] = total[b, n]
        return kernel

    if kind == 'gdn_recurrent':
        slots, heads, key, value = (p[n] for n in ('slots', 'heads', 'key', 'value'))
        threads = p.get('threads', 128)
        tile = p.get('value_tile', 32)
        if (any(type(v) is not int or v <= 0 for v in (slots, heads, key, value, tile, threads))
                or key & (key - 1) or tile & (tile - 1) or value % tile
                or threads not in (64, 128, 256)):
            raise ValueError('invalid gated delta schedule')
        # Each request's compact Q/K is already expanded to value heads and
        # normalized. log_decay is -exp(A_log)*softplus(a+dt_bias); beta=sigmoid(b).
        @T.prim_func
        def kernel(q: T.Tensor((slots, heads, key), 'float32'),
                   k: T.Tensor((slots, heads, key), 'float32'),
                   v: T.Tensor((slots, heads, value), 'float32'),
                   log_decay: T.Tensor((slots, heads), 'float32'),
                   beta: T.Tensor((slots, heads), 'float32'),
                   active: T.Tensor((slots,), 'int32'),
                   state: T.Tensor((slots, heads, value, key), 'float32'),
                   out: T.Tensor((slots, heads, value), 'float32')):
            with T.Kernel(slots, heads, value // tile, threads=threads) as (slot, head, part):
                if active[slot] != 0:
                    matrix = T.alloc_fragment((tile, key), 'float32')
                    prediction = T.alloc_fragment((tile, key), 'float32')
                    inner = T.alloc_fragment((tile,), 'float32')
                    result = T.alloc_fragment((tile,), 'float32')
                    q_shared = T.alloc_shared((key,), 'float32')
                    k_shared = T.alloc_shared((key,), 'float32')
                    for j in T.Parallel(key):
                        q_shared[j] = q[slot, head, j]
                        k_shared[j] = k[slot, head, j]
                    for i, j in T.Parallel(tile, key):
                        matrix[i, j] = state[slot, head, part * tile + i, j] * T.exp(log_decay[slot, head])
                        prediction[i, j] = matrix[i, j] * k_shared[j]
                    T.reduce_sum(prediction, inner, dim=1)
                    for i, j in T.Parallel(tile, key):
                        matrix[i, j] += k_shared[j] * beta[slot, head] * (v[slot, head, part * tile + i] - inner[i])
                        state[slot, head, part * tile + i, j] = matrix[i, j]
                        prediction[i, j] = matrix[i, j] * q_shared[j]
                    T.reduce_sum(prediction, result, dim=1)
                    for i in T.Parallel(tile):
                        out[slot, head, part * tile + i] = result[i]
                else:
                    for i in T.Parallel(tile):
                        out[slot, head, part * tile + i] = 0
        return kernel

    raise ValueError('unsupported Qwen primitive: ' + kind)
