"""Native join/projection adapters for the official Qwen3.5 MTP branch."""


def make_kernel(kind, p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r, c = p['r'], p['c']
    if kind == 'mtp_join':
        eps = p['eps']
        @T.macro
        def algorithm(embedding, hidden, embedding_weight, hidden_weight, out):
            with T.Kernel(r, threads=256) as b:
                values = T.alloc_fragment((2, c), 'float32')
                squares = T.alloc_fragment((2, c), 'float32')
                sums = T.alloc_fragment((2,), 'float32')
                for i, j in T.Parallel(2, c):
                    if i == 0:
                        values[i, j] = embedding[b, j]
                    else:
                        values[i, j] = hidden[b, j]
                    squares[i, j] = values[i, j] * values[i, j]
                T.reduce_sum(squares, sums, dim=1)
                for i, j in T.Parallel(2, c):
                    weight = T.alloc_var('float32')
                    if i == 0:
                        weight = embedding_weight[j]
                    else:
                        weight = hidden_weight[j]
                    out[b, i*c+j] = values[i, j] * T.rsqrt(sums[i]/c+eps) * (1+weight)
        return primitive([('embedding', (r, c), 'bfloat16'), ('hidden', (r, c), 'bfloat16'),
            ('embedding_weight', (c,), 'bfloat16'), ('hidden_weight', (c,), 'bfloat16'),
            ('out', (r, 2*c), 'bfloat16')], algorithm)
    if kind == 'mtp_cast':
        @T.macro
        def algorithm(x, out):
            with T.Kernel(r, T.ceildiv(c, 256), threads=256) as (b, tile):
                for j in T.Parallel(256):
                    if tile*256+j < c:
                        out[b, tile*256+j] = x[b, tile*256+j]
        return primitive([('x', (r, c), 'float32'), ('out', (r, c), 'bfloat16')], algorithm)
    raise ValueError('unsupported MTP adapter')
