"""Bounded Hopper query tiles for long speculative verification windows."""

def partial(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r, chunk, cap, splits, d = p['slots'],p['chunk'],p['capacity'],p.get('splits',16),256
    rows=r*chunk
    packed=p.get('packed_loads',False)
    decoded=p.get('decoded_kv',False)
    tile_rows=p.get('key_rows',64)
    query_tokens=min(chunk,p.get('query_tokens',8))
    query_rows=max(16,query_tokens*8)
    query_tiles=T.ceildiv(chunk,query_tokens)
    threads=p.get('threads',128 if query_rows<=64 else 256)
    policy=T.GemmWarpPolicy.FullRow if query_rows>=128 else T.GemmWarpPolicy.Square
    if packed and cap%tile_rows:raise ValueError('packed KV requires capacity divisible by the key tile')
    arguments=[('q',(rows,16,d),'bfloat16'),
        ('kc',(r,2,cap,d),'bfloat16' if decoded else 'uint8'),
        ('vc',(r,2,cap,d),'bfloat16' if decoded else 'uint8')]
    if not decoded:
        arguments += [('ks',(r,2,cap,2),'float32'),('vs',(r,2,cap,2),'float32')]
    arguments += [('positions',(r,),'int32'),('lengths',(r,),'int32'),
        ('out',(r,2,splits,chunk*8,d),'float32'),('stats',(r,2,splits,chunk*8,2),'float32')]
    @T.macro
    def algorithm(q,kc,vc,ks,vs,positions,lengths,out,stats):
        with T.Kernel(r, 2, splits*query_tiles, threads=threads) as (slot, head, group):
            part=group//query_tiles;query_tile=group%query_tiles
            query = T.alloc_shared((query_rows, d), 'bfloat16')
            key = T.alloc_shared((tile_rows, d), 'bfloat16')
            value = T.alloc_shared((tile_rows, d), 'bfloat16')
            prob = T.alloc_shared((query_rows, tile_rows), 'bfloat16')
            if packed and not decoded:
                encoded=T.alloc_shared((tile_rows,d),'float8_e4m3fn')
                scales=T.alloc_shared((tile_rows,2),'float32')
            scores = T.alloc_fragment((query_rows, tile_rows), 'float32')
            result = T.alloc_fragment((query_rows, d), 'float32')
            maximum = T.alloc_fragment((query_rows,), 'float32')
            previous = T.alloc_fragment((query_rows,), 'float32')
            factor = T.alloc_fragment((query_rows,), 'float32')
            normalizer = T.alloc_fragment((query_rows,), 'float32')
            total = T.alloc_fragment((query_rows,), 'float32')
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity('float32'))
            for i, j in T.Parallel(query_rows, d):
                query[i, j] = T.if_then_else((query_tile*query_tokens+i//8) < lengths[slot], q[slot*chunk+T.min(query_tile*query_tokens+i//8,chunk-1), head*8+i%8,j], 0)
            count = positions[slot] + lengths[slot]
            tiles = T.ceildiv(T.ceildiv(count, tile_rows), splits)
            if (lengths[slot] > query_tile*query_tokens) & (part * tiles * tile_rows < count):
                for tile in T.serial(tiles):
                    if decoded:
                        start=(part*tiles+tile)*tile_rows
                        if start<cap:
                            T.copy(kc[slot,head,start:start+tile_rows,0:d],key)
                            T.copy(vc[slot,head,start:start+tile_rows,0:d],value)
                        else:
                            T.clear(key);T.clear(value)
                    elif packed:
                        start=(part*tiles+tile)*tile_rows
                        if start<cap:
                            keys=T.view(kc,dtype='float8_e4m3fn')
                            values=T.view(vc,dtype='float8_e4m3fn')
                            T.copy(keys[slot,head,start:start+tile_rows,0:d],encoded)
                            T.copy(ks[slot,head,start:start+tile_rows,0:2],scales)
                            for i,j in T.Parallel(tile_rows,d):
                                key[i,j]=T.if_then_else(start+i<count,T.cast(encoded[i,j],'float32')*scales[i,j//128],0)
                            T.copy(values[slot,head,start:start+tile_rows,0:d],encoded)
                            T.copy(vs[slot,head,start:start+tile_rows,0:2],scales)
                            for i,j in T.Parallel(tile_rows,d):
                                value[i,j]=T.if_then_else(start+i<count,T.cast(encoded[i,j],'float32')*scales[i,j//128],0)
                        else:
                            T.clear(key);T.clear(value)
                    else:
                        for i, j in T.Parallel(tile_rows, d):
                            pos = (part * tiles + tile) * tile_rows + i
                            key[i, j] = T.if_then_else(pos < count, T.call_extern('float32', 'tensor_decode_e4m3', T.cast(kc[slot, head, T.min(pos, cap - 1), j], 'uint32')) * ks[slot, head, T.min(pos, cap - 1), j // 128], 0)
                            value[i, j] = T.if_then_else(pos < count, T.call_extern('float32', 'tensor_decode_e4m3', T.cast(vc[slot, head, T.min(pos, cap - 1), j], 'uint32')) * vs[slot, head, T.min(pos, cap - 1), j // 128], 0)
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True,policy=policy)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.if_then_else(((part*tiles+tile)*tile_rows+j < count)&((part*tiles+tile)*tile_rows+j <= positions[slot]+query_tile*query_tokens+i//8),
                            scores[i, j] * d ** -.5, -T.infinity('float32'))
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(query_rows):
                        factor[i] = T.if_then_else(previous[i]>-T.infinity('float32'),T.exp(previous[i]-maximum[i]),0)
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i,j] = T.if_then_else(scores[i,j]>-T.infinity('float32'),T.exp(scores[i,j]-maximum[i]),0)
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(query_rows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(query_rows, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result,policy=policy)
            for i, j in T.Parallel(query_tokens*8,d):
                if query_tile*query_tokens*8+i<chunk*8:
                    out[slot, head, part, query_tile*query_tokens*8+i, j] = result[i, j]
            for i in T.Parallel(query_tokens*8):
                if query_tile*query_tokens*8+i<chunk*8:
                    stats[slot, head, part, query_tile*query_tokens*8+i, 0] = maximum[i]
                    stats[slot, head, part, query_tile*query_tokens*8+i, 1] = normalizer[i]
    if decoded:
        @T.macro
        def decoded_algorithm(q,kc,vc,positions,lengths,out,stats):
            algorithm(q,kc,vc,kc,vc,positions,lengths,out,stats)
        return primitive(arguments,decoded_algorithm)
    return primitive(arguments,algorithm)
