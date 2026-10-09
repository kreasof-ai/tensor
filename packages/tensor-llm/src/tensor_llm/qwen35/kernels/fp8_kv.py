"""Block-128 E4M3 KV storage, with FP32 scales and BF16 attention arithmetic.

The cache profile is explicitly separate from the BF16 reference. Recurrent
GDN state remains FP32. Factories run only while producing CUDA artifacts.
"""


def make_kernel(kind,p):
    if kind=='attention_kv':return write(p)
    if kind=='attention_partial':return partial(p)
    if kind=='attention':return attention(p)
    raise ValueError('unknown FP8 KV operation: '+kind)


def write(p):
    import tilelang.language as T
    slots=p.get('slots',p.get('r'));chunk=p.get('chunk',1);rows=slots*chunk
    d,cap,eps,theta=256,p['capacity'],p['eps'],p['theta']
    @T.prim_func
    def kernel(projection:T.Tensor((rows,512),'float32'),values:T.Tensor((rows,512),'float32'),
               w:T.Tensor((d,),'bfloat16'),positions:T.Tensor((rows,),'int32'),active:T.Tensor((rows,),'int32'),
               kc:T.Tensor((slots,2,cap,d),'uint8'),vc:T.Tensor((slots,2,cap,d),'uint8'),
               ks:T.Tensor((slots,2,cap,2),'float32'),vs:T.Tensor((slots,2,cap,2),'float32')):
        with T.Kernel(rows,2,threads=256) as (row,head):
            if active[row]!=0:
                squares=T.alloc_fragment((d,),'float32');total=T.alloc_fragment((1,),'float32')
                normal=T.alloc_shared((d,),'bfloat16')
                rotated=T.alloc_shared((d,),'bfloat16')
                ka=T.alloc_fragment((2,128),'float32');va=T.alloc_fragment((2,128),'float32')
                km=T.alloc_fragment((2,),'float32');vm=T.alloc_fragment((2,),'float32')
                scales_k=T.alloc_shared((2,),'float32');scales_v=T.alloc_shared((2,),'float32')
                for j in T.Parallel(d):
                    value=T.cast(T.cast(projection[row,head*d+j],'bfloat16'),'float32')
                    squares[j]=value*value
                T.reduce_sum(squares,total,dim=0)
                for j in T.Parallel(d):
                    value=T.cast(T.cast(projection[row,head*d+j],'bfloat16'),'float32')
                    normal[j]=value*T.rsqrt(total[0]/d+eps)*(1+T.cast(w[j],'float32'))
                for j in T.Parallel(d):
                    value=T.alloc_var('float32');value=T.cast(normal[j],'float32')
                    if j<64:
                        angle=positions[row]*T.exp(T.cast(-2*(j%32),'float32')/64*T.log(T.float32(theta)))
                        pair=T.cast(normal[(j+32)%64],'float32')
                        value=value*T.cos(angle)+T.if_then_else(j<32,-pair,pair)*T.sin(angle)
                    rotated[j]=value
                for block,j in T.Parallel(2,128):
                    ka[block,j]=T.abs(T.cast(rotated[block*128+j],'float32'))
                    va[block,j]=T.abs(T.cast(T.cast(values[row,head*d+block*128+j],'bfloat16'),'float32'))
                T.reduce_max(ka,km,dim=1);T.reduce_max(va,vm,dim=1)
                for block in T.Parallel(2):
                    scales_k[block]=T.max(km[block],T.float32(1e-12))/448
                    scales_v[block]=T.max(vm[block],T.float32(1e-12))/448
                    ks[row//chunk,head,positions[row],block]=scales_k[block]
                    vs[row//chunk,head,positions[row],block]=scales_v[block]
                T.sync_threads()
                for j in T.Parallel(d):
                    quantized_key=T.cast(rotated[j],'float32')/scales_k[j//128]
                    quantized_value=T.cast(T.cast(values[row,head*d+j],'bfloat16'),'float32')/scales_v[j//128]
                    kc[row//chunk,head,positions[row],j]=T.cast(T.call_extern('uint32','tensor_encode_e4m3',quantized_key),'uint8')
                    vc[row//chunk,head,positions[row],j]=T.cast(T.call_extern('uint32','tensor_encode_e4m3',quantized_value),'uint8')
    return kernel


def partial(p):
    import tilelang.language as T
    r, cap, splits, d = p['r'], p['capacity'], p.get('splits', 16), 256
    packed=p.get('packed_loads',False)
    tile_rows=p.get('key_rows',64)
    query_rows=32 if tile_rows==16 else 16
    if packed and cap%tile_rows:raise ValueError('packed KV requires capacity divisible by the key tile')
    @T.prim_func
    def kernel(q: T.Tensor((r, 16, d), 'bfloat16'),
               kc: T.Tensor((r, 2, cap, d), 'uint8'), vc: T.Tensor((r, 2, cap, d), 'uint8'),
               ks: T.Tensor((r, 2, cap, 2), 'float32'), vs: T.Tensor((r, 2, cap, 2), 'float32'),
               positions: T.Tensor((r,), 'int32'), active: T.Tensor((r,), 'int32'),
               out: T.Tensor((r, 2, splits, 8, d), 'float32'),
               stats: T.Tensor((r, 2, splits, 8, 2), 'float32')):
        with T.Kernel(r, 2, splits, threads=128) as (slot, head, part):
            query = T.alloc_shared((query_rows, d), 'bfloat16')
            key = T.alloc_shared((tile_rows, d), 'bfloat16')
            value = T.alloc_shared((tile_rows, d), 'bfloat16')
            prob = T.alloc_shared((query_rows, tile_rows), 'bfloat16')
            if packed:
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
                query[i, j] = T.if_then_else(i < 8, q[slot, head * 8 + i % 8, j], 0)
            count = positions[slot] + 1
            tiles = T.ceildiv(T.ceildiv(count, tile_rows), splits)
            if (active[slot] != 0) & (part * tiles * tile_rows < count):
                for tile in T.serial(tiles):
                    if packed:
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
                    T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
                    T.copy(maximum, previous)
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.if_then_else((part * tiles + tile) * tile_rows + j < count,
                            scores[i, j] * d ** -.5, -T.infinity('float32'))
                    T.reduce_max(scores, maximum, dim=1, clear=False)
                    for i in T.Parallel(query_rows):
                        factor[i] = T.exp(previous[i] - maximum[i])
                    for i, j in T.Parallel(query_rows, tile_rows):
                        scores[i, j] = T.exp(scores[i, j] - maximum[i])
                    T.reduce_sum(scores, total, dim=1)
                    for i in T.Parallel(query_rows):
                        normalizer[i] = normalizer[i] * factor[i] + total[i]
                    for i, j in T.Parallel(query_rows, d):
                        result[i, j] *= factor[i]
                    T.copy(scores, prob)
                    T.gemm(prob, value, result)
            for i, j in T.Parallel(8, d):
                out[slot, head, part, i, j] = result[i, j]
            for i in T.Parallel(8):
                stats[slot, head, part, i, 0] = maximum[i]
                stats[slot, head, part, i, 1] = normalizer[i]
    return kernel



def attention(p):
    import tilelang.language as T
    slots,chunk=p['slots'],p['chunk'];rows=slots*chunk
    d,cap=256,p['capacity']
    qrows,kvrows=p.get('query_rows',64),p.get('key_rows',32)
    token_rows=qrows//8;packed=p.get('packed_loads',False)
    policy=T.GemmWarpPolicy.FullRow if qrows>=128 else T.GemmWarpPolicy.Square
    if packed and cap%kvrows:raise ValueError('packed KV requires capacity divisible by key tile')
    @T.prim_func
    def kernel(q:T.Tensor((rows,16,d),'bfloat16'),kc:T.Tensor((slots,2,cap,d),'uint8'),
               vc:T.Tensor((slots,2,cap,d),'uint8'),
               ks:T.Tensor((slots,2,cap,2),'float32'),vs:T.Tensor((slots,2,cap,2),'float32'),projection:T.Tensor((rows,8192),'float32'),
               positions:T.Tensor((slots,),'int32'),lengths:T.Tensor((slots,),'int32'),
               out:T.Tensor((rows,4096),'bfloat16')):
        with T.Kernel(slots,2,T.ceildiv(chunk,token_rows),threads=256) as (slot,head,tile):
            if tile*token_rows<lengths[slot]:
                query=T.alloc_shared((qrows,d),'bfloat16')
                key=T.alloc_shared((kvrows,d),'bfloat16');value=T.alloc_shared((kvrows,d),'bfloat16')
                prob=T.alloc_shared((qrows,kvrows),'bfloat16')
                if packed:
                    encoded=T.alloc_shared((kvrows,d),'float8_e4m3fn')
                    scales=T.alloc_shared((kvrows,2),'float32')
                scores=T.alloc_fragment((qrows,kvrows),'float32')
                result=T.alloc_fragment((qrows,d),'float32')
                maximum=T.alloc_fragment((qrows,),'float32');previous=T.alloc_fragment((qrows,),'float32')
                factor=T.alloc_fragment((qrows,),'float32');normalizer=T.alloc_fragment((qrows,),'float32')
                total=T.alloc_fragment((qrows,),'float32')
                T.clear(result);T.clear(normalizer);T.fill(maximum,-T.infinity('float32'))
                for i,j in T.Parallel(qrows,d):
                    local=tile*token_rows+i//8
                    query[i,j]=T.if_then_else(local<lengths[slot],q[slot*chunk+T.min(local,chunk-1),head*8+i%8,j],0)
                # Causality bounds the read to the last query in this tile.
                count=positions[slot]+T.min(lengths[slot],tile*token_rows+token_rows)
                for step in T.serial(T.ceildiv(count,kvrows)):
                    if packed:
                        start=step*kvrows
                        keys=T.view(kc,dtype='float8_e4m3fn');values=T.view(vc,dtype='float8_e4m3fn')
                        T.copy(keys[slot,head,start:start+kvrows,0:d],encoded)
                        T.copy(ks[slot,head,start:start+kvrows,0:2],scales)
                        for i,j in T.Parallel(kvrows,d):
                            key[i,j]=T.if_then_else(start+i<count,T.cast(encoded[i,j],'float32')*scales[i,j//128],0)
                        T.copy(values[slot,head,start:start+kvrows,0:d],encoded)
                        T.copy(vs[slot,head,start:start+kvrows,0:2],scales)
                        for i,j in T.Parallel(kvrows,d):
                            value[i,j]=T.if_then_else(start+i<count,T.cast(encoded[i,j],'float32')*scales[i,j//128],0)
                    else:
                        for i,j in T.Parallel(kvrows,d):
                            index=step*kvrows+i
                            key[i,j]=T.if_then_else(index<count,T.call_extern('float32','tensor_decode_e4m3',T.cast(kc[slot,head,T.min(index,cap-1),j],'uint32'))*ks[slot,head,T.min(index,cap-1),j//128],0)
                            value[i,j]=T.if_then_else(index<count,T.call_extern('float32','tensor_decode_e4m3',T.cast(vc[slot,head,T.min(index,cap-1),j],'uint32'))*vs[slot,head,T.min(index,cap-1),j//128],0)
                    T.gemm(query,key,scores,transpose_B=True,clear_accum=True,policy=policy)
                    T.copy(maximum,previous)
                    for i,j in T.Parallel(qrows,kvrows):
                        valid=(step*kvrows+j<=positions[slot]+tile*token_rows+i//8)&(step*kvrows+j<count)
                        scores[i,j]=T.if_then_else(valid,scores[i,j]/16,-T.infinity('float32'))
                    T.reduce_max(scores,maximum,dim=1,clear=False)
                    for i in T.Parallel(qrows):factor[i]=T.exp(previous[i]-maximum[i])
                    for i,j in T.Parallel(qrows,kvrows):scores[i,j]=T.exp(scores[i,j]-maximum[i])
                    T.reduce_sum(scores,total,dim=1)
                    for i in T.Parallel(qrows):normalizer[i]=normalizer[i]*factor[i]+total[i]
                    for i,j in T.Parallel(qrows,d):result[i,j]*=factor[i]
                    T.copy(scores,prob);T.gemm(prob,value,result,policy=policy)
                for i,j in T.Parallel(qrows,d):
                    local=tile*token_rows+i//8
                    if local<lengths[slot]:
                        row=slot*chunk+local
                        gate=T.cast(T.cast(projection[row,(head*8+i%8)*512+d+j],'bfloat16'),'float32')
                        normalized=T.cast(T.cast(result[i,j]/normalizer[i],'bfloat16'),'float32')
                        out[row,(head*8+i%8)*d+j]=normalized/(1+T.exp(-gate))
    return kernel
