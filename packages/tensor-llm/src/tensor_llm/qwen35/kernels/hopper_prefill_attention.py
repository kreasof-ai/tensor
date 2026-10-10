"""Hopper attention with padded MMA operands and 16-key softmax updates."""

def attention(p):
    import tilelang.language as T
    slots,chunk=p['slots'],p['chunk'];rows=slots*chunk
    d,cap=256,p['capacity']
    qrows,kvrows,stride=128,32,16
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
                    encoded=T.alloc_shared((16,d),'float8_e4m3fn')
                    scales=T.alloc_shared((16,2),'float32')
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
                for step in T.serial(T.ceildiv(count,stride)):
                    if packed:
                        start=step*stride
                        keys=T.view(kc,dtype='float8_e4m3fn');values=T.view(vc,dtype='float8_e4m3fn')
                        T.copy(keys[slot,head,start:start+16,0:d],encoded)
                        T.copy(ks[slot,head,start:start+16,0:2],scales)
                        T.sync_threads()
                        for i,j in T.Parallel(16,d):
                            key[i,j]=0
                            if start+i<count:key[i,j]=T.cast(encoded[i,j],'float32')*scales[i,j//128]
                            key[i+16,j]=0
                        T.sync_threads()
                        T.copy(values[slot,head,start:start+16,0:d],encoded)
                        T.copy(vs[slot,head,start:start+16,0:2],scales)
                        T.sync_threads()
                        for i,j in T.Parallel(16,d):
                            value[i,j]=0
                            if start+i<count:value[i,j]=T.cast(encoded[i,j],'float32')*scales[i,j//128]
                            value[i+16,j]=0
                        T.sync_threads()
                    else:
                        for i,j in T.Parallel(kvrows,d):
                            index=step*stride+i
                            key[i,j]=T.if_then_else((i<stride)&(index<count),T.call_extern('float32','tensor_decode_e4m3',T.cast(kc[slot,head,T.min(index,cap-1),j],'uint32'))*ks[slot,head,T.min(index,cap-1),j//128],0)
                            value[i,j]=T.if_then_else((i<stride)&(index<count),T.call_extern('float32','tensor_decode_e4m3',T.cast(vc[slot,head,T.min(index,cap-1),j],'uint32'))*vs[slot,head,T.min(index,cap-1),j//128],0)
                    T.sync_threads()
                    T.gemm(query,key,scores,transpose_B=True,clear_accum=True,policy=policy)
                    T.copy(maximum,previous)
                    for i,j in T.Parallel(qrows,kvrows):
                        valid=(j<stride)&(step*stride+j<=positions[slot]+tile*token_rows+i//8)&(step*stride+j<count)
                        scores[i,j]=T.if_then_else(valid,scores[i,j]/16,-T.infinity('float32'))
                    T.reduce_max(scores,maximum,dim=1,clear=False)
                    for i in T.Parallel(qrows):factor[i]=T.exp(previous[i]-maximum[i])
                    for i,j in T.Parallel(qrows,kvrows):scores[i,j]=T.exp(scores[i,j]-maximum[i])
                    T.reduce_sum(scores,total,dim=1)
                    for i in T.Parallel(qrows):normalizer[i]=normalizer[i]*factor[i]+total[i]
                    for i,j in T.Parallel(qrows,d):result[i,j]*=factor[i]
                    T.copy(scores,prob);T.sync_threads();T.gemm(prob,value,result,policy=policy)
                for i,j in T.Parallel(qrows,d):
                    local=tile*token_rows+i//8
                    if local<lengths[slot]:
                        row=slot*chunk+local
                        gate=T.cast(T.cast(projection[row,(head*8+i%8)*512+d+j],'bfloat16'),'float32')
                        normalized=T.cast(T.cast(result[i,j]/normalizer[i],'bfloat16'),'float32')
                        out[row,(head*8+i%8)*d+j]=normalized/(1+T.exp(-gate))
    return kernel
