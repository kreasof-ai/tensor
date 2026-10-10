"""Hopper attention with padded MMA operands and 16-key softmax updates."""

def attention(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    slots,chunk=p['slots'],p['chunk'];rows=slots*chunk
    d,cap=256,p['capacity']
    qrows,kvrows,stride=p.get('query_rows',128),32,16
    threads=p.get('threads',256)
    joint=p.get('joint_kv',False)
    decoded=p.get('decoded_kv',False)
    warpgroup_qk=p.get('warpgroup_qk',False)
    warpgroup_pv=p.get('warpgroup_pv',False)
    if (warpgroup_qk or warpgroup_pv) and (not decoded or qrows!=128 or threads!=256):
        raise ValueError('warp-group attention requires the decoded 128-row, 256-thread profile')
    value_splits=p.get('value_splits',1)
    if value_splits not in (1,2) or (value_splits!=1 and not decoded):
        raise ValueError('split value dimensions require decoded KV')
    vd=d//value_splits
    cache_dtype='bfloat16' if decoded else 'uint8'
    if qrows not in (64,128,256) or threads not in (128,256,512):
        raise ValueError('unsupported Hopper attention geometry')
    token_rows=qrows//8;packed=p.get('packed_loads',False)
    policy=T.GemmWarpPolicy.FullRow if qrows>=128 else T.GemmWarpPolicy.Square
    if p.get('square_warps',False):policy=T.GemmWarpPolicy.Square
    if packed and cap%kvrows:raise ValueError('packed KV requires capacity divisible by key tile')
    @T.macro
    def algorithm(q,kc,vc,ks,vs,projection,positions,lengths,out):
        with T.Kernel(slots,2,T.ceildiv(chunk,token_rows)*value_splits,threads=threads) as (slot,head,group):
            tile=group//value_splits;value_part=group%value_splits
            if tile*token_rows<lengths[slot]:
                query=T.alloc_shared((qrows,d),'bfloat16')
                key=T.alloc_shared((kvrows,d),'bfloat16');value=T.alloc_shared((kvrows,vd),'bfloat16')
                prob=T.alloc_shared((qrows,kvrows),'bfloat16')
                if packed and not decoded:
                    encoded=T.alloc_shared((16,d),'float8_e4m3fn')
                    scales=T.alloc_shared((16,2),'float32')
                    if joint:
                        value_encoded=T.alloc_shared((16,d),'float8_e4m3fn')
                        value_scales=T.alloc_shared((16,2),'float32')
                scores=T.alloc_fragment((qrows,kvrows),'float32')
                result=T.alloc_fragment((qrows,vd),'float32')
                maximum=T.alloc_fragment((qrows,),'float32');previous=T.alloc_fragment((qrows,),'float32')
                factor=T.alloc_fragment((qrows,),'float32');normalizer=T.alloc_fragment((qrows,),'float32')
                total=T.alloc_fragment((qrows,),'float32')
                T.clear(result);T.clear(normalizer);T.fill(maximum,-T.infinity('float32'))
                if joint or decoded:
                    # The padded half never participates in a softmax update.
                    # Initialize it once instead of rewriting it for every KV block.
                    for i,j in T.Parallel(16,d):
                        key[i+16,j]=0
                    for i,j in T.Parallel(16,vd):value[i+16,j]=0
                for i,j in T.Parallel(qrows,d):
                    local=tile*token_rows+i//8
                    query[i,j]=T.if_then_else(local<lengths[slot],q[slot*chunk+T.min(local,chunk-1),head*8+i%8,j],0)
                # Causality bounds the read to the last query in this tile.
                count=positions[slot]+T.min(lengths[slot],tile*token_rows+token_rows)
                for step in T.serial(T.ceildiv(count,stride)):
                    if decoded:
                        start=step*stride
                        T.copy(kc[slot,head,start:start+16,0:d],key[0:16,0:d])
                        T.copy(vc[slot,head,start:start+16,value_part*vd:value_part*vd+vd],value[0:16,0:vd])
                    elif packed:
                        start=step*stride
                        keys=T.view(kc,dtype='float8_e4m3fn');values=T.view(vc,dtype='float8_e4m3fn')
                        T.copy(keys[slot,head,start:start+16,0:d],encoded)
                        T.copy(ks[slot,head,start:start+16,0:2],scales)
                        if joint:
                            T.copy(values[slot,head,start:start+16,0:d],value_encoded)
                            T.copy(vs[slot,head,start:start+16,0:2],value_scales)
                            T.sync_threads()
                            for i,j in T.Parallel(16,d):
                                key[i,j]=0;value[i,j]=0
                                if start+i<count:
                                    key[i,j]=T.cast(encoded[i,j],'float32')*scales[i,j//128]
                                    value[i,j]=T.cast(value_encoded[i,j],'float32')*value_scales[i,j//128]
                        else:
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
                    if warpgroup_qk:
                        T.wgmma_gemm(query,key,scores,transpose_B=True,clear_accum=True,policy=policy)
                        T.wait_wgmma(0)
                    else:T.gemm(query,key,scores,transpose_B=True,clear_accum=True,policy=policy)
                    T.copy(maximum,previous)
                    for i,j in T.Parallel(qrows,kvrows):
                        valid=(j<stride)&(step*stride+j<=positions[slot]+tile*token_rows+i//8)&(step*stride+j<count)
                        scores[i,j]=T.if_then_else(valid,scores[i,j]/16,-T.infinity('float32'))
                    T.reduce_max(scores,maximum,dim=1,clear=False)
                    for i in T.Parallel(qrows):factor[i]=T.exp(previous[i]-maximum[i])
                    for i,j in T.Parallel(qrows,kvrows):scores[i,j]=T.exp(scores[i,j]-maximum[i])
                    T.reduce_sum(scores,total,dim=1)
                    for i in T.Parallel(qrows):normalizer[i]=normalizer[i]*factor[i]+total[i]
                    for i,j in T.Parallel(qrows,vd):result[i,j]*=factor[i]
                    T.copy(scores,prob);T.sync_threads()
                    if warpgroup_pv:
                        T.wgmma_gemm(prob,value,result,policy=policy)
                        T.wait_wgmma(0)
                    else:T.gemm(prob,value,result,policy=policy)
                for i,j in T.Parallel(qrows,vd):
                    local=tile*token_rows+i//8
                    if local<lengths[slot]:
                        row=slot*chunk+local
                        gate=T.cast(T.cast(projection[row,(head*8+i%8)*512+d+value_part*vd+j],'bfloat16'),'float32')
                        normalized=T.cast(T.cast(result[i,j]/normalizer[i],'bfloat16'),'float32')
                        out[row,(head*8+i%8)*d+value_part*vd+j]=normalized/(1+T.exp(-gate))
    arguments=[('q',(rows,16,d),'bfloat16'),('kc',(slots,2,cap,d),cache_dtype),
               ('vc',(slots,2,cap,d),cache_dtype)]
    tail=[('projection',(rows,8192),'float32'),('positions',(slots,),'int32'),
          ('lengths',(slots,),'int32'),('out',(rows,4096),'bfloat16')]
    if decoded:
        @T.macro
        def decoded_algorithm(q,kc,vc,projection,positions,lengths,out):
            algorithm(q,kc,vc,None,None,projection,positions,lengths,out)
        return primitive(arguments+tail,decoded_algorithm)
    return primitive(arguments+[('ks',(slots,2,cap,2),'float32'),('vs',(slots,2,cap,2),'float32')]+tail,algorithm)
