"""Experimental decoded attention: two 16-key updates share one QK MMA."""

def attention(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    slots,chunk=p['slots'],p['chunk'];rows=slots*chunk
    d,cap=256,p['capacity']
    qrows,kvrows,stride=p.get('query_rows',128),32,16
    threads=p.get('threads',256)
    joint=p.get('joint_kv',False)
    decoded=p.get('decoded_kv',False)
    if not decoded:raise ValueError('batched keys require decoded KV')
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
                batched_scores=T.alloc_fragment((qrows,kvrows),'float32')
                batched_values=T.alloc_shared((kvrows,vd),'bfloat16')
                staged_scores=T.alloc_shared((qrows,kvrows),'float32')
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
                for block in T.serial(T.ceildiv(count,32)):
                    start=block*32
                    T.copy(kc[slot,head,start:start+32,0:d],key)
                    T.copy(vc[slot,head,start:start+32,value_part*vd:value_part*vd+vd],batched_values)
                    T.sync_threads()
                    T.gemm(query,key,batched_scores,transpose_B=True,clear_accum=True,policy=policy)
                    T.copy(batched_scores,staged_scores);T.sync_threads()
                    for sub in T.unroll(2):
                        step=block*2+sub
                        if step*stride<count:
                            T.copy(batched_values[sub*16:sub*16+16,0:vd],value[0:16,0:vd])
                            for i,j in T.Parallel(qrows,kvrows):
                                scores[i,j]=T.if_then_else(j<16,staged_scores[i,sub*16+j%16],-T.infinity('float32'))
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
                            T.copy(scores,prob);T.sync_threads();T.gemm(prob,value,result,policy=policy)
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
