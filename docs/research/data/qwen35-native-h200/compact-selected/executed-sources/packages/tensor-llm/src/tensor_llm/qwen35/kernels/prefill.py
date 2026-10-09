"""Sequence kernels for chunked native Qwen prefill, with persistent slot state."""


def make_kernel(kind,p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    if p.get('kv_dtype') == 'fp8':
        from .fp8_kv import make_kernel as fp8_kv
        return fp8_kv(kind,p)
    slots,chunk=p['slots'],p['chunk']
    rows=slots*chunk
    if kind=='controls':
        @T.prim_func
        def kernel(position:T.Tensor((slots,),'int32'),lengths:T.Tensor((slots,),'int32'),
                   flat_position:T.Tensor((rows,),'int32'),active:T.Tensor((rows,),'int32')):
            with T.Kernel(T.ceildiv(rows,256),threads=256) as tile:
                for j in T.Parallel(256):
                    row=tile*256+j
                    if row<rows:
                        flat_position[row]=position[row//chunk]+row%chunk
                        active[row]=T.cast(row%chunk<lengths[row//chunk],'int32')
        return kernel
    if kind=='gdn_conv':
        channels=8192
        @T.prim_func
        def kernel(x:T.Tensor((rows,channels),'float32'),w:T.Tensor((channels,1,4),'bfloat16'),
                   lengths:T.Tensor((slots,),'int32'),state:T.Tensor((slots,channels,3),'float32'),
                   out:T.Tensor((rows,channels),'float32')):
            with T.Kernel(slots,channels//256,threads=256) as (slot,tile):
                for j in T.Parallel(256):
                    col=tile*256+j
                    h0=T.alloc_var('float32');h1=T.alloc_var('float32');h2=T.alloc_var('float32')
                    h0=state[slot,col,0];h1=state[slot,col,1];h2=state[slot,col,2]
                    for t in T.serial(chunk):
                        if t<lengths[slot]:
                            current=T.cast(T.cast(x[slot*chunk+t,col],'bfloat16'),'float32')
                            value=current*T.cast(w[col,0,3],'float32')
                            value+=h0*T.cast(w[col,0,0],'float32')
                            value+=h1*T.cast(w[col,0,1],'float32')
                            value+=h2*T.cast(w[col,0,2],'float32')
                            out[slot*chunk+t,col]=T.cast(value/(1+T.exp(-value)),'bfloat16')
                            h0=h1;h1=h2;h2=current
                        else:out[slot*chunk+t,col]=0
                    state[slot,col,0]=h0;state[slot,col,1]=h1;state[slot,col,2]=h2
        return kernel
    if kind=='gdn_scan':
        heads,key,value=32,128,128
        tile=p.get('value_tile',32)
        @T.prim_func
        def kernel(q:T.Tensor((rows,heads,key),'float32'),k:T.Tensor((rows,heads,key),'float32'),
                   v:T.Tensor((rows,heads,value),'float32'),g:T.Tensor((rows,heads),'float32'),
                   beta:T.Tensor((rows,heads),'float32'),lengths:T.Tensor((slots,),'int32'),
                   state:T.Tensor((slots,heads,value,key),'float32'),out:T.Tensor((rows,heads,value),'float32')):
            with T.Kernel(slots,heads,value//tile,threads=128) as (slot,head,part):
                matrix=T.alloc_fragment((tile,key),'float32')
                prediction=T.alloc_fragment((tile,key),'float32')
                inner=T.alloc_fragment((tile,),'float32');result=T.alloc_fragment((tile,),'float32')
                qq=T.alloc_shared((key,),'float32');kk=T.alloc_shared((key,),'float32')
                T.copy(state[slot,head,part*tile:part*tile+tile,:],matrix)
                for t in T.serial(chunk):
                    row=slot*chunk+t
                    if t<lengths[slot]:
                        for j in T.Parallel(key):qq[j]=q[row,head,j];kk[j]=k[row,head,j]
                        for i,j in T.Parallel(tile,key):
                            matrix[i,j]*=T.exp(g[row,head])
                            prediction[i,j]=matrix[i,j]*kk[j]
                        T.reduce_sum(prediction,inner,dim=1)
                        for i,j in T.Parallel(tile,key):
                            matrix[i,j]+=kk[j]*beta[row,head]*(v[row,head,part*tile+i]-inner[i])
                            prediction[i,j]=matrix[i,j]*qq[j]
                        T.reduce_sum(prediction,result,dim=1)
                        for i in T.Parallel(tile):out[row,head,part*tile+i]=result[i]
                    else:
                        for i in T.Parallel(tile):out[row,head,part*tile+i]=0
                T.copy(matrix,state[slot,head,part*tile:part*tile+tile,:])
        return kernel
    if kind=='expert_routes':
        experts,top=256,8
        @T.prim_func
        def kernel(ids:T.Tensor((rows,top),'int32'),active:T.Tensor((rows,),'int32'),
                   counts:T.Tensor((experts,),'int32'),routes:T.Tensor((experts,rows),'int32')):
            with T.Kernel(experts,threads=256) as expert:
                tx=T.get_thread_binding();counter=T.alloc_shared((1,),'int32')
                if tx==0:counter[0]=0
                T.sync_threads()
                for tile in T.serial(T.ceildiv(rows,256)):
                    row=tile*256+tx
                    if row<rows:
                        if active[row]!=0:
                            for rank in T.unroll(top):
                                if ids[row,rank]==expert:
                                    index=T.atomic_add(counter[0],1,return_prev=True)
                                    routes[expert,index]=row*top+rank
                T.sync_threads()
                if tx==0:counts[expert]=counter[0]
        return kernel
    if kind=='last_rows':
        width=p['width']
        @T.prim_func
        def kernel(x:T.Tensor((rows,width),'bfloat16'),lengths:T.Tensor((slots,),'int32'),
                   out:T.Tensor((slots,width),'bfloat16')):
            with T.Kernel(slots,T.ceildiv(width,256),threads=256) as (slot,tile):
                for j in T.Parallel(256):
                    col=tile*256+j
                    if col<width:
                        out[slot,col]=T.if_then_else(lengths[slot]>0,x[slot*chunk+T.max(lengths[slot]-1,0),col],0)
        return kernel
    if kind=='advance':
        @T.prim_func
        def kernel(position:T.Tensor((slots,),'int32'),lengths:T.Tensor((slots,),'int32'),active:T.Tensor((slots,),'int32')):
            with T.Kernel(1,threads=32):
                for slot in T.Parallel(slots):
                    # Decode argmax already advances one position for each
                    # active slot; prefill accounts for the remaining rows.
                    active[slot]=T.cast(lengths[slot]>0,'int32')
                    position[slot]+=T.max(lengths[slot]-1,0)
        return kernel
    if kind=='attention_kv':
        d,cap,eps,theta=256,p['capacity'],p['eps'],p['theta']
        @T.prim_func
        def kernel(projection:T.Tensor((rows,512),'float32'),values:T.Tensor((rows,512),'float32'),
                   w:T.Tensor((d,),'bfloat16'),positions:T.Tensor((rows,),'int32'),active:T.Tensor((rows,),'int32'),
                   kc:T.Tensor((slots,2,cap,d),'bfloat16'),vc:T.Tensor((slots,2,cap,d),'bfloat16')):
            with T.Kernel(rows,2,threads=256) as (row,head):
                if active[row]!=0:
                    squares=T.alloc_fragment((d,),'float32');total=T.alloc_fragment((1,),'float32')
                    normal=T.alloc_shared((d,),'bfloat16')
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
                        kc[row//chunk,head,positions[row],j]=value
                        vc[row//chunk,head,positions[row],j]=T.cast(values[row,head*d+j],'bfloat16')
        return kernel
    if kind=='attention':
        d,cap=256,p['capacity']
        qrows,kvrows=64,32
        @T.prim_func
        def kernel(q:T.Tensor((rows,16,d),'bfloat16'),kc:T.Tensor((slots,2,cap,d),'bfloat16'),
                   vc:T.Tensor((slots,2,cap,d),'bfloat16'),projection:T.Tensor((rows,8192),'float32'),
                   positions:T.Tensor((slots,),'int32'),lengths:T.Tensor((slots,),'int32'),
                   out:T.Tensor((rows,4096),'bfloat16')):
            with T.Kernel(slots,2,T.ceildiv(chunk,8),threads=256) as (slot,head,tile):
                if tile*8<lengths[slot]:
                    query=T.alloc_shared((qrows,d),'bfloat16')
                    key=T.alloc_shared((kvrows,d),'bfloat16');value=T.alloc_shared((kvrows,d),'bfloat16')
                    prob=T.alloc_shared((qrows,kvrows),'bfloat16')
                    scores=T.alloc_fragment((qrows,kvrows),'float32')
                    result=T.alloc_fragment((qrows,d),'float32')
                    maximum=T.alloc_fragment((qrows,),'float32');previous=T.alloc_fragment((qrows,),'float32')
                    factor=T.alloc_fragment((qrows,),'float32');normalizer=T.alloc_fragment((qrows,),'float32')
                    total=T.alloc_fragment((qrows,),'float32')
                    T.clear(result);T.clear(normalizer);T.fill(maximum,-T.infinity('float32'))
                    for i,j in T.Parallel(qrows,d):
                        local=tile*8+i//8
                        query[i,j]=T.if_then_else(local<lengths[slot],q[slot*chunk+T.min(local,chunk-1),head*8+i%8,j],0)
                    # Causality bounds the read to the last query in this tile.
                    count=positions[slot]+T.min(lengths[slot],tile*8+8)
                    for step in T.serial(T.ceildiv(count,kvrows)):
                        for i,j in T.Parallel(kvrows,d):
                            index=step*kvrows+i
                            key[i,j]=T.if_then_else(index<count,kc[slot,head,T.min(index,cap-1),j],0)
                            value[i,j]=T.if_then_else(index<count,vc[slot,head,T.min(index,cap-1),j],0)
                        T.gemm(query,key,scores,transpose_B=True,clear_accum=True)
                        T.copy(maximum,previous)
                        for i,j in T.Parallel(qrows,kvrows):
                            valid=(step*kvrows+j<=positions[slot]+tile*8+i//8)&(step*kvrows+j<count)
                            scores[i,j]=T.if_then_else(valid,scores[i,j]/16,-T.infinity('float32'))
                        T.reduce_max(scores,maximum,dim=1,clear=False)
                        for i in T.Parallel(qrows):factor[i]=T.exp(previous[i]-maximum[i])
                        for i,j in T.Parallel(qrows,kvrows):scores[i,j]=T.exp(scores[i,j]-maximum[i])
                        T.reduce_sum(scores,total,dim=1)
                        for i in T.Parallel(qrows):normalizer[i]=normalizer[i]*factor[i]+total[i]
                        for i,j in T.Parallel(qrows,d):result[i,j]*=factor[i]
                        T.copy(scores,prob);T.gemm(prob,value,result)
                    for i,j in T.Parallel(qrows,d):
                        local=tile*8+i//8
                        if local<lengths[slot]:
                            row=slot*chunk+local
                            gate=T.cast(T.cast(projection[row,(head*8+i%8)*512+d+j],'bfloat16'),'float32')
                            normalized=T.cast(T.cast(result[i,j]/normalizer[i],'bfloat16'),'float32')
                            out[row,(head*8+i%8)*d+j]=normalized/(1+T.exp(-gate))
        return kernel
    raise ValueError('unknown Qwen prefill operation: '+kind)


def expert_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    rows,k,o=p['rows'],p['k'],p['o']
    n,top,experts=p.get('columns',64),8,256
    m,threads=p.get('block_m',16),p.get('threads',128)
    routed=p.get('routed_input',False)
    compact=p.get('compact',False)
    max_tiles=T.ceildiv(rows*top,m)+experts-1
    arguments=[('x',(rows,top,k) if routed else (rows,k),'uint8'),
               ('activation_scales',(rows,top,k//128) if routed else (rows,k//128),'float32'),
               ('w',(experts,o,k),'uint8'),('scales',(experts,T.ceildiv(o,128),k//128),'bfloat16'),
               ('counts',(experts,),'int32'),('routes',(experts,rows),'int32'),('out',(rows,top,o),'float32')]
    @T.macro
    def body(x,activation_scales,w,scales,counts,routes,out,bx,expert,tokens):
        if tokens*m<counts[expert]:
            lhs=T.alloc_shared((m,128),'float8_e4m3fn')
            rhs=T.alloc_shared((n,128),'float8_e4m3fn')
            scale=T.alloc_shared((m,),'float32')
            selected=T.alloc_shared((m,),'int32')
            block=T.alloc_fragment((m,n),'float32');total=T.alloc_fragment((m,n),'float32')
            for b in T.Parallel(m):
                selected[b]=T.if_then_else(tokens*m+b<counts[expert],routes[expert,T.min(tokens*m+b,rows-1)],-1)
            T.sync_threads();T.clear(total)
            for tile in T.serial(k//128):
                for b in T.Parallel(m):
                    scale[b]=1
                    if selected[b]>=0:
                        if routed:scale[b]=activation_scales[selected[b]//top,selected[b]%top,tile]
                        else:scale[b]=activation_scales[selected[b]//top,tile]
                T.sync_threads()
                for b,j in T.Parallel(m,128):
                    bits=T.alloc_var('uint8');bits=0
                    if selected[b]>=0:
                        if routed:bits=x[selected[b]//top,selected[b]%top,tile*128+j]
                        else:bits=x[selected[b]//top,tile*128+j]
                    lhs[b,j]=T.reinterpret('float8_e4m3fn',bits)
                weights=T.view(w,dtype='float8_e4m3fn')
                T.copy(weights[expert,bx*n:bx*n+n,tile*128:tile*128+128],rhs)
                T.gemm(lhs,rhs,block,transpose_B=True,clear_accum=True)
                for b,j in T.Parallel(m,n):
                    if compact:
                        # Preserve the control profile's rounded product and
                        # fused accumulation when indirection changes lowering.
                        product=T.call_extern('float32','__fmul_rn',block[b,j],scale[b])
                        total[b,j]=T.call_extern('float32','__fmaf_rn',product,
                            T.cast(scales[expert,(bx*n+j)//128,tile],'float32'),total[b,j])
                    else:
                        total[b,j]+=block[b,j]*scale[b]*T.cast(scales[expert,(bx*n+j)//128,tile],'float32')
            for b,j in T.Parallel(m,n):
                if selected[b]>=0:out[selected[b]//top,selected[b]%top,bx*n+j]=total[b,j]
    @T.macro
    def algorithm(x,activation_scales,w,scales,counts,routes,out):
        with T.Kernel(o//n,experts,T.ceildiv(rows,m),threads=threads) as (bx,expert,tokens):
            body(x,activation_scales,w,scales,counts,routes,out,bx,expert,tokens)

    @T.macro
    def compact_algorithm(x,activation_scales,w,scales,counts,routes,out,tile_experts,tile_offsets):
        with T.Kernel(o//n,max_tiles,threads=threads) as (bx,tile):
            expert=tile_experts[tile]
            if expert>=0:
                body(x,activation_scales,w,scales,counts,routes,out,bx,expert,tile_offsets[tile])

    if compact:
        arguments += [('tile_experts',(max_tiles,),'int32'),('tile_offsets',(max_tiles,),'int32')]
    return primitive(arguments,compact_algorithm if compact else algorithm)
