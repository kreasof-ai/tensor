"""Native SM89 FP8 tensor-core projections with block-128 dynamic scales."""


def make_kernel(kind,p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r,k,o=p['r'],p['k'],p['o']
    n=p.get('columns',64)
    m=p.get('block_m',16)
    threads=p.get('threads',128)
    grouped=kind.startswith('fp8_experts')
    prequantized=kind.endswith('_prequantized')
    top,e=p.get('top',8),p.get('experts',256)
    routed=p.get('routed_input',False)
    parts=p.get('partitions',1)
    stages=p.get('stages',1)
    packed_copy=p.get('packed_copy',False)
    if ((grouped and r>16) or k%(128*parts) or o%n
            or m not in (m,32,64,128) or (grouped and m!=16)
            or n not in (32,64,128) or threads not in (128,256)):
        raise ValueError('invalid FP8 tensor-core projection schedule')
    arguments=[('x',(r,top,k) if grouped and routed else (r,k),'uint8' if prequantized else 'bfloat16'),
        ('w',(e,o,k) if grouped else (o,k),'uint8'),
        ('scales',(e,T.ceildiv(o,128),k//128) if grouped else (T.ceildiv(o,128),k//128),'bfloat16')]
    if prequantized:
        arguments += [('activation_scales',(r,top,k//128) if grouped and routed else (r,k//128),'float32')]
    if grouped:
        arguments += [('experts',(r*top,),'int32'),('routes',(r*top,r),'int32')]
    output_shape=((r,top,parts,o) if grouped else (r,parts,o)) if parts>1 else ((r,top,o) if grouped else (r,o))
    arguments += [('out',output_shape,'float32')]

    @T.macro
    def body(x,w,scales,input_scales,experts,routes,out,bx,group,expert,rowbase,part):
        lhs=T.alloc_shared((m,128),'float8_e4m3fn')
        rhs=T.alloc_shared((n,128),'float8_e4m3fn')
        absolute=T.alloc_fragment((m,128),'float32')
        maxima=T.alloc_fragment((m,),'float32')
        activation_scales=T.alloc_shared((m,),'float32')
        block=T.alloc_fragment((m,n),'float32')
        total=T.alloc_fragment((m,n),'float32')
        T.clear(total)
        for local_tile in T.Pipelined(k//(128*parts),num_stages=stages):
            tile=part*(k//(128*parts))+local_tile
            if prequantized:
                for b in T.Parallel(m):
                    activation_scales[b]=1
                    if rowbase+b<r:
                        if grouped:
                            if routes[group,b]>=0:
                                if routed:activation_scales[b]=input_scales[b,routes[group,b],tile]
                                else:activation_scales[b]=input_scales[b,tile]
                        else:activation_scales[b]=input_scales[rowbase+b,tile]
                T.sync_threads()
                for b,j in T.Parallel(m,128):
                    bits=T.alloc_var('uint8')
                    bits=0
                    if rowbase+b<r:
                        if grouped:
                            if routes[group,b]>=0:
                                if routed:bits=x[b,routes[group,b],tile*128+j]
                                else:bits=x[b,tile*128+j]
                        else:bits=x[rowbase+b,tile*128+j]
                    lhs[b,j]=T.reinterpret('float8_e4m3fn',bits)
            else:
                for b,j in T.Parallel(m,128):
                    if rowbase+b<r:
                        if grouped:
                            if routes[group,b]>=0:
                                if routed:absolute[b,j]=T.abs(T.cast(x[b,routes[group,b],tile*128+j],'float32'))
                                else:absolute[b,j]=T.abs(T.cast(x[b,tile*128+j],'float32'))
                            else:absolute[b,j]=0
                        else:absolute[b,j]=T.abs(T.cast(x[rowbase+b,tile*128+j],'float32'))
                    else:absolute[b,j]=0
                T.reduce_max(absolute,maxima,dim=1)
                for b in T.Parallel(m):
                    activation_scales[b]=T.max(maxima[b],T.float32(1e-12))/T.float32(448)
                T.sync_threads()
                for b,j in T.Parallel(m,128):
                    value=T.alloc_var('float32')
                    value=0
                    if rowbase+b<r:
                        if grouped:
                            if routes[group,b]>=0:
                                if routed:value=T.cast(x[b,routes[group,b],tile*128+j],'float32')
                                else:value=T.cast(x[b,tile*128+j],'float32')
                        else:value=T.cast(x[rowbase+b,tile*128+j],'float32')
                    bits=T.cast(T.call_extern('uint32','tensor_encode_e4m3',value/activation_scales[b]),'uint8')
                    lhs[b,j]=T.reinterpret('float8_e4m3fn',bits)
            if packed_copy:
                weights=T.view(w,dtype='float8_e4m3fn')
                if grouped:T.copy(weights[expert,bx*n:bx*n+n,tile*128:tile*128+128],rhs)
                else:T.copy(weights[bx*n:bx*n+n,tile*128:tile*128+128],rhs)
            else:
                for i,j in T.Parallel(n,128):
                    if grouped:rhs[i,j]=T.reinterpret('float8_e4m3fn',w[expert,bx*n+i,tile*128+j])
                    else:rhs[i,j]=T.reinterpret('float8_e4m3fn',w[bx*n+i,tile*128+j])
            T.gemm(lhs,rhs,block,transpose_B=True,clear_accum=True)
            for b,j in T.Parallel(m,n):
                if grouped:
                    total[b,j]+=block[b,j]*activation_scales[b]*T.cast(scales[expert,(bx*n+j)//128,tile],'float32')
                else:
                    total[b,j]+=block[b,j]*activation_scales[b]*T.cast(scales[(bx*n+j)//128,tile],'float32')
        for b,j in T.Parallel(m,n):
            if rowbase+b<r:
                if grouped:
                    if routes[group,b]>=0:
                        if parts>1:out[b,routes[group,b],part,bx*n+j]=total[b,j]
                        else:out[b,routes[group,b],bx*n+j]=total[b,j]
                else:
                    if parts>1:out[rowbase+b,part,bx*n+j]=total[b,j]
                    else:out[rowbase+b,bx*n+j]=total[b,j]

    @T.macro
    def dense(x,w,scales,out):
        with T.Kernel(o//n,T.ceildiv(r,m),parts,threads=threads) as (bx,by,part):
            body(x,w,scales,None,None,None,out,bx,0,0,by*m,part)

    @T.macro
    def dense_quantized(x,w,scales,input_scales,out):
        with T.Kernel(o//n,T.ceildiv(r,m),parts,threads=threads) as (bx,by,part):
            body(x,w,scales,input_scales,None,None,out,bx,0,0,by*m,part)

    @T.macro
    def experts(x,w,scales,expert_ids,routes,out):
        with T.Kernel(o//n,r*top,parts,threads=threads) as (bx,group,part):
            expert=expert_ids[group]
            if expert>=0:body(x,w,scales,None,expert_ids,routes,out,bx,group,expert,0,part)

    @T.macro
    def experts_quantized(x,w,scales,input_scales,expert_ids,routes,out):
        with T.Kernel(o//n,r*top,parts,threads=threads) as (bx,group,part):
            expert=expert_ids[group]
            if expert>=0:body(x,w,scales,input_scales,expert_ids,routes,out,bx,group,expert,0,part)

    return primitive(arguments,(experts_quantized if grouped else dense_quantized)
                     if prequantized else (experts if grouped else dense))


def quantize_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r,k,top=p['r'],p['k'],p.get('top')
    rows=r*(top or 1)
    shape=(r,top,k) if top else (r,k)
    scales_shape=(r,top,k//128) if top else (r,k//128)
    @T.macro
    def algorithm(x,out,scales):
        with T.Kernel(rows,k//128,threads=128) as (b,tile):
            values=T.alloc_fragment((128,),'float32')
            absolute=T.alloc_fragment((128,),'float32')
            maximum=T.alloc_fragment((1,),'float32')
            for j in T.Parallel(128):
                if top:values[j]=T.cast(x[b//top,b%top,tile*128+j],'float32')
                else:values[j]=T.cast(x[b,tile*128+j],'float32')
                absolute[j]=T.abs(values[j])
            T.reduce_max(absolute,maximum,dim=0)
            for j in T.Parallel(128):
                scale=T.max(maximum[0],T.float32(1e-12))/T.float32(448)
                bits=T.cast(T.call_extern('uint32','tensor_encode_e4m3',values[j]/scale),'uint8')
                if top:out[b//top,b%top,tile*128+j]=bits
                else:out[b,tile*128+j]=bits
            for j in T.Parallel(1):
                scale=T.max(maximum[0],T.float32(1e-12))/T.float32(448)
                if top:scales[b//top,b%top,tile]=scale
                else:scales[b,tile]=scale
    return primitive([('x',shape,'bfloat16'),('out',shape,'uint8'),('scales',scales_shape,'float32')],algorithm)


def merge_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r,o,parts=p['r'],p['o'],p['partitions']
    top=p.get('top')
    rows=r*(top or 1)
    arguments=[('partial',(r,top,parts,o) if top else (r,parts,o),'float32'),
               ('out',(r,top,o) if top else (r,o),'float32')]
    @T.macro
    def algorithm(partial,out):
        with T.Kernel(rows,T.ceildiv(o,256),threads=256) as (b,tile):
            for j in T.Parallel(256):
                col=tile*256+j
                if col<o:
                    total=T.alloc_var('float32')
                    total=0
                    for part in T.serial(parts):
                        if top:total+=partial[b//top,b%top,part,col]
                        else:total+=partial[b,part,col]
                    if top:out[b//top,b%top,col]=total
                    else:out[b,col]=total
    return primitive(arguments,algorithm)


def bf16_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r,k,o=p['r'],p['k'],p['o']
    m,n,depth=16,32,64
    @T.macro
    def algorithm(x,w,out):
        with T.Kernel(T.ceildiv(o,n),T.ceildiv(r,m),threads=128) as (bx,by):
            lhs=T.alloc_shared((m,depth),'bfloat16');rhs=T.alloc_shared((n,depth),'bfloat16')
            accum=T.alloc_fragment((m,n),'float32');T.clear(accum)
            for tile in T.Pipelined(k//depth,num_stages=2):
                for i,j in T.Parallel(m,depth):lhs[i,j]=T.if_then_else(by*m+i<r,x[T.min(by*m+i,r-1),tile*depth+j],0)
                for i,j in T.Parallel(n,depth):rhs[i,j]=T.if_then_else(bx*n+i<o,w[T.min(bx*n+i,o-1),tile*depth+j],0)
                T.gemm(lhs,rhs,accum,transpose_B=True)
            for i,j in T.Parallel(m,n):
                if (by*m+i<r)&(bx*n+j<o):out[by*m+i,bx*n+j]=accum[i,j]
    return primitive([('x',(r,k),'bfloat16'),('w',(o,k),'bfloat16'),('out',(r,o),'float32')],algorithm)


def bf16_decode_kernel(p):
    """Parallel full-depth reductions for the small unquantized projections."""
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r,k,o=p['r'],p['k'],p['o']
    @T.macro
    def algorithm(x,w,out):
        with T.Kernel(r,o,threads=256) as (row,column):
            products=T.alloc_fragment((k,),'float32')
            total=T.alloc_fragment((1,),'float32')
            for j in T.Parallel(k):products[j]=T.cast(x[row,j],'float32')*T.cast(w[column,j],'float32')
            T.reduce_sum(products,total,dim=0)
            for j in T.Parallel(1):out[row,column]=total[0]
    return primitive([('x',(r,k),'bfloat16'),('w',(o,k),'bfloat16'),('out',(r,o),'float32')],algorithm)


def bf16_head_kernel(p):
    """Vector-loaded BF16 tensor-core projection for the large vocabulary head."""
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r,k,o=p['r'],p['k'],p['o'];m=16
    n,depth,threads=p.get('columns',64),p.get('depth',128),p.get('threads',128)
    parts=p.get('partitions',1)
    if o%n or k%(depth*parts):raise ValueError('invalid BF16 projection partition')
    @T.macro
    def algorithm(x,w,out):
        with T.Kernel(o//n,T.ceildiv(r,m),parts,threads=threads) as (bx,by,part):
            lhs=T.alloc_shared((m,depth),'bfloat16');rhs=T.alloc_shared((n,depth),'bfloat16')
            accum=T.alloc_fragment((m,n),'float32');T.clear(accum)
            for tile in T.Pipelined(k//(depth*parts),num_stages=2):
                start=(part*(k//(depth*parts))+tile)*depth
                for i,j in T.Parallel(m,depth):
                    lhs[i,j]=T.if_then_else(by*m+i<r,x[T.min(by*m+i,r-1),start+j],0)
                T.copy(w[bx*n:bx*n+n,start:start+depth],rhs)
                T.gemm(lhs,rhs,accum,transpose_B=True)
            for i,j in T.Parallel(m,n):
                if by*m+i<r:
                    if parts>1:out[by*m+i,part,bx*n+j]=accum[i,j]
                    else:out[by*m+i,bx*n+j]=accum[i,j]
    shape=(r,parts,o) if parts>1 else (r,o)
    return primitive([('x',(r,k),'bfloat16'),('w',(o,k),'bfloat16'),('out',shape,'float32')],algorithm)
