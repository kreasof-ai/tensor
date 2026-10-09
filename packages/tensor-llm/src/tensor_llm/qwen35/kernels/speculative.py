"""AOT factories for speculative verification and accepted-prefix state restore."""

def make_kernel(kind,p):
    import tilelang.language as T
    slots,chunk=p['slots'],p['chunk']
    rows=slots*chunk
    if kind=='gdn_conv':
        channels=8192
        @T.prim_func
        def kernel(x:T.Tensor((rows,channels),'float32'),w:T.Tensor((channels,1,4),'bfloat16'),
                   lengths:T.Tensor((slots,),'int32'),state:T.Tensor((slots,channels,3),'float32'),
                   out:T.Tensor((rows,channels),'float32'),
                   checkpoints:T.Tensor((slots,chunk-1,channels,3),'float32')):
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
                            if t<chunk-1:
                                checkpoints[slot,t,col,0]=h0
                                checkpoints[slot,t,col,1]=h1
                                checkpoints[slot,t,col,2]=h2
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
                   state:T.Tensor((slots,heads,value,key),'float32'),out:T.Tensor((rows,heads,value),'float32'),
                   checkpoints:T.Tensor((slots,chunk-1,heads,value,key),'float32')):
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
                        if t<chunk-1:
                            T.copy(matrix,checkpoints[slot,t,head,part*tile:part*tile+tile,:])
                        for i in T.Parallel(tile):out[row,head,part*tile+i]=result[i]
                    else:
                        for i in T.Parallel(tile):out[row,head,part*tile+i]=0
                T.copy(matrix,state[slot,head,part*tile:part*tile+tile,:])
        return kernel
    if kind=='restore':
        shape=tuple(p['shape']);width=1
        for size in shape:width*=size
        @T.prim_func
        def kernel(checkpoints:T.Tensor((slots,chunk-1,*shape),'float32'),
                   state:T.Tensor((slots,*shape),'float32'),
                   accepted:T.Tensor((slots,),'int32'),lengths:T.Tensor((slots,),'int32')):
            saved=T.view(checkpoints,shape=(slots,chunk-1,width))
            dest=T.view(state,shape=(slots,width))
            with T.Kernel(slots,T.ceildiv(width,256),threads=256) as (slot,tile):
                if (accepted[slot]>0)&(accepted[slot]<lengths[slot]):
                    for j in T.Parallel(256):
                        col=tile*256+j
                        if col<width:dest[slot,col]=saved[slot,accepted[slot]-1,col]
        return kernel
    raise ValueError('unknown speculative operation: '+kind)


def head_kernel(p):
    """Vector-loaded BF16 tensor-core projection for the large vocabulary head."""
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r,k,o=p['r'],p['k'],p['o'];m=p.get('block_m',32)
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
