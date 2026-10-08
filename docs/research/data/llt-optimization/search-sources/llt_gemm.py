"""Coalesced projection schedules with explicit search knobs and FP32 sums."""


def make_kernel(p):
    import tilelang.language as T

    m,k,n=p['m'],p['k'],p['c']
    ta,tb,dt,outdt=p['ta'],p['tb'],p['dtype'],p['out_dtype']
    schedule=p['schedule']
    if schedule['family']=='gemv':
        if ta or not tb:
            raise ValueError('GEMV requires row-major linear weights and no A transpose')
        rows,chunk,threads=schedule['rows'],schedule['chunk'],schedule['threads']

        @T.prim_func
        def kernel(x:T.Tensor((m,k),dt),w:T.Tensor((n,k),dt),out:T.Tensor((m,n),outdt)):
            with T.Kernel(T.ceildiv(n,rows),m,threads=threads) as (block,row):
                products=T.alloc_fragment((rows,chunk),'float32')
                sums=T.alloc_fragment((rows,),'float32')
                acc=T.alloc_fragment((rows,),'float32')
                T.clear(acc)
                for tile in T.serial(T.ceildiv(k,chunk)):
                    for i,j in T.Parallel(rows,chunk):
                        products[i,j]=T.if_then_else(
                            (block*rows+i<n)&(tile*chunk+j<k),
                            T.cast(x[row,tile*chunk+j],'float32')*T.cast(w[block*rows+i,tile*chunk+j],'float32'),0)
                    T.reduce_sum(products,sums,dim=1,clear=True)
                    for i in T.Parallel(rows):acc[i]+=sums[i]
                for i in T.Parallel(rows):
                    if block*rows+i<n:out[row,block*rows+i]=acc[i]
        return kernel

    bm,bn,bk,threads,stages=(schedule[name] for name in ('bm','bn','bk','threads','stages'))

    @T.prim_func
    def kernel(x:T.Tensor((k,m) if ta else (m,k),dt),
               w:T.Tensor((n,k) if tb else (k,n),dt),out:T.Tensor((m,n),outdt)):
        with T.Kernel(T.ceildiv(n,bn),T.ceildiv(m,bm),threads=threads) as (bx,by):
            # Shared storage follows global row-major layout. Transpose at MMA,
            # rather than striding global weight loads across output channels.
            lhs=T.alloc_shared((bk,bm) if ta else (bm,bk),dt)
            rhs=T.alloc_shared((bn,bk) if tb else (bk,bn),dt)
            acc=T.alloc_fragment((bm,bn),'float32')
            T.clear(acc)
            for tile in T.Pipelined(T.ceildiv(k,bk),num_stages=stages):
                if ta:
                    T.copy(x[tile*bk:(tile+1)*bk,by*bm:(by+1)*bm],lhs)
                else:
                    T.copy(x[by*bm:(by+1)*bm,tile*bk:(tile+1)*bk],lhs)
                if tb:
                    T.copy(w[bx*bn:(bx+1)*bn,tile*bk:(tile+1)*bk],rhs)
                else:
                    T.copy(w[tile*bk:(tile+1)*bk,bx*bn:(bx+1)*bn],rhs)
                T.gemm(lhs,rhs,acc,transpose_A=ta,transpose_B=tb)
            T.copy(acc,out[by*bm:(by+1)*bm,bx*bn:(bx+1)*bn])
    return kernel
