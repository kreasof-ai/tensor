"""Small-chunk routed FP8 Split-K with shared expert weights."""

def expert_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    rows,k,o=p['rows'],p['k'],p['o']
    n,top,experts=p.get('columns',64),8,256
    m,threads=p.get('block_m',16),p.get('threads',128)
    parts=p.get('partitions',4);stages=p.get('stages',2)
    routed=p.get('routed_input',False)
    arguments=[('x',(rows,top,k) if routed else (rows,k),'uint8'),
               ('activation_scales',(rows,top,k//128) if routed else (rows,k//128),'float32'),
               ('w',(experts,o,k),'uint8'),('scales',(experts,T.ceildiv(o,128),k//128),'bfloat16'),
               ('counts',(experts,),'int32'),('routes',(experts,rows),'int32'),('out',(rows,top,parts,o),'float32')]
    @T.macro
    def algorithm(x,activation_scales,w,scales,counts,routes,out):
        with T.Kernel(o//n,experts,T.ceildiv(rows,m)*parts,threads=threads) as (bx,expert,group):
            tokens=group//parts;part=group%parts
            if tokens*m<counts[expert]:
                lhs=T.alloc_shared((m,128),'float8_e4m3fn')
                rhs=T.alloc_shared((n,128),'float8_e4m3fn')
                scale=T.alloc_shared((m,),'float32')
                selected=T.alloc_shared((m,),'int32')
                block=T.alloc_fragment((m,n),'float32');total=T.alloc_fragment((m,n),'float32')
                for b in T.Parallel(m):
                    selected[b]=T.if_then_else(tokens*m+b<counts[expert],routes[expert,T.min(tokens*m+b,rows-1)],-1)
                T.sync_threads();T.clear(total)
                for local in T.Pipelined(k//(128*parts),num_stages=stages):
                    tile=part*(k//(128*parts))+local
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
                        total[b,j]+=block[b,j]*scale[b]*T.cast(scales[expert,(bx*n+j)//128,tile],'float32')
                for b,j in T.Parallel(m,n):
                    if selected[b]>=0:out[selected[b]//top,selected[b]%top,part,bx*n+j]=total[b,j]
    return primitive(arguments,algorithm)
