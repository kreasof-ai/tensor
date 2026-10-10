"""Packed FP8 storage with exact shared BF16 operands for Hopper projections."""


def make_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    r,k,o=p['r'],p['k'],p['o'];m=p.get('block_m',64);n=p.get('columns',128)
    parts=p.get('partitions',1);threads=p.get('threads',256)
    stages=p.get('stages',1)
    packed=p.get('packed_gather',True)
    reduction=p.get('mma_reduction',128)
    reorder=p.get('mma_reorder',False)
    asynchronous=p.get('async_mma',False)
    packed_widen=p.get('packed_widen',False)
    if packed_widen and (not reorder or asynchronous):raise ValueError('packed widening requires synchronous paired MMA')
    if asynchronous and not (reorder and reduction==32):raise ValueError('async MMA requires paired K32')
    if reorder and reduction!=32:raise ValueError('paired operands require K32')
    if reduction not in (32,128):raise ValueError('invalid Hopper MMA reduction')
    if k%(128*parts) or o%n:raise ValueError('invalid Hopper dense geometry')
    shape=(r,parts,o) if parts>1 else (r,o)
    arguments=[('x',(r,k),'uint8'),('w',(o,k),'uint8'),
        ('scales',(T.ceildiv(o,128),k//128),'bfloat16'),
        ('activation_scales',(r,k//128),'float32'),('out',shape,'float32')]
    @T.macro
    def algorithm(x,w,scales,activation_scales,out):
        with T.Kernel(o//n,T.ceildiv(r,m),parts,threads=threads) as (bx,by,part):
            if packed_widen:
                from tensor_llm.qwen35.kernels.fp8_operand import CUDA_SOURCE
                T.import_source(CUDA_SOURCE)
            lhs=T.alloc_shared((m,128),'float8_e4m3fn');rhs=T.alloc_shared((n,128),'float8_e4m3fn')
            if asynchronous:
                a_batch=T.alloc_shared((m,128),'float16');b_batch=T.alloc_shared((n,128),'float16')
                p1=T.alloc_fragment((m,n),'float32');p2=T.alloc_fragment((m,n),'float32');p3=T.alloc_fragment((m,n),'float32')
            if reduction==128:
                a=T.alloc_shared((m,128),'bfloat16');b=T.alloc_shared((n,128),'bfloat16')
            if reduction<128:
                a_sub=T.alloc_shared((m,reduction),'float16');b_sub=T.alloc_shared((n,reduction),'float16')
                partial=T.alloc_fragment((m,n),'float32')
            row_scale=T.alloc_shared((m,),'float32')
            block=T.alloc_fragment((m,n),'float32');total=T.alloc_fragment((m,n),'float32')
            T.clear(total)
            words=T.view(x,shape=(r,k//4),dtype='uint32')
            destination=T.view(lhs,shape=(m,32),dtype='uint32')
            weights=T.view(w,dtype='float8_e4m3fn')
            for local in T.Pipelined(k//(128*parts),num_stages=stages):
                tile=part*(k//(128*parts))+local
                if packed:
                    for i,j in T.Parallel(m,32):
                        bits=T.if_then_else(by*m+i<r,words[T.min(by*m+i,r-1),tile*32+j],0)
                        if stages>1:
                            for byte in T.unroll(4):
                                lhs[i,j*4+byte]=T.reinterpret('float8_e4m3fn',T.cast((bits>>(byte*8))&255,'uint8'))
                        else:destination[i,j]=bits
                else:
                    activations=T.view(x,dtype='float8_e4m3fn')
                    T.copy(activations[by*m:by*m+m,tile*128:tile*128+128],lhs)
                for i in T.Parallel(m):
                    row_scale[i]=T.if_then_else(by*m+i<r,activation_scales[T.min(by*m+i,r-1),tile],1)
                T.copy(weights[bx*n:bx*n+n,tile*128:tile*128+128],rhs)
                # The packed write aliases lhs through a different dtype view.
                # Make its cross-warp dependency explicit before widening.
                T.sync_threads()
                if asynchronous:
                    for i,j in T.Parallel(m,128):
                        a_batch[i,j]=lhs[i,j//32*32+(j%16)//2*4+j%2+(j%32)//16*2]
                    for i,j in T.Parallel(n,128):
                        b_batch[i,j]=rhs[i,j//32*32+(j%16)//2*4+j%2+(j%32)//16*2]
                    T.sync_threads()
                    T.wgmma_gemm(a_batch[:,0:32],b_batch[:,0:32],block,transpose_B=True,clear_accum=True)
                    T.wgmma_gemm(a_batch[:,32:64],b_batch[:,32:64],p1,transpose_B=True,clear_accum=True)
                    T.wgmma_gemm(a_batch[:,64:96],b_batch[:,64:96],p2,transpose_B=True,clear_accum=True)
                    T.wgmma_gemm(a_batch[:,96:128],b_batch[:,96:128],p3,transpose_B=True,clear_accum=True)
                    T.wait_wgmma(0)
                    for i,j in T.Parallel(m,n):
                        block[i,j]=T.call_extern('float32','__fadd_rn',block[i,j],p1[i,j])
                        block[i,j]=T.call_extern('float32','__fadd_rn',block[i,j],p2[i,j])
                        block[i,j]=T.call_extern('float32','__fadd_rn',block[i,j],p3[i,j])
                elif reduction==128:
                    for i,j in T.Parallel(m,128):a[i,j]=lhs[i,j]
                    for i,j in T.Parallel(n,128):b[i,j]=rhs[i,j]
                    T.sync_threads()
                    T.gemm(a,b,block,transpose_B=True,clear_accum=True)
                else:
                    T.clear(block)
                    for sub in T.serial(128//reduction):
                        if packed_widen:
                            lhs_pairs=T.view(lhs,shape=(m,64),dtype='uint16')
                            rhs_pairs=T.view(rhs,shape=(n,64),dtype='uint16')
                            a_pairs=T.view(a_sub,shape=(m,16),dtype='uint32')
                            b_pairs=T.view(b_sub,shape=(n,16),dtype='uint32')
                            for i,j in T.Parallel(m,16):
                                a_pairs[i,j]=T.call_extern('uint32','tensor_widen_e4m3x2',lhs_pairs[i,sub*16+j%8*2+j//8])
                            for i,j in T.Parallel(n,16):
                                b_pairs[i,j]=T.call_extern('uint32','tensor_widen_e4m3x2',rhs_pairs[i,sub*16+j%8*2+j//8])
                            T.sync_threads()
                        elif reorder:
                            for i,j in T.Parallel(m,reduction):
                                a_sub[i,j]=lhs[i,sub*32+(j%16)//2*4+j%2+j//16*2]
                            for i,j in T.Parallel(n,reduction):
                                b_sub[i,j]=rhs[i,sub*32+(j%16)//2*4+j%2+j//16*2]
                        else:
                            T.copy(lhs[:,sub*reduction:(sub+1)*reduction],a_sub)
                            T.copy(rhs[:,sub*reduction:(sub+1)*reduction],b_sub)
                        T.gemm(a_sub,b_sub,partial,transpose_B=True,clear_accum=True)
                        for i,j in T.Parallel(m,n):
                            block[i,j]=T.call_extern('float32','__fadd_rn',block[i,j],partial[i,j])
                for i,j in T.Parallel(m,n):
                    product=T.call_extern('float32','__fmul_rn',block[i,j],row_scale[i])
                    total[i,j]=T.call_extern('float32','__fmaf_rn',product,
                        T.cast(scales[(bx*n+j)//128,tile],'float32'),total[i,j])
            for i,j in T.Parallel(m,n):
                if by*m+i<r:
                    if parts>1:out[by*m+i,part,bx*n+j]=total[i,j]
                    else:out[by*m+i,bx*n+j]=total[i,j]
    return primitive(arguments,algorithm)
