"""Hopper expert schedule candidates; preserve block-128 scale arithmetic.

Only launch geometry and load pipelining vary. The selected profile must pass
primitive and same-state full-model checks before serving benchmarks.
"""

def expert_kernel(p):
    import tilelang.language as T
    from tensor.compiler.entry import primitive
    rows,k,o=p['rows'],p['k'],p['o']
    n,top,experts=p.get('columns',64),8,256
    m,threads=p.get('block_m',16),p.get('threads',128)
    routed=p.get('routed_input',False)
    compact=p.get('compact',True)
    stages=p.get('stages',1)
    packed_gather=p.get('packed_gather',False)
    bf16_mma=p.get('bf16_mma',False)
    parts=p.get('partitions',1)
    persistent=p.get('persistent_tiles',0)
    if type(persistent) is not int or persistent<0:raise ValueError('invalid persistent expert grid')
    reduction=p.get('mma_reduction',128)
    reorder=p.get('mma_reorder',False)
    asynchronous=p.get('async_mma',False)
    packed_widen=p.get('packed_widen',False)
    resident_weights=p.get('resident_weights',False)
    if resident_weights and (not bf16_mma or not reorder or asynchronous or packed_widen):
        raise ValueError('resident operands require synchronous paired MMA')
    if packed_widen and (not bf16_mma or not reorder or asynchronous):
        raise ValueError('packed widening requires synchronous paired MMA')
    if asynchronous and not (reorder and reduction==32):raise ValueError('async MMA requires paired K32')
    if reorder and reduction!=32:raise ValueError('paired operands require K32')
    if reduction not in (32,128):raise ValueError('invalid Hopper MMA reduction')
    if k%(128*parts):raise ValueError('invalid expert reduction partition')
    max_tiles=T.ceildiv(rows*top,m)+experts-1
    arguments=[('x',(rows,top,k) if routed else (rows,k),'uint8'),
               ('activation_scales',(rows,top,k//128) if routed else (rows,k//128),'float32'),
               ('w',(experts,o,k),'float16' if resident_weights else 'uint8'),('scales',(experts,T.ceildiv(o,128),k//128),'bfloat16'),
               ('counts',(experts,),'int32'),('routes',(experts,rows),'int32'),
               ('out',(rows,top,parts,o) if parts>1 else (rows,top,o),'float32')]
    @T.macro
    def body(x,activation_scales,w,scales,counts,routes,out,bx,expert,tokens,part):
        if tokens*m<counts[expert]:
            if packed_widen:
                from tensor_llm.qwen35.kernels.fp8_operand import CUDA_SOURCE
                T.import_source(CUDA_SOURCE)
            lhs=T.alloc_shared((m,128),'float8_e4m3fn')
            rhs=T.alloc_shared((n,128),'float8_e4m3fn')
            if bf16_mma:
                if asynchronous:
                    lhs_batch=T.alloc_shared((m,128),'float16')
                    rhs_batch=T.alloc_shared((n,128),'float16')
                    p1=T.alloc_fragment((m,n),'float32');p2=T.alloc_fragment((m,n),'float32');p3=T.alloc_fragment((m,n),'float32')
                if reduction==128:
                    lhs_wide=T.alloc_shared((m,128),'bfloat16')
                    rhs_wide=T.alloc_shared((n,128),'bfloat16')
                if reduction<128:
                    lhs_sub=T.alloc_shared((m,reduction),'float16')
                    rhs_sub=T.alloc_shared((n,reduction),'float16')
                    partial=T.alloc_fragment((m,n),'float32')
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
                if packed_gather:
                    words=T.view(x,shape=(rows,top,k//4) if routed else (rows,k//4),dtype='uint32')
                    destination=T.view(lhs,shape=(m,32),dtype='uint32')
                    for b,j in T.Parallel(m,32):
                        bits=T.alloc_var('uint32');bits=0
                        if selected[b]>=0:
                            if routed:bits=words[selected[b]//top,selected[b]%top,tile*32+j]
                            else:bits=words[selected[b]//top,tile*32+j]
                        if stages>1:
                            # Write through the buffer that the pipeline owns;
                            # a second dtype view hides double-buffer aliases.
                            for byte in T.unroll(4):
                                lhs[b,j*4+byte]=T.reinterpret('float8_e4m3fn',T.cast((bits>>(byte*8))&255,'uint8'))
                        else:destination[b,j]=bits
                else:
                    for b,j in T.Parallel(m,128):
                        bits=T.alloc_var('uint8');bits=0
                        if selected[b]>=0:
                            if routed:bits=x[selected[b]//top,selected[b]%top,tile*128+j]
                            else:bits=x[selected[b]//top,tile*128+j]
                        lhs[b,j]=T.reinterpret('float8_e4m3fn',bits)
                if not resident_weights:
                    weights=T.view(w,dtype='float8_e4m3fn')
                    T.copy(weights[expert,bx*n:bx*n+n,tile*128:tile*128+128],rhs)
                if packed_gather:T.sync_threads()
                if bf16_mma:
                    if asynchronous:
                        for b,j in T.Parallel(m,128):
                            lhs_batch[b,j]=lhs[b,j//32*32+(j%16)//2*4+j%2+(j%32)//16*2]
                        for i,j in T.Parallel(n,128):
                            rhs_batch[i,j]=rhs[i,j//32*32+(j%16)//2*4+j%2+(j%32)//16*2]
                        T.sync_threads()
                        T.wgmma_gemm(lhs_batch[:,0:32],rhs_batch[:,0:32],block,transpose_B=True,clear_accum=True)
                        T.wgmma_gemm(lhs_batch[:,32:64],rhs_batch[:,32:64],p1,transpose_B=True,clear_accum=True)
                        T.wgmma_gemm(lhs_batch[:,64:96],rhs_batch[:,64:96],p2,transpose_B=True,clear_accum=True)
                        T.wgmma_gemm(lhs_batch[:,96:128],rhs_batch[:,96:128],p3,transpose_B=True,clear_accum=True)
                        T.wait_wgmma(0)
                        for b,j in T.Parallel(m,n):
                            block[b,j]=T.call_extern('float32','__fadd_rn',block[b,j],p1[b,j])
                            block[b,j]=T.call_extern('float32','__fadd_rn',block[b,j],p2[b,j])
                            block[b,j]=T.call_extern('float32','__fadd_rn',block[b,j],p3[b,j])
                    elif reduction==128:
                        for b,j in T.Parallel(m,128):lhs_wide[b,j]=lhs[b,j]
                        for i,j in T.Parallel(n,128):rhs_wide[i,j]=rhs[i,j]
                        T.sync_threads()
                        T.gemm(lhs_wide,rhs_wide,block,transpose_B=True,clear_accum=True)
                    else:
                        T.clear(block)
                        for sub in T.serial(128//reduction):
                            if packed_widen:
                                lhs_pairs=T.view(lhs,shape=(m,64),dtype='uint16')
                                rhs_pairs=T.view(rhs,shape=(n,64),dtype='uint16')
                                a_pairs=T.view(lhs_sub,shape=(m,16),dtype='uint32')
                                b_pairs=T.view(rhs_sub,shape=(n,16),dtype='uint32')
                                for b,j in T.Parallel(m,16):
                                    a_pairs[b,j]=T.call_extern('uint32','tensor_widen_e4m3x2',lhs_pairs[b,sub*16+j%8*2+j//8])
                                for i,j in T.Parallel(n,16):
                                    b_pairs[i,j]=T.call_extern('uint32','tensor_widen_e4m3x2',rhs_pairs[i,sub*16+j%8*2+j//8])
                                T.sync_threads()
                            elif reorder:
                                for b,j in T.Parallel(m,reduction):
                                    lhs_sub[b,j]=lhs[b,sub*32+(j%16)//2*4+j%2+j//16*2]
                                if resident_weights:
                                    T.copy(w[expert,bx*n:bx*n+n,tile*128+sub*32:tile*128+(sub+1)*32],rhs_sub)
                                else:
                                    for i,j in T.Parallel(n,reduction):
                                        rhs_sub[i,j]=rhs[i,sub*32+(j%16)//2*4+j%2+j//16*2]
                            else:
                                T.copy(lhs[:,sub*reduction:(sub+1)*reduction],lhs_sub)
                                T.copy(rhs[:,sub*reduction:(sub+1)*reduction],rhs_sub)
                            T.gemm(lhs_sub,rhs_sub,partial,transpose_B=True,clear_accum=True)
                            for b,j in T.Parallel(m,n):
                                block[b,j]=T.call_extern('float32','__fadd_rn',block[b,j],partial[b,j])
                else:
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
                if selected[b]>=0:
                    if parts>1:out[selected[b]//top,selected[b]%top,part,bx*n+j]=total[b,j]
                    else:out[selected[b]//top,selected[b]%top,bx*n+j]=total[b,j]
    @T.macro
    def algorithm(x,activation_scales,w,scales,counts,routes,out):
        with T.Kernel(o//n,experts,T.ceildiv(rows,m)*parts,threads=threads) as (bx,expert,group):
            body(x,activation_scales,w,scales,counts,routes,out,bx,expert,group//parts,group%parts)

    @T.macro
    def compact_algorithm(x,activation_scales,w,scales,counts,routes,out,tile_experts,tile_offsets):
        with T.Kernel(o//n,persistent if persistent else max_tiles,parts,threads=threads) as (bx,block,part):
            if persistent:
                for iteration in T.serial(T.ceildiv(max_tiles,persistent)):
                    tile=block+iteration*persistent
                    if tile<max_tiles:
                        expert=tile_experts[tile]
                        if expert>=0:
                            body(x,activation_scales,w,scales,counts,routes,out,bx,expert,tile_offsets[tile],part)
            else:
                expert=tile_experts[block]
                if expert>=0:
                    body(x,activation_scales,w,scales,counts,routes,out,bx,expert,tile_offsets[block],part)

    if compact:
        arguments += [('tile_experts',(max_tiles,),'int32'),('tile_offsets',(max_tiles,),'int32')]
    return primitive(arguments,compact_algorithm if compact else algorithm)
