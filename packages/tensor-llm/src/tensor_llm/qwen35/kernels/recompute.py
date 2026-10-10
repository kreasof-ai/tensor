"""Save compact verifier inputs and restore only a rejected accepted prefix."""


def save_scan(p):
    import tilelang.language as T
    s,c=p['slots'],p['chunk'];r=s*c;h,k,v=32,128,128
    state_count=s*h*v*k;input_count=r*h*k;small_count=r*h
    @T.prim_func
    def kernel(state:T.Tensor((s,h,v,k),'float32'),query:T.Tensor((r,h,k),'float32'),key:T.Tensor((r,h,k),'float32'),
               value:T.Tensor((r,h,v),'float32'),g:T.Tensor((r,h),'float32'),beta:T.Tensor((r,h),'float32'),
               initial:T.Tensor((s,h,v,k),'float32'),saved_q:T.Tensor((r,h,k),'float32'),saved_k:T.Tensor((r,h,k),'float32'),
               saved_v:T.Tensor((r,h,v),'float32'),saved_g:T.Tensor((r,h),'float32'),
               saved_beta:T.Tensor((r,h),'float32')):
        with T.Kernel(T.ceildiv(max(state_count,input_count),1024),threads=256) as block:
            source=T.view(state,shape=(state_count,));destination=T.view(initial,shape=(state_count,))
            qq=T.view(query,shape=(input_count,));sq=T.view(saved_q,shape=(input_count,))
            kk=T.view(key,shape=(input_count,));sk=T.view(saved_k,shape=(input_count,))
            vv=T.view(value,shape=(input_count,));sv=T.view(saved_v,shape=(input_count,))
            gg=T.view(g,shape=(small_count,));sg=T.view(saved_g,shape=(small_count,))
            bb=T.view(beta,shape=(small_count,));sb=T.view(saved_beta,shape=(small_count,))
            for j in T.Parallel(1024):
                index=block*1024+j
                if index<state_count:destination[index]=source[index]
                if index<input_count:sq[index]=qq[index];sk[index]=kk[index];sv[index]=vv[index]
                if index<small_count:sg[index]=gg[index];sb[index]=bb[index]
    return kernel


def save_conv(p):
    import tilelang.language as T
    s,c=p['slots'],p['chunk'];r=s*c;channels=8192;count=r*channels;state_count=s*channels*3
    @T.prim_func
    def kernel(x:T.Tensor((r,channels),'float32'),state:T.Tensor((s,channels,3),'float32'),
               saved_x:T.Tensor((r,channels),'bfloat16'),initial:T.Tensor((s,channels,3),'float32')):
        with T.Kernel(T.ceildiv(max(count,state_count),1024),threads=256) as block:
            source=T.view(state,shape=(state_count,));destination=T.view(initial,shape=(state_count,))
            for j in T.Parallel(1024):
                index=block*1024+j
                # T.view checks total bits with an int32 literal. C64/window128
                # reaches 2**31 bits; direct indexing keeps identical BF16 copies.
                if index<count:saved_x[index//channels,index%channels]=x[index//channels,index%channels]
                if index<state_count:destination[index]=source[index]
    return kernel


def restore_scan(p):
    """Reset only partial slots; the unchanged scan then replays their inputs.

    Using the original scan retains its reduction layout and FP32 rounding.
    """
    import tilelang.language as T
    s,c=p['slots'],p['chunk'];h,k,v=32,128,128;count=h*k*v
    @T.prim_func
    def kernel(initial:T.Tensor((s,h,v,k),'float32'),accepted:T.Tensor((s,),'int32'),
               lengths:T.Tensor((s,),'int32'),state:T.Tensor((s,h,v,k),'float32'),
               replay_lengths:T.Tensor((s,),'int32')):
        with T.Kernel(s,T.ceildiv(count,1024),threads=256) as (slot,block):
            source=T.view(initial,shape=(s,count));dest=T.view(state,shape=(s,count))
            partial=(accepted[slot]>0)&(accepted[slot]<lengths[slot])
            if block==0:
                for j in T.Parallel(1):replay_lengths[slot]=T.if_then_else(partial,accepted[slot],0)
            if partial:
                for j in T.Parallel(1024):
                    index=block*1024+j
                    if index<count:dest[slot,index]=source[slot,index]
    return kernel


def restore_conv(p):
    import tilelang.language as T
    s,c=p['slots'],p['chunk'];r=s*c;channels=8192
    @T.prim_func
    def kernel(initial:T.Tensor((s,channels,3),'float32'),saved_x:T.Tensor((r,channels),'bfloat16'),
               accepted:T.Tensor((s,),'int32'),lengths:T.Tensor((s,),'int32'),
               state:T.Tensor((s,channels,3),'float32')):
        with T.Kernel(s,channels//256,threads=256) as (slot,tile):
            if (accepted[slot]>0)&(accepted[slot]<lengths[slot]):
                for j in T.Parallel(256):
                    col=tile*256+j
                    for history in T.unroll(3):
                        pos=accepted[slot]-3+history
                        if pos>=0:state[slot,col,history]=T.cast(saved_x[slot*c+pos,col],'float32')
                        else:state[slot,col,history]=initial[slot,col,3+pos]
    return kernel
