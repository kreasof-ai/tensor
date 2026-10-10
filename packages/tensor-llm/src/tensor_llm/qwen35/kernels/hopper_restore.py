"""Restore long-window recurrent checkpoints without oversized bit-count views."""


def restore_kernel(p):
    import tilelang.language as T
    slots,chunk=p['slots'],p['chunk'];shape=tuple(p['shape'])
    strides=[];width=1
    for size in reversed(shape):strides.insert(0,width);width*=size
    @T.prim_func
    def kernel(checkpoints:T.Tensor((slots,chunk-1,*shape),'float32'),
               state:T.Tensor((slots,*shape),'float32'),
               accepted:T.Tensor((slots,),'int32'),lengths:T.Tensor((slots,),'int32')):
        with T.Kernel(slots,T.ceildiv(width,256),threads=256) as (slot,tile):
            if (accepted[slot]>0)&(accepted[slot]<lengths[slot]):
                for j in T.Parallel(256):
                    col=tile*256+j
                    if col<width:
                        indices=tuple((col//stride)%size for stride,size in zip(strides,shape))
                        state[(slot,*indices)]=checkpoints[(slot,accepted[slot]-1,*indices)]
    return kernel
