"""Decode authoritative FP8 KV once per attention call into BF16 scratch."""


def decode(p):
    import tilelang.language as T
    slots,cap=p['slots'],p['capacity'];d=256;block=128
    if cap%block:raise ValueError('attention workspace requires capacity divisible by 128')
    @T.prim_func
    def kernel(kc:T.Tensor((slots,2,cap,d),'uint8'),vc:T.Tensor((slots,2,cap,d),'uint8'),
               ks:T.Tensor((slots,2,cap,2),'float32'),vs:T.Tensor((slots,2,cap,2),'float32'),
               positions:T.Tensor((slots,),'int32'),lengths:T.Tensor((slots,),'int32'),
               decoded_k:T.Tensor((slots,2,cap,d),'bfloat16'),
               decoded_v:T.Tensor((slots,2,cap,d),'bfloat16')):
        with T.Kernel(slots,2,T.ceildiv(cap,block),threads=256) as (slot,head,tile):
            count=positions[slot]+lengths[slot]
            if lengths[slot]>0 and tile*block<count:
                ek=T.alloc_shared((block,d),'float8_e4m3fn')
                ev=T.alloc_shared((block,d),'float8_e4m3fn')
                sk=T.alloc_shared((block,2),'float32');sv=T.alloc_shared((block,2),'float32')
                keys=T.view(kc,dtype='float8_e4m3fn');values=T.view(vc,dtype='float8_e4m3fn')
                T.copy(keys[slot,head,tile*block:tile*block+block,0:d],ek)
                T.copy(values[slot,head,tile*block:tile*block+block,0:d],ev)
                T.copy(ks[slot,head,tile*block:tile*block+block,0:2],sk)
                T.copy(vs[slot,head,tile*block:tile*block+block,0:2],sv)
                T.sync_threads()
                for i,j in T.Parallel(block,d):
                    decoded_k[slot,head,tile*block+i,j]=0
                    decoded_v[slot,head,tile*block+i,j]=0
                    if tile*block+i<count:
                        decoded_k[slot,head,tile*block+i,j]=T.cast(ek[i,j],'float32')*sk[i,j//128]
                        decoded_v[slot,head,tile*block+i,j]=T.cast(ev[i,j],'float32')*sv[i,j//128]
    return kernel
