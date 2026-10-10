"""Losslessly widen and pair-permute FP8 weight operands, leaving scales separate."""


def widen(p):
    import tilelang.language as T
    shape=tuple(p['shape'])
    if len(shape)!=3:raise ValueError('expert operand cache needs [experts,output,input]')
    experts,rows,k=shape;count=rows*k
    if k%32:raise ValueError('operand pairing requires K32')
    @T.prim_func
    def kernel(source:T.Tensor(shape,'uint8'),out:T.Tensor(shape,'float16')):
        with T.Kernel(experts,T.ceildiv(count,1024),threads=256) as (expert,block):
            for index in T.Parallel(1024):
                pos=block*1024+index
                if pos<count:
                    row=pos//k;j=pos%32
                    original=(pos%k)//32*32+(j%16)//2*4+j%2+j//16*2
                    out[expert,row,pos%k]=T.reinterpret('float8_e4m3fn',source[expert,row,original])
    return kernel
