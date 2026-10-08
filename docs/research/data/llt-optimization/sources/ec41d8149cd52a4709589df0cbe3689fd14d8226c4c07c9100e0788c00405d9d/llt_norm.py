"""LayerNorm and channel bias for actual nanoGPT; FP32 reductions."""


def make_kernel(p):
    import tilelang.language as T
    r,c,dt=p['r'],p['c'],p['dtype']
    kind=p['kind']
    width=1<<(c-1).bit_length()
    if kind=='bias':
        @T.prim_func
        def kernel(x:T.Tensor((r,c),dt),bias:T.Tensor((c,),dt),out:T.Tensor((r,c),dt)):
            with T.Kernel(T.ceildiv(r*c,256),threads=256) as block:
                for lane in T.Parallel(256):
                    i=block*256+lane
                    if i<r*c:out[i//c,i%c]=T.cast(x[i//c,i%c],'float32')+T.cast(bias[i%c],'float32')
        return kernel
    if kind=='ln':
        eps=p['eps']
        @T.prim_func
        def kernel(x:T.Tensor((r,c),dt),w:T.Tensor((c,),'float32'),bias:T.Tensor((c,),'float32'),
                   out:T.Tensor((r,c),dt),mean:T.Tensor((r,),'float32'),inv:T.Tensor((r,),'float32')):
            with T.Kernel(r,threads=256) as row:
                vals=T.alloc_fragment((width,),'float32')
                total=T.alloc_fragment((1,),'float32')
                stat=T.alloc_shared((2,),'float32')
                for j in T.Parallel(width):vals[j]=T.if_then_else(j<c,T.cast(x[row,j],'float32'),0)
                T.reduce_sum(vals,total,dim=0,clear=True)
                total[0]/=c
                T.copy(total,stat[0:1])
                for j in T.Parallel(width):vals[j]=T.if_then_else(j<c,(T.cast(x[row,j],'float32')-stat[0])*(T.cast(x[row,j],'float32')-stat[0]),0)
                T.reduce_sum(vals,total,dim=0,clear=True)
                total[0]=T.rsqrt(total[0]/c+eps)
                T.copy(total,stat[1:2])
                for j in T.Parallel(c):out[row,j]=(T.cast(x[row,j],'float32')-stat[0])*stat[1]*w[j]+bias[j]
                mean[row]=stat[0];inv[row]=stat[1]
        return kernel
    if kind=='ln_dx':
        @T.prim_func
        def kernel(x:T.Tensor((r,c),dt),w:T.Tensor((c,),'float32'),dy:T.Tensor((r,c),dt),
                   mean:T.Tensor((r,),'float32'),inv:T.Tensor((r,),'float32'),out:T.Tensor((r,c),dt)):
            with T.Kernel(r,threads=256) as row:
                vals=T.alloc_fragment((width,),'float32')
                total=T.alloc_fragment((1,),'float32')
                stat=T.alloc_shared((2,),'float32')
                for j in T.Parallel(width):vals[j]=T.if_then_else(j<c,T.cast(dy[row,j],'float32')*w[j],0)
                T.reduce_sum(vals,total,dim=0,clear=True)
                total[0]/=c;T.copy(total,stat[0:1])
                for j in T.Parallel(width):vals[j]=T.if_then_else(j<c,T.cast(dy[row,j],'float32')*w[j]*(T.cast(x[row,j],'float32')-mean[row])*inv[row],0)
                T.reduce_sum(vals,total,dim=0,clear=True)
                total[0]/=c;T.copy(total,stat[1:2])
                for j in T.Parallel(c):
                    out[row,j]=inv[row]*(T.cast(dy[row,j],'float32')*w[j]-stat[0]-(T.cast(x[row,j],'float32')-mean[row])*inv[row]*stat[1])
        return kernel
    if kind=='params':
        tiles=T.ceildiv(r,32)
        @T.prim_func
        def kernel(x:T.Tensor((r,c),dt),dy:T.Tensor((r,c),dt),mean:T.Tensor((r,),'float32'),
                   inv:T.Tensor((r,),'float32'),dw:T.Tensor((tiles,c),'float32'),db:T.Tensor((tiles,c),'float32')):
            with T.Kernel(T.ceildiv(c,32),tiles,threads=128) as (block,tile):
                vals=T.alloc_fragment((32,32),'float32')
                sums=T.alloc_fragment((32,),'float32')
                for i,j in T.Parallel(32,32):
                    vals[i,j]=T.if_then_else((tile*32+i<r)&(block*32+j<c),
                        T.cast(dy[tile*32+i,block*32+j],'float32')*(T.cast(x[tile*32+i,block*32+j],'float32')-mean[tile*32+i])*inv[tile*32+i],0)
                T.reduce_sum(vals,sums,dim=0,clear=True)
                T.copy(sums,dw[tile,block*32:(block+1)*32])
                for i,j in T.Parallel(32,32):vals[i,j]=T.if_then_else((tile*32+i<r)&(block*32+j<c),T.cast(dy[tile*32+i,block*32+j],'float32'),0)
                T.reduce_sum(vals,sums,dim=0,clear=True)
                T.copy(sums,db[tile,block*32:(block+1)*32])
        return kernel
    if kind=='column_sum':
        tiles=T.ceildiv(r,32)
        @T.prim_func
        def kernel(x:T.Tensor((r,c),dt),out:T.Tensor((tiles,c),'float32')):
            with T.Kernel(T.ceildiv(c,32),tiles,threads=128) as (block,tile):
                vals=T.alloc_fragment((32,32),'float32')
                sums=T.alloc_fragment((32,),'float32')
                for i,j in T.Parallel(32,32):vals[i,j]=T.if_then_else((tile*32+i<r)&(block*32+j<c),T.cast(x[tile*32+i,block*32+j],'float32'),0)
                T.reduce_sum(vals,sums,dim=0,clear=True)
                T.copy(sums,out[tile,block*32:(block+1)*32])
        return kernel
    raise ValueError(kind)
