"""Additional P0 kernels for runtime shape and scheduling boundary probes."""

import tilelang.language as T


def composition_stage(size, tile=128, activation=False):
    if activation:
        @T.prim_func
        def activate(x:T.Tensor((size,),"float32"), out:T.Tensor((size,),"float32")):
            with T.Kernel(T.ceildiv(size,tile),threads=tile) as block:
                for lane in T.Parallel(tile):
                    i=block*tile+lane
                    if i<size:
                        out[i]=T.max(x[i],0.0)
        return activate
    @T.prim_func
    def affine(a:T.Tensor((size,),"float32"), b:T.Tensor((size,),"float32"), out:T.Tensor((size,),"float32")):
        with T.Kernel(T.ceildiv(size,tile),threads=tile) as block:
            for lane in T.Parallel(tile):
                i=block*tile+lane
                if i<size:
                    out[i]=2.0*a[i]+b[i]
    return affine


def dynamic_elementwise():
    size = T.dynamic("size")

    @T.prim_func
    def elementwise(a: T.Tensor((size,), "float32"),
                    b: T.Tensor((size,), "float32"),
                    c: T.Tensor((size,), "float32")):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
            for lane in T.Parallel(128):
                i = block * 128 + lane
                if i < size:
                    c[i] = T.max(2.0 * a[i] + b[i], 0.0)

    return elementwise


def dynamic_gemm(dynamic_tile=False, static_rows=None):
    rows = T.dynamic("rows") if static_rows is None else static_rows
    tile = T.dynamic("tile") if dynamic_tile else 32

    @T.prim_func
    def matmul(a: T.Tensor((rows, 32), "float16"),
               b: T.Tensor((32, 32), "float16"),
               c: T.Tensor((rows, 32), "float16")):
        with T.Kernel(T.ceildiv(rows, tile), threads=128) as block:
            aa = T.alloc_shared((tile, 32), "float16")
            bb = T.alloc_shared((32, 32), "float16")
            cc = T.alloc_fragment((tile, 32), "float32")
            T.copy(a[block * tile, 0], aa)
            T.copy(b, bb)
            T.gemm(aa, bb, cc, clear_accum=True)
            T.copy(cc, c[block * tile, 0])

    return matmul
