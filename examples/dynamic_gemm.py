"""One float16 GEMM binary for multiple row counts, with static 32x32 tiles."""

import tilelang.language as T

ROWS = T.dynamic("rows")


@T.prim_func
def matmul(a: T.Tensor((ROWS, 32), "float16"),
           b: T.Tensor((32, 32), "float16"),
           c: T.Tensor((ROWS, 32), "float16")):
    with T.Kernel(T.ceildiv(ROWS, 32), threads=128) as block:
        aa = T.alloc_shared((32, 32), "float16")
        bb = T.alloc_shared((32, 32), "float16")
        cc = T.alloc_fragment((32, 32), "float32")
        T.copy(a[block * 32, 0], aa)
        T.copy(b, bb)
        T.gemm(aa, bb, cc, clear_accum=True)
        T.copy(cc, c[block * 32, 0])


def tensor_export():
    return {"kernel": matmul, "outputs": ["c"]}
