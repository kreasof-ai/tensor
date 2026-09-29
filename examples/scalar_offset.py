"""An int64 scalar offset must retain all 64 bits in the CUDA launch ABI."""

import tilelang.language as T


@T.prim_func
def offset(a: T.Tensor((129,), "int64"), c: T.Tensor((129,), "int64"), delta: T.int64):
    with T.Kernel(2, threads=128) as block:
        for lane in T.Parallel(128):
            index = block * 128 + lane
            if index < 129:
                c[index] = a[index] + delta


def tensor_export():
    return {"kernel": offset, "outputs": ["c"]}
