"""One binary for multiple lengths, with a runtime float32 scale."""

import tilelang.language as T

SIZE = T.dynamic("size")


@T.prim_func
def affine(a: T.Tensor((SIZE,), "float32"),
           b: T.Tensor((SIZE,), "float32"),
           c: T.Tensor((SIZE,), "float32"),
           scale: T.float32):
    with T.Kernel(T.ceildiv(SIZE, 128), threads=128) as block:
        for lane in T.Parallel(128):
            index = block * 128 + lane
            if index < SIZE:
                c[index] = T.max(scale * a[index] + b[index], 0.0)


def tensor_export():
    return {"kernel": affine, "outputs": ["c"]}
