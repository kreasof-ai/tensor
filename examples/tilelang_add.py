"""An ordinary TileLang factory with a Tensor export; see the migration guide."""

import tilelang.language as T


def make_add(n, block=128):
    @T.prim_func
    def add(a: T.Tensor((n,), "float32"),
            b: T.Tensor((n,), "float32"),
            out: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for lane in T.Parallel(block):
                i = bx * block + lane
                if i < n:
                    out[i] = a[i] + b[i]

    return add


def tensor_export():
    return {"kernel": make_add(1025), "outputs": ["out"]}
