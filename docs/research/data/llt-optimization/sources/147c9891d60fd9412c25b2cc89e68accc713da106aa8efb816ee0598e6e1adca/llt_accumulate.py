"""Streaming classifier gradient accumulation without full-size temporary sums."""


def accumulate(p):
    import tilelang.language as T

    n, dtype = p

    @T.prim_func
    def kernel(value: T.Tensor((n,), dtype), accumulator: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 256), threads=256) as block:
            for lane in T.Parallel(256):
                i = block * 256 + lane
                if i < n:
                    accumulator[i] += T.cast(value[i], "float32")

    return kernel
