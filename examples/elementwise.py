"""Build with: tensor build examples/elementwise.py --out build/elementwise.tbin"""

import tilelang.language as T

SIZE = 129


@T.prim_func
def elementwise(a: T.Tensor((SIZE,), "float32"),
                b: T.Tensor((SIZE,), "float32"),
                c: T.Tensor((SIZE,), "float32")):
    with T.Kernel(T.ceildiv(SIZE, 128), threads=128) as block:
        for lane in T.Parallel(128):
            index = block * 128 + lane
            if index < SIZE:
                c[index] = T.max(a[index] * 2.0 + b[index], 0.0)


def tensor_export():
    return {
        "kernel": elementwise,
        "launch": {
            "grid": [(SIZE + 127) // 128, 1, 1],
            "block": [128, 1, 1],
            "shared_memory_bytes": 0,
        },
    }
