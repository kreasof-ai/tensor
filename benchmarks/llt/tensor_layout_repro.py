"""Minimal pinned-TileLang layout limitation; building this is expected to fail.

TENSOR_NVRTC_HOME=build/nvrtc-12.9 .venv/bin/tensor build \
    benchmarks/llt/tensor_layout_repro.py --out /tmp/layout-repro.tbin
"""

import tilelang.language as T


@T.prim_func
def repro(
    q: T.Tensor((32, 32), "float16"),
    k: T.Tensor((64, 32), "float16"),
    out: T.Tensor((1,), "float32"),
):
    with T.Kernel(1, threads=128):
        a = T.alloc_shared((32, 32), "float16")
        b = T.alloc_shared((64, 32), "float16")
        scores = T.alloc_fragment((32, 64), "float32")
        maximum = T.alloc_fragment((32,), "float32")
        T.copy(q, a)
        T.copy(k, b)
        T.gemm(
            a,
            b,
            scores,
            transpose_B=True,
            clear_accum=True,
            policy=T.GemmWarpPolicy.FullRow,
        )
        T.reduce_max(scores, maximum, dim=1, clear=True)
        out[0] = maximum[0]


def tensor_export():
    return {"kernel": repro, "outputs": ["out"]}
