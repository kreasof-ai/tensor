"""Generate a TIRx artifact in whatever TileLang version is running.

Used to test the *forward* direction of E4a: does an artifact produced by an
older TileLang still load in a newer one? That is the direction that matters
for a module cache, because users upgrade and want their artifacts to survive.

    <python> make_artifact.py <output.json>
"""

import json
import sys

import tilelang
import tilelang.language as T
import tvm
import tvm.ir as ir

M = N = K = 512


@T.prim_func
def main(
    A: T.Tensor((M, K), "float16"),
    B: T.Tensor((K, N), "float16"),
    bias: T.Tensor((N,), "float16"),
    C: T.Tensor((M, N), "float16"),
):
    with T.Kernel(T.ceildiv(N, 64), T.ceildiv(M, 64), threads=128) as (bx, by):
        A_shared = T.alloc_shared((64, 32), "float16")
        B_shared = T.alloc_shared((32, 64), "float16")
        C_local = T.alloc_fragment((64, 64), "float32")
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 32), num_stages=3):
            T.copy(A[by * 64, k * 32], A_shared)
            T.copy(B[k * 32, bx * 64], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        for i, j in T.Parallel(64, 64):
            C_local[i, j] = T.max(C_local[i, j] + bias[bx * 64 + j], 0)
        T.copy(C_local, C[by * 64, bx * 64])


out = sys.argv[1]
mod = tvm.IRModule({"main": main})
js = ir.save_json(mod)
open(out, "w", encoding="utf-8").write(js)
print(f"MADE tilelang={tilelang.__version__} tvm={tvm.__version__} bytes={len(js)}")
print(f"MADE_META {json.dumps(json.loads(js).get('metadata', {}))}")
