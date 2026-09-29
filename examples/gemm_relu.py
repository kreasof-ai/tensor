"""A second product export: tiled float16 GEMM with bias and ReLU."""

import tilelang.language as T

M = N = K = 64
BLOCK = 64
BLOCK_K = 32


@T.prim_func
def gemm_relu(a: T.Tensor((M, K), "float16"),
              b: T.Tensor((K, N), "float16"),
              bias: T.Tensor((N,), "float16"),
              out: T.Tensor((M, N), "float16")):
    with T.Kernel(T.ceildiv(N, BLOCK), T.ceildiv(M, BLOCK), threads=128) as (bx, by):
        a_shared = T.alloc_shared((BLOCK, BLOCK_K), "float16")
        b_shared = T.alloc_shared((BLOCK_K, BLOCK), "float16")
        accum = T.alloc_fragment((BLOCK, BLOCK), "float32")
        T.clear(accum)
        for tile in T.Pipelined(T.ceildiv(K, BLOCK_K), num_stages=3):
            T.copy(a[by * BLOCK, tile * BLOCK_K], a_shared)
            T.copy(b[tile * BLOCK_K, bx * BLOCK], b_shared)
            T.gemm(a_shared, b_shared, accum)
        for row, col in T.Parallel(BLOCK, BLOCK):
            accum[row, col] = T.max(accum[row, col] + bias[bx * BLOCK + col], 0)
        T.copy(accum, out[by * BLOCK, bx * BLOCK])


def tensor_export():
    return {
        "kernel": gemm_relu,
        "outputs": ["out"],
    }
