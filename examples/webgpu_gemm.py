"""Portable inference tile schedule; specialize constants before AOT build."""
import tilelang.language as T

M = 33
N = 65
K = 37
DTYPE = "float16"
OUTPUT_DTYPE = "float16"
TRANSPOSE_B = False
USE_BIAS = True
RELU = True
BLOCK = 32
BLOCK_K = 16
B_SHAPE = (N, K) if TRANSPOSE_B else (K, N)


@T.prim_func
def linear(a: T.Tensor((M, K), DTYPE), b: T.Tensor(B_SHAPE, DTYPE),
           bias: T.Tensor((N,), DTYPE), out: T.Tensor((M, N), OUTPUT_DTYPE)):
    with T.Kernel(T.ceildiv(N, BLOCK), T.ceildiv(M, BLOCK), threads=128) as (bx, by):
        aa = T.alloc_shared((BLOCK, BLOCK_K), DTYPE)
        bb = T.alloc_shared((BLOCK, BLOCK_K) if TRANSPOSE_B else (BLOCK_K, BLOCK), DTYPE)
        cc = T.alloc_fragment((BLOCK, BLOCK), "float32")
        T.clear(cc)
        for tile in T.serial(T.ceildiv(K, BLOCK_K)):
            T.copy(a[by * BLOCK, tile * BLOCK_K], aa)
            if TRANSPOSE_B:
                T.copy(b[bx * BLOCK, tile * BLOCK_K], bb)
            else:
                T.copy(b[tile * BLOCK_K, bx * BLOCK], bb)
            T.gemm(aa, bb, cc, transpose_B=TRANSPOSE_B)
        for i, j in T.Parallel(BLOCK, BLOCK):
            if USE_BIAS:
                cc[i, j] += bias[bx * BLOCK + j]
            if RELU:
                cc[i, j] = T.max(cc[i, j], 0)
        T.copy(cc, out[by * BLOCK, bx * BLOCK])


def tensor_export():
    kernel = linear
    if not USE_BIAS:
        from tilelang import tvm
        mod = tvm.tirx.transform.Simplify()(tvm.IRModule({"linear": kernel}))
        kernel = mod["linear"]
        removed = [p for p, buffer in kernel.buffer_map.items() if str(buffer.name) == "bias"]
        kernel = tvm.tirx.PrimFunc([p for p in kernel.params if p not in removed], kernel.body,
            kernel.ret_type, {p: b for p,b in kernel.buffer_map.items() if p not in removed}, kernel.attrs)
    return {"kernel": kernel, "outputs": ["out"]}
