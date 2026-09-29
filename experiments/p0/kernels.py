"""Phase 0 workload set.

Five kernels covering the shapes the proposal names in §26:
fused elementwise, tiled GEMM, reduction, irregular gather, FlashAttention-like.

These implement the named operations, but remain numerically unverified until
the NVIDIA validation runs. Source emission is not an execution test.

Every kernel is written once and lowered to several targets, so the source
that runs on the NVIDIA box is the same source measured on a no-GPU laptop.
"""

import tilelang
import tilelang.language as T

# ---------------------------------------------------------------- elementwise


def fused_elementwise(M, N, dtype="float16"):
    """c = relu(2 * a + b), including partial tiles."""

    @T.prim_func
    def main(
        a: T.Tensor((M, N), dtype),
        b: T.Tensor((M, N), dtype),
        c: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(M, 16), T.ceildiv(N, 128), threads=128) as (bx, by):
            for i, j in T.Parallel(16, 128):
                row, col = bx * 16 + i, by * 128 + j
                if row < M and col < N:
                    c[row, col] = T.max(a[row, col] * 2.0 + b[row, col], 0)

    return main


# ------------------------------------------------------------------------ gemm


def gemm_relu(M, N, K, block_M=64, block_N=64, block_K=32, dtype="float16", accum_dtype="float32"):
    """C = relu(A @ B + bias) -- GEMM with a fused bias+ReLU epilogue."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        bias: T.Tensor((N,), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                if by * block_M + i < M and bx * block_N + j < N:
                    C_local[i, j] = T.max(C_local[i, j] + bias[bx * block_N + j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


# ------------------------------------------------------------------ reduction


def row_sum(M, N, dtype="float32"):
    """Sum exactly N values per row using TileLang's initialized reduction."""
    padded_N = ((N + 127) // 128) * 128

    @T.prim_func
    def main(
        x: T.Tensor((M, N), dtype),
        out: T.Tensor((M,), dtype),
    ):
        with T.Kernel(M, threads=128) as bx:
            values = T.alloc_fragment((1, padded_N), dtype)
            total = T.alloc_fragment((1,), dtype)
            for i in T.Parallel(padded_N):
                values[0, i] = T.if_then_else(i < N, x[bx, i], 0)
            T.reduce_sum(values, total, dim=1, clear=True)
            T.copy(total, out[bx])

    return main


# --------------------------------------------------------------------- gather


def gather_rows(M, N, K, dtype="float16"):
    """Irregular gather: out[i, :] = x[index[i], :].

    Not a tile-friendly workload. Included to see how the IR and the
    diagnostics behave on control flow driven by loaded data.
    """

    @T.prim_func
    def main(
        index: T.Tensor((M,), "int32"),
        x: T.Tensor((N, K), dtype),
        out: T.Tensor((M, K), dtype),
    ):
        with T.Kernel(T.ceildiv(M, 16), T.ceildiv(K, 128), threads=128) as (bx, by):
            for i, j in T.Parallel(16, 128):
                row, col = bx * 16 + i, by * 128 + j
                if row < M and col < K:
                    src = index[row]
                    if src >= 0 and src < N:
                        out[row, col] = x[src, col]
                    else:
                        out[row, col] = 0

    return main


# ------------------------------------------------------------------ attention


def flash_attention(seq_len, heads, head_dim, block_M=32, block_N=64):
    """Non-causal attention: softmax(Q @ K.T / sqrt(head_dim)) @ V.

    Q/K/V/O have layout [heads, sequence, head_dim]. Online softmax keeps
    scores tile-local; head_dim and tiles must be multiples of 16.
    This is a correctness candidate, not a tuned performance baseline.
    """
    if min(seq_len, heads, head_dim, block_M, block_N) <= 0:
        raise ValueError("attention dimensions must be positive")
    if any(d % 16 for d in (head_dim, block_M, block_N)):
        raise ValueError("head dimension and attention tiles must be multiples of 16")
    scale = head_dim ** -0.5

    @T.prim_func
    def main(
        Q: T.Tensor((heads, seq_len, head_dim), "float16"),
        K: T.Tensor((heads, seq_len, head_dim), "float16"),
        V: T.Tensor((heads, seq_len, head_dim), "float16"),
        O: T.Tensor((heads, seq_len, head_dim), "float16"),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, threads=128) as (blk, h):
            query = T.alloc_shared((block_M, head_dim), "float16")
            key = T.alloc_shared((block_N, head_dim), "float16")
            value = T.alloc_shared((block_N, head_dim), "float16")
            probability = T.alloc_shared((block_M, block_N), "float16")
            scores = T.alloc_fragment((block_M, block_N), "float32")
            result = T.alloc_fragment((block_M, head_dim), "float32")
            maximum = T.alloc_fragment((block_M,), "float32")
            previous = T.alloc_fragment((block_M,), "float32")
            factor = T.alloc_fragment((block_M,), "float32")
            normalizer = T.alloc_fragment((block_M,), "float32")
            tile_sum = T.alloc_fragment((block_M,), "float32")
            T.copy(Q[h, blk * block_M:(blk + 1) * block_M, :], query)
            T.clear(result)
            T.clear(normalizer)
            T.fill(maximum, -T.infinity("float32"))
            for tile in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=1):
                T.copy(K[h, tile * block_N:(tile + 1) * block_N, :], key)
                T.gemm(query, key, scores, transpose_B=True, clear_accum=True,
                       policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, block_N):
                    scores[i, j] = T.if_then_else(tile * block_N + j < seq_len,
                                                 scores[i, j] * scale,
                                                 -T.infinity("float32"))
                T.copy(maximum, previous)
                T.reduce_max(scores, maximum, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    maximum[i] = T.max(previous[i], maximum[i])
                    factor[i] = T.exp(previous[i] - maximum[i])
                for i, j in T.Parallel(block_M, block_N):
                    scores[i, j] = T.exp(scores[i, j] - maximum[i])
                T.reduce_sum(scores, tile_sum, dim=1, clear=True)
                for i in T.Parallel(block_M):
                    normalizer[i] = normalizer[i] * factor[i] + tile_sum[i]
                for i, j in T.Parallel(block_M, head_dim):
                    result[i, j] *= factor[i]
                T.copy(scores, probability)
                T.copy(V[h, tile * block_N:(tile + 1) * block_N, :], value)
                T.gemm(probability, value, result, policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(block_M, head_dim):
                result[i, j] /= normalizer[i]
            T.copy(result, O[h, blk * block_M:(blk + 1) * block_M, :])

    return main


# ------------------------------------------------------------------- registry

# (name, callable) -- the harness lowers each across every requested target.
KERNELS = {
    "fused_elementwise": lambda: fused_elementwise(1024, 1024),
    "gemm_relu": lambda: gemm_relu(1024, 1024, 1024),
    "row_sum": lambda: row_sum(256, 1024),
    "gather_rows": lambda: gather_rows(1024, 1024, 128),
    "flash_attention": lambda: flash_attention(128, 8, 64),
}


def artifact_elementwise(size):
    """The opaque-artifact milestone: three float32 pointers, no shared memory."""
    if not isinstance(size, int) or not 1 <= size <= 2**31 - 1:
        raise ValueError("size must be a positive int32 extent")

    @T.prim_func
    def elementwise(
        a: T.Tensor((size,), "float32"),
        b: T.Tensor((size,), "float32"),
        c: T.Tensor((size,), "float32"),
    ):
        with T.Kernel(T.ceildiv(size, 128), threads=128) as block:
            for lane in T.Parallel(128):
                i = block * 128 + lane
                if i < size:
                    c[i] = T.max(a[i] * 2.0 + b[i], 0.0)

    return elementwise
