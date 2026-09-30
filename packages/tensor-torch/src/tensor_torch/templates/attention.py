"""FP16 self-attention with online softmax and no global score matrix.

Inputs/outputs are contiguous [batch, heads, sequence, head_dim]. This forward
profile supports causal or non-causal attention, no mask/dropout/GQA. Dimensions
are compile-time specializations; the demo specializes this same source.
"""
import tilelang.language as T

BATCH = 1
HEADS = 8
SEQ_LEN = 128
HEAD_DIM = 64
IS_CAUSAL = False
BLOCK_M = 32
BLOCK_N = 64
SCALE = HEAD_DIM ** -0.5


@T.prim_func
def flash_attention(
    q: T.Tensor((BATCH, HEADS, SEQ_LEN, HEAD_DIM), "float16"),
    k: T.Tensor((BATCH, HEADS, SEQ_LEN, HEAD_DIM), "float16"),
    v: T.Tensor((BATCH, HEADS, SEQ_LEN, HEAD_DIM), "float16"),
    out: T.Tensor((BATCH, HEADS, SEQ_LEN, HEAD_DIM), "float16"),
):
    with T.Kernel(T.ceildiv(SEQ_LEN, BLOCK_M), HEADS, BATCH, threads=128) as (blk, h, b):
        query = T.alloc_shared((BLOCK_M, HEAD_DIM), "float16")
        key = T.alloc_shared((BLOCK_N, HEAD_DIM), "float16")
        value = T.alloc_shared((BLOCK_N, HEAD_DIM), "float16")
        probability = T.alloc_shared((BLOCK_M, BLOCK_N), "float16")
        scores = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
        result = T.alloc_fragment((BLOCK_M, HEAD_DIM), "float32")
        maximum = T.alloc_fragment((BLOCK_M,), "float32")
        previous = T.alloc_fragment((BLOCK_M,), "float32")
        factor = T.alloc_fragment((BLOCK_M,), "float32")
        normalizer = T.alloc_fragment((BLOCK_M,), "float32")
        tile_sum = T.alloc_fragment((BLOCK_M,), "float32")
        T.copy(q[b, h, blk * BLOCK_M:(blk + 1) * BLOCK_M, :], query)
        T.clear(result)
        T.clear(normalizer)
        T.fill(maximum, -T.infinity("float32"))
        tiles = T.ceildiv(T.min(SEQ_LEN, (blk + 1) * BLOCK_M), BLOCK_N) if IS_CAUSAL else T.ceildiv(SEQ_LEN, BLOCK_N)
        # Query-dependent causal trip counts need a serial schedule: the pinned
        # frontend's software pipeline emitted incorrect short-block epilogues.
        for tile in T.serial(tiles):
            T.copy(k[b, h, tile * BLOCK_N:(tile + 1) * BLOCK_N, :], key)
            T.gemm(query, key, scores, transpose_B=True, clear_accum=True,
                   policy=T.GemmWarpPolicy.FullRow)
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                scores[i, j] = T.if_then_else(
                    (tile * BLOCK_N + j < SEQ_LEN)
                    & ((not IS_CAUSAL) | (tile * BLOCK_N + j <= blk * BLOCK_M + i)),
                    scores[i, j] * SCALE, -T.infinity("float32"))
            T.copy(maximum, previous)
            T.reduce_max(scores, maximum, dim=1, clear=True)
            for i in T.Parallel(BLOCK_M):
                maximum[i] = T.max(previous[i], maximum[i])
                factor[i] = T.exp(previous[i] - maximum[i])
            for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                scores[i, j] = T.exp(scores[i, j] - maximum[i])
            T.reduce_sum(scores, tile_sum, dim=1, clear=True)
            for i in T.Parallel(BLOCK_M):
                normalizer[i] = normalizer[i] * factor[i] + tile_sum[i]
            for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
                result[i, j] *= factor[i]
            T.copy(scores, probability)
            T.copy(v[b, h, tile * BLOCK_N:(tile + 1) * BLOCK_N, :], value)
            T.gemm(probability, value, result, policy=T.GemmWarpPolicy.FullRow)
        for i, j in T.Parallel(BLOCK_M, HEAD_DIM):
            result[i, j] /= normalizer[i]
        T.copy(result, out[b, h, blk * BLOCK_M:(blk + 1) * BLOCK_M, :])


def tensor_export():
    return {"kernel": flash_attention, "outputs": ["out"]}
