"""Independent, fixed-schedule Triton inference baselines for the direct benchmark.

Blocked GEMM and online-softmax attention follow the algorithms documented in
Triton's official matrix multiplication and fused attention tutorials. Tails,
BHSD storage, bias, and half-precision intermediates match our measured profiles.
This is benchmark code, not a new Tensor compiler backend.
"""

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

import torch
import triton
import triton.language as tl


@triton.jit
def pointwise(A, B, Out, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    a = tl.load(A+index, index<N, other=0)
    b = tl.load(B+index, index<N, other=0)
    value = a*2.0+b
    value = tl.where(value != value, value, tl.maximum(value, 0.0))
    tl.store(Out+index, value, index<N)


@triton.jit
def linear(A, W, Bias, Out, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
           RELU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    col = tl.program_id(1)*BN + tl.arange(0, BN)
    reduction = tl.arange(0, BK)
    accum = tl.full((BM, BN), 0, tl.float32)
    for tile in range(tl.cdiv(K, BK)):
        kk = tile*BK+reduction
        a = tl.load(A+row[:, None]*K+kk[None, :], (row[:, None]<M)&(kk[None, :]<K), 0)
        w = tl.load(W+col[None, :]*K+kk[:, None], (col[None, :]<N)&(kk[:, None]<K), 0)
        accum = tl.dot(a, w, accum)
    bias = tl.load(Bias+col, col<N, 0).to(tl.float32)
    value = (accum+bias[None, :]).to(tl.float16)
    if RELU:
        value = tl.where(value != value, value, tl.maximum(value, 0.0))
    tl.store(Out+row[:, None]*N+col[None, :], value, (row[:, None]<M)&(col[None, :]<N))


@triton.jit
def attention(Q, K, V, Out, S: tl.constexpr, D: tl.constexpr, CAUSAL: tl.constexpr,
              BM: tl.constexpr, BN: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    row = block*BM+tl.arange(0, BM)
    col = tl.arange(0, BN)
    dim = tl.arange(0, D)
    offset = head*S*D
    q = tl.load(Q+offset+row[:, None]*D+dim[None, :], row[:, None]<S, 0)
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    normalizer = tl.full((BM,), 0, tl.float32)
    result = tl.full((BM, D), 0, tl.float32)
    end = S
    if CAUSAL:
        end = tl.minimum(S, (block+1)*BM)
    for start in range(0, end, BN):
        key_row = start+col
        k = tl.load(K+offset+key_row[None, :]*D+dim[:, None], key_row[None, :]<S, 0)
        score = tl.dot(q, k)*D**-0.5
        valid = key_row[None, :]<S
        if CAUSAL:
            valid = valid & (key_row[None, :] <= row[:, None])
        score = tl.where(valid, score, float('-inf'))
        next_max = tl.maximum(maximum, tl.max(score, 1))
        factor = tl.exp(maximum-next_max)
        probability = tl.exp(score-next_max[:, None])
        normalizer = normalizer*factor+tl.sum(probability, 1)
        result = result*factor[:, None]
        v = tl.load(V+offset+key_row[:, None]*D+dim[None, :], key_row[:, None]<S, 0)
        result = tl.dot(probability.to(tl.float16), v, result)
        maximum = next_max
    result = result/normalizer[:, None]
    tl.store(Out+offset+row[:, None]*D+dim[None, :], result, row[:, None]<S)


class Operation:
    """One compiled Triton operation; caller chooses whether to allocate output."""
    def __init__(self, kind, args, *, relu=True, causal=False):
        self.kind, self.relu, self.causal = kind, relu, causal
        self.shape = tuple(args[0].shape)
        if kind == 'linear':
            self.shape = (args[0].shape[0], args[1].shape[0])
        self.dtype, self.device = args[0].dtype, args[0].device
        self.config = ({'block': 256, 'warps': 4} if kind=='pointwise' else
                       {'bm': 32, 'bn': 64, 'bk': 32, 'warps': 4, 'stages': 3} if kind=='linear' else
                       {'bm': 32, 'bn': 64, 'warps': 4, 'stages': 1})
        self.compiled = None
        self.runner = None

    def into(self, args, output):
        if self.kind == 'pointwise':
            self.compiled = pointwise[(triton.cdiv(output.numel(), 256),)](*args, output,
                N=output.numel(), BLOCK=256, num_warps=4, enable_fp_fusion=False)
        elif self.kind == 'linear':
            m,n = self.shape
            self.compiled = linear[(triton.cdiv(m,32),triton.cdiv(n,64))](*args, output,
                M=m,N=n,K=args[0].shape[1],RELU=self.relu,BM=32,BN=64,BK=32,
                num_warps=4,num_stages=3,enable_fp_fusion=False)
        else:
            batch,heads,length,dim = self.shape
            self.compiled = attention[(triton.cdiv(length,32),batch*heads)](*args, output,
                S=length,D=dim,CAUSAL=self.causal,BM=32,BN=64,
                num_warps=4,num_stages=1,enable_fp_fusion=False)
        return output

    def __call__(self, *args):
        output = torch.empty(self.shape,dtype=self.dtype,device=self.device)
        return self.into(args, output)

    def compiled_into(self, args, output):
        if self.runner is None:
            self.into(args,output)
            if self.kind=='pointwise':
                grid = (triton.cdiv(output.numel(),256),1,1)
                constants = (output.numel(),256)
            elif self.kind=='linear':
                m,n = self.shape
                grid = (triton.cdiv(m,32),triton.cdiv(n,64),1)
                constants = (m,n,args[0].shape[1],self.relu,32,64,32)
            else:
                batch,heads,length,dim = self.shape
                grid = (triton.cdiv(length,32),batch*heads,1)
                constants = (length,dim,self.causal,32,64)
            self.constants = constants
            self.runner = self.compiled[grid]
        self.runner(*args,output,*self.constants)
        return output

    def allocating_compiled(self,*args):
        output = torch.empty(self.shape,dtype=self.dtype,device=self.device)
        return self.compiled_into(args,output)
