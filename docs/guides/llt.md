# LLT training and cached attention

The explicit workload API in `tensor_torch.llt` supplies Tensor CUDA kernels for
[loop latent transformer](https://github.com/kreasof-ai/loop-latent-transformer).
The [L40S qualification report](../research/llt-readiness.md) records supported
shapes, performance, and limitations. This is an LLT dependency profile, not a
claim of Tensor 1.0 or model quality.

```python
import torch
from tensor_torch.llt import Operators, AdamW, KVCache

ops = Operators(compute_dtype=torch.bfloat16)
q = torch.randn(1, 8, 129, 64, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
c = torch.randn(1, 1, 129, 64, device="cuda", dtype=torch.bfloat16,
                requires_grad=True)
y = ops.attention(q, c, c, causal=True, scale=32 ** -0.5)
y.float().square().mean().backward()
```

`scale` must use the original head dimension when projections are absorbed.
`query_offset` implements the mask `key_position <= query_offset + query_position`.
Attention accepts contiguous CUDA BHSD tensors, FP16 or BF16, shared or ordinary
KV heads, independent score/value dimensions, and positive extents. Score/value
dimensions are multiples of 16, bounded by 256/128; the measured rank sweep is
32/64/96/128. There is no dropout or general additive-mask profile. Unsupported
inputs raise rather than silently selecting Torch attention. Other GPU targets
need separate qualification; retained artifacts target L40S `sm_89`.

The operator supplies first-order custom autograd for attention, transposed and
batched GEMM, linear, RMSNorm, exact GELU, residual add, embedding, split-half RoPE,
and mean cross-entropy. Higher-order differentiation is unsupported. GEMMs use
FP16/BF16 operands and FP32 accumulation. FP32 master-weight matrix gradients are
rounded through the compute dtype, matching Torch autocast; optimizer moments,
softmax/normalization reductions, and residual state use FP32. `rotary` supports
absolute offsets as integers or nonnegative CUDA int64 batch vectors. Runtime
offset vectors can change during graph replay without a new specialization. Concatenating a separate positional query/key component while
keeping latent values realizes decoupled positional attention.

`AdamW(parameters, ops, max_norm=1.0)` requires contiguous dense FP32 weights and
gradients. It computes a global norm, clips, and skips the entire update and
step counter when gradients are nonfinite. Standard `state_dict` and parameter
group learning-rate changes work. `linear_cross_entropy(x, weight, labels,
chunk_size=32)` bounds live classifier logits by row chunking with recomputation
and in-place FP32 weight-gradient accumulation; it trades memory for time and supports ignored labels.

PyTorch owns storage, layout and slice copies, scalar/count control, autograd
scheduling, tied-gradient accumulation, exact
checkpoint scheduling, RNG, layouts, concatenation, optimizer parameter groups,
and serialization. These controls are not Tensor kernels. `ops.report` enumerates
actual Tensor artifacts, dispatch counts, layout copies, and semantic fallbacks.
Counts reflect Python calls and graph capture; graph replays are measured separately.
Read-only unaligned views may be copied to satisfy frontend alignment; mutable
buffers require their declared alignment and contiguity. The explicit API has no
semantic fallback. The separate generic `torch.compile(backend="tensor")` remains
a bounded adapter and does not automatically select this training/cache API.

## Persistent decode

```python
with torch.inference_mode():
    cache = KVCache(ops, 1, 1, 8192, 64, shared=True)
    cache.append(c.detach())
    new_latent = torch.randn(1, 1, 1, 64, device="cuda", dtype=torch.bfloat16)
    cache.append(new_latent)
    next_q = torch.randn(1, 8, 1, 64, device="cuda", dtype=torch.bfloat16)
    out = ops.decode(next_q, cache, scale=32 ** -0.5)
```

Storage is preallocated; append never reallocates or copies history. `shared=True`
requires K and V aliasing. Separate positional keys and latent values use separate
buffers; the demonstrated concatenated-key RoPE model duplicates the latent in
key/value storage and reports those bytes. `lengths` is a CUDA int32 vector for
variable-length batches. Regular append rejects capacity overflow; captured
append sets an overflow flag and prevents out-of-bounds writes. Call `check()`
after a replay sequence to synchronize host length and raise on overflow, then
`reset()` before reuse. Repeated decode is inference-only. A model must invalidate
cached folds and prefixes whenever weights change; the fixture checks versions.

Decode uses a 16-row tensor-core tile; automatic splitting selects 64 partitions
for capacities at least 32,768 and score dimensions up to 64, otherwise 32.
Explicit split counts 8/16/32/64/128/256 are available; tuning results apply to
the measured shapes rather than all configurations.

First use compiles and needs the pinned producer environment and NVRTC bundle.
Subsequent execution uses dependency-bound artifacts and prepared/native calls.
Warm every required shape before CUDA graph capture. Capture must preserve input
and cache lifetimes. Dynamic lengths live on the GPU; shape changes need another
specialization. Compiler-free consumers need prebuilt cache artifacts whose source
identity matches the installed wheels. The native executor is built for a specific
Torch minor; qualification uses Torch 2.14 and the matching extension. Other Torch
minors use the adapter's portable/Python execution paths unless separately built.

## Reproduce qualification

Use the locked producer environment, install `tensor-torch`, build its optional
native executor, and install NVRTC 12.9 as described in the [development guide](../development.md).
Run `bash benchmarks/llt/run.sh` on an otherwise idle L40S. Raw JSON stores all
samples, tolerances, coverage, hashes, and environment metadata. The sustained
training fixture is synthetic backend acceptance; quality/dataset studies belong
in the LLT repository.
