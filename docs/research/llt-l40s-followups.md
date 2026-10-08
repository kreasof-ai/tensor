# Tensor follow-ups from the L40S study

These entries retain the initial study findings. The [Tensor LLT qualification](llt-readiness.md)
records the subsequent BF16, backward, cache, and adapter implementation.
The NVRTC integration also supplies the integral trait missing from the pinned
TileLang standard shim; real CUDA warp-reduction regression coverage passes. The
TileLang reduced-fragment limitation remains pinned and uses the tested
shared-memory extraction workaround.

For the complete dependency order and acceptance gates, see the
[Tensor readiness plan](../plan/llt-readiness.md).

The source evidence is the [LLT L40S report](https://github.com/kreasof-ai/loop-latent-transformer/blob/main/benchmarks/L40S.md).

No modification to [Tensor](https://github.com/kreasof-ai/tensor) source was needed for the completed forward kernels.
The checkout's pinned environment and NVRTC bundle were installed locally.

## 1. Reduced-fragment scalar extraction: pinned TileLang limitation

The first partitioned decode implementation exported `maximum[0]` and one row
of the output fragment directly after GEMM/row reduction. Tensor's pinned
TileLang 0.1.14 rejects the inferred fragment layout. The reduced fragment is
replicated across lanes, while scalar indexing constrains a conflicting layout.

Reproduce from the Tensor repository root with a fresh output filename:

```bash
TENSOR_NVRTC_HOME=build/nvrtc-12.9 .venv/bin/tensor build \
  benchmarks/llt/tensor_layout_repro.py --out /tmp/llt-layout-repro-new.tbin
```

Observed build failure:

```text
TileLang lowering failed: InternalError:
Layout may conflict with ReduceOp for buffer maximum vs. scores
...
You may need to use a shared memory to transform the layout
```

The standalone reproducer passed construction but failed lowering at the pinned
Tensor commit recorded in the reports. This is a limitation in the compiler
dependency; the evidence does not identify a Tensor runtime defect. The working
LLT kernel copies the fragment into shared memory and then exports row zero.
Useful Tensor work would add a regression/documented extraction pattern, improve
the diagnostic, or evaluate an upstream frontend fix. The workaround increases
shared-memory traffic and scratch and is a candidate for optimization.

## 2. Optimized shared-latent attention backward

The current Tensor Torch adapter documents forward self-attention and experimental
partial training coverage; it does not supply a FlashAttention backward. The
manual `tensor-nn` training kernels are a separate bounded template. They do not
provide an optimized absorbed, shared-cache LLT attention backward through this
adapter.

For a fully Tensor-backed LLT training study, add a numerically validated CUDA
forward/backward operator for shared K=V latents. It needs causal masks, explicit
original-head scaling, ranks 32/64/96/128, dQ and a sum of both dK/dV contributions
over heads, current-stream launches, and autograd registration compatible with
exact checkpoint recomputation. Compare its allocator footprint and step time
with Flash SDPA. This study uses PyTorch Flash backward instead; forward inference
results do not establish Tensor training speed.

## 3. Adapter coverage for cached decode and shared KV

The generic `torch.compile(backend='tensor')` SDPA profile excludes cached decode
and GQA/shared-head layouts. This study bypasses that bounded profile by loading
custom `.tbin` kernels through `tensor_torch.load`, which works on the L40S.
Promoting the tested shared-KV prefill and partitioned decode kernels into a
supported adapter profile would make the implementation reusable without custom
exports. Validate query/KV length differences, KV aliasing, odd tails, batch and
the partition merge, and preserve semantic fallback for unsupported masks.

The kernels in the [LLT study](https://github.com/kreasof-ai/loop-latent-transformer/tree/main/experiments/l40s) can serve as a starting point for that work. Native
L40S execution passed; other SM targets require separate builds and validation.

## 4. BF16 ABI and DLPack interoperability

Tensor's current dtype enumeration has FP16/FP32/FP64 but no BF16. Importing a
CUDA BF16 Torch tensor through DLPack fails, independently of any kernel. This
matters if Tensor kernels are to replace the BF16 Flash operations used in this
study's GPU training configuration.

```bash
.venv/bin/python benchmarks/llt/tensor_bf16_repro.py
```

The observed exception is `BufferError: unsupported DLPack element dtype`. Add an explicit
BF16 storage type consistently through the ABI/manifest, DLPack importer, Torch
bridge and compiler validation; preserve existing numeric dtype identities and
add forward/backward numerical coverage. FP16 inference is unaffected. This is a
feature gap, rather than evidence of incorrect existing FP16 behavior.
