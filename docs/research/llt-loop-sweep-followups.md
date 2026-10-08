# Tensor follow-ups from the LLT loop sweep — 2026-10-08

The [LLT four-model L40S sweep](https://github.com/kreasof-ai/loop-latent-transformer/blob/main/benchmarks/L40S_LOOP_SWEEP.md)
profiles batch 4, sequence 1024, width 768, 12 heads, and loop counts 1–16.
The fixed-depth control keeps 12 blocks and widens its MLPs to exactly match
12T independent conventional blocks. Its hidden width reaches 72,192 at T=16.
The measured Tensor implementation is pinned to `a86159c`; native Torch 2.14
executor/launch binaries match the earlier qualified runtime. Numerical kernels
were unchanged throughout the sweep.

No new functional Tensor feature was needed to run this grid. Eight small
output/all-parameter-gradient/cache checks passed; full training verifies every
parameter gradient is finite. The study audits 544 CUDA artifact hashes and
reports no numerical fallback for successful Tensor cases. Performance remains
geometry dependent. These are untrained synthetic resource measurements.

## Wide MLP schedules

At T=16 the exact parameter-matched fixed-depth model has 1,437,008,640 active
parameters. Tensor cached decode takes 9.56 ms versus Torch's 5.04 ms. Its Tensor
full-step training graph takes 513.00 ms / 31.32 GiB. Torch, with eager gradients
released before capture, takes 346.83 ms / 35.22 GiB. By comparison, recurrent
LLT's cached decode favors Tensor: 8.26 versus 13.83 ms. This establishes a wide
MLP performance follow-up, rather than a universal GEMV improvement claim.

Extend isolated numerical qualification and actual schedule search to these
BF16 shapes; attribute time before changing defaults:

| Projection | M | N | K | Layout |
|---|---:|---:|---:|---|
| MLP input, prompt/training | 4096 | 72192 | 768 | transpose B |
| MLP output, prompt/training | 4096 | 768 | 72192 | transpose B |
| Input projection dW | 72192 | 768 | 4096 | transpose A |
| Output projection dW | 768 | 72192 | 4096 | transpose A |
| MLP input, cached decode | 4 | 72192 | 768 | transpose B |
| MLP output, cached decode | 4 | 768 | 72192 | transpose B |

Also cover intermediate hidden widths, especially the observed T=15–16 latency
increase, and dX layouts. A segmented/split-K GEMV or another tiny-M schedule is
an optimization candidate for the long-K output projection. The full-model
measurements do not yet isolate which projection or scheduling limit explains
the gap. Preserve FP32 accumulation/BF16 rounding, oracle checks, legal sm89
shared-memory limits, and real beam search when evaluating replacements.

## Training activation retention

LLT training's Tensor allocation advantage crosses over at T=9. At T=16,
Tensor LLT graph training takes 411.77 ms / 25.11 GiB versus Torch's 350.49 ms /
23.47 GiB. Its persistent latent cache remains constant; full residual, norm,
MLP and autograd storage do not.

[`_Matmul.forward`](https://github.com/kreasof-ai/tensor/blob/a86159c/packages/tensor-torch/src/tensor_torch/llt.py#L465)
saves original inputs with `ctx.save_for_backward(x, y)`. LLT normalization
outputs are FP32, while GEMM computes in BF16; backward recasts the saved inputs.
Saving compute-precision activation inputs with original dtype/shape metadata
is a candidate to reduce retention and repeated conversion. This code inspection
is not a complete allocation attribution. Do not retain a fresh BF16 copy of
every master weight per loop as an incidental consequence: that could increase
memory. Preserve mutation/version guards, all transpose gradients, tied-gradient
accumulation and captured training semantics, and compare against Torch before
claiming an improvement.

## Capture setup is a separate constraint

The primary sweep retains successful eager measurements even when capture OOMs.
Six controlled retries release eager gradients and cached allocations after
graph warmup, before creating private graph storage; five then pass. Examples:

- The 12-layer fixed-depth Torch T=16 case releases 5.36 GiB and captures at
  35.22 GiB peak allocation.
- Tensor's independent stacks recover T=14 and T=15 graph training at 36.98 and
  39.50 GiB; T=16 still capture-OOMs after cleanup, but eager training completes
  at 42.02 GiB / 853.31 ms.
- Torch independent-stack eager training OOMs at T=15–16 under the tested
  allocator configuration. Its cleaned captures recover T=13–14.

These are allocator/graph-lifetime observations, not evidence of a Tensor kernel
correctness bug. Do not silently reduce batch/context or add checkpoints to
replace failed cases. Capture peak allocation, reserved memory, CUDA OOM chains,
cleanup protocol, and eager-versus-captured operation must stay explicit.

Raw results, archived exception chains, source snapshots, and controlled retry
records are in the [LLT results repository](https://github.com/kreasof-ai/loop-latent-transformer/tree/main/benchmarks/results/l40s-loop-sweep).
Tensor implementation work and development records belong in this repository.
