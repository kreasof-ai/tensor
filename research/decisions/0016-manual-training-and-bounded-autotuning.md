# ADR 0016: Manual training templates and bounded autotuning

Status: accepted for the Phase 6 nanoGPT profile.

## Decision

Deliver a complete CUDA training workload before a general tensor algebra or
standalone autograd engine. Expose `ManualFunction` and `BackwardContext` as
framework-independent contracts, and implement an explicit forward/reverse tape
for ten nanoGPT updates. Consumers use Tensor and NumPy, with every arithmetic
operation executed by packaged kernels. PyTorch is an independent development
oracle and performance control.

Compile ordinary TileLang/TIRx kernel templates through Tensor's NVRTC producer
and existing module/artifact contracts. Templates include backward, clipping and
AdamW. Fuse MLP projection/GELU and projection/residual epilogues while preserving
the specified FP16 rounding boundaries. Reuse static buffers and prepared calls;
retain session/buffer lifetime checks. Avoid claims of arbitrary graph fusion.

TileLang already offers configuration search, validation and profiling through
its [autotuning API](https://www.tilelang.com/programming_guides/autotuning.html).
This profile uses a small explicit harness over Tensor NVRTC artifacts so tuning
measures the distribution/runtime path that consumers actually execute. Search
three tile/pipeline schedules for each of 18 GEMM or fused-GEMM shapes. Check
outputs, including saved GELU preactivations, against the default Tensor schedule
before timing; independently validate the selected training path against Torch.
Retain candidates, measurements and selected schedules in the producer manifest.
CUDA-event intervals can include host submission gaps; they are measurements of
this execution path rather than fully isolated shader costs.

## Contract and scope

Manual contexts hold saved buffers until consumption or discard. One outstanding
context per operation protects reused workspace. Gradient shape, dtype, session,
liveness and integer-input rules are validated. A callback failure consumes the
context to avoid repeating partial accumulation. The caller must preserve saved
contents and explicitly handle branches, accumulation and recomputation.

The initial profile is static, contiguous, CUDA-only and uses dense causal
attention, FP16 compute and FP32 master/state buffers. Artifacts retain portable
IR/source for explicit rebuilding. Exact runtime source fingerprints bind this
experimental training bundle to its wheel, in addition to existing artifact ABI
and device compatibility checks. NVRTC remains the default; direct PTX remains
experimental after v1.

Standalone autograd, arbitrary user graph fusion, generic `nn` layer coverage,
FlashAttention backward and WebGPU training are separate extensions. This does
not expand Phase 4's inference-first Torch adapter contract. Numerical acceptance
compares all parameter gradients independently, isolates optimizer arithmetic on
identical gradients, and checks an independent ten-update eager loss trajectory.
Performance includes native SDPA and fused AdamW controls; no universal speed
gate or convergence claim follows from this workload.

Evidence and exact limitations are in the [Phase 6 report](../research/phase6-nanogpt.md).
