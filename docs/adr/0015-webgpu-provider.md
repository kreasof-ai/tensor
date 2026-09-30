# ADR 0015: Portable WebGPU inference provider and opaque buffers

Status: accepted implementation direction, 2026-09-30. Physical AMD Vulkan
execution [verified on Windows RX 6700 XT](../research/webgpu-rx6700xt.md),
2026-10-01; the two-host AMD/Apple acceptance transfer remains open.

## Decision

Use native wgpu for the second physical GPU provider. Preserve original frontend
TIRx in each artifact and emit WGSL through TileLang's existing WebGPU pipeline.
Tensor maintains a producer-only SIMT expansion for ordinary GEMM and two-axis
sum/max reductions. It materializes fragments as workgroup storage, accumulates
dot products/reductions in thread-local FP32 storage, serializes software-pipeline
schedules and inserts workgroup barriers at tile boundaries. Shared dynamic
buffers become typed static WGSL allocations. Batch/head launches flatten onto
the y axis because TileLang reserves z for large x grids.

This extends lowering of the actual tile operations and composes with the
existing copy/fill/layout/flatten/codegen passes. It does not select handwritten
WGSL based on recognizing an entire GEMM or attention kernel. The original IR is
retained for retargeting and the extension's source hash identifies the producer.
The extension can be proposed upstream independently of Tensor's runtime/envelope.

Portable SIMT kernels establish coverage and provider correctness. They do not
claim tensor-core performance, native FlashAttention parity or complete TileLang
support. Inference includes elementwise, FP16/FP32 GEMM/linear, bias/ReLU and MLP
chains, and FP16 forward attention with causal/noncausal masking, batching,
sequence tails and head dimensions 64/128. Small WebGPU tile schedules bound
workgroup memory. Larger CUDA schedules may exceed this profile and are rejected
explicitly instead of silently changing their algorithm or falling back to CUDA.

Add ABI minor 2 with argument kind 3 for opaque, session-owned buffers. Keep raw
addresses zero and put the buffer token in the explicitly tagged payload. Keep
all existing CPU/CUDA layouts and minor-1 producer requirements. Validate logical
shapes, buffer lifetime, typed scalar payloads, entrypoints, binding order, FP16
features and device limits before dispatch.

Use the optional `webgpu` extra, pinned wgpu-py 0.29.0 with its bundled native
wgpu 27.0.2.0. The native library owns Vulkan/Metal/DX12 shader translation and
pipeline creation. Consumers need no TileLang, TVM, CUDA toolkit, NVRTC or
PyTorch. Core Tensor remains NumPy-only. Runtime WGSL compilation is part of
wgpu/driver loading; WGSL artifacts are not CUDA-style machine-code images.

## Acceptance

GPU-free producers, opaque-handle contract tests, module transfer, import-guarded
clean consumers and the inference suite must pass. Software Vulkan is useful CI
evidence, but Phase 5 completes only after the same suite's transferred artifacts
execute on a real AMD or Apple GPU with recorded adapter capabilities and timings.
Kernel tuning and accelerated subgroup/matrix paths require separate measurements;
they are not implied by functional acceptance.

CUDA remains on NVRTC; direct PTX stays experimental after version 1. PyTorch's
CUDA storage/native executor does not gain WebGPU interop through this decision.
