# Phase 5 WebGPU implementation and open hardware gate

2026-09-30. The native WebGPU provider and agreed inference lowering are
implemented. **Phase 5 is not yet accepted:** the user will run the transferred
suite on a physical AMD or Apple GPU. The current machine exposes an A10G for
CUDA and only Mesa llvmpipe for WebGPU; software Vulkan cannot close that gate.

The full local suite passed **164 tests, zero skips, in 184.75 seconds**, with
CUDA, native Torch execution, direct Triton controls and WebGPU enabled. Nine
additional evidence-audit tests and one packed-dispatch regression passed after
that run. No CUDA kernel schedules or NVRTC default changed.

An isolated environment installed the built Tensor wheel, NumPy 2.5.3 and wgpu
0.29.0, with native wgpu 27.0.2.0. Its six distributions were Tensor, NumPy, wgpu,
CFFI, pycparser and rendercanvas. The consumer blocked imports of TileLang, TVM,
TVM-FFI, Torch and Triton; no compiler/framework import occurred. Packaged-module
restoration and all **33 inference/composition checks** passed. The
[raw result](data/webgpu-software.json) records source/artifact/package hashes,
adapter capabilities, tolerances, errors and timing metadata. The evidence audit
passes software validation and explicitly reports the physical hardware gate open.

The suite covers symbolic FP32 affine lengths 1/127/128/129/4097/1M, FP16
elementwise tails, FP16/FP32 linear/GEMM with bias/ReLU/transpose-B and M/N/K
tails, dynamic GEMM rows 1/31/32/33/65, square GEMM 256/512, a device-resident
two-layer MLP, and batched causal/noncausal FP16 attention with D=64/128 and
sequence lengths 65/129/512. Attention uses the existing streaming online-softmax
source with WebGPU tile sizes 8×16, not a global dense score matrix.

| Software Vulkan case | Median completed call | Maximum absolute error |
|---|---:|---:|
| Affine, 1M | 3.305 ms | 0 |
| FP16 GEMM, 256 square | 25.743 ms | 0.0078125 |
| FP16 GEMM, 512 square | 201.058 ms | 0.03125 |
| Attention, D64/S512 noncausal | 76.703 ms | 0.0001221 |
| Attention, D128/S512 noncausal | 188.799 ms | 0.0002441 |

These are CPU software-adapter timings, with five samples after two warmups.
They include host submission and queue completion, use preallocated outputs,
and exclude upload/download and cold pipeline creation. They establish execution
and benchmark plumbing; they make no claim about AMD/Apple performance or parity
with CUDA, native TileLang, Triton or optimized attention libraries. Numerical
tolerances are recorded per case; absolute GEMM error alone does not express the
relative error at large output magnitudes.

Implementation details and the transfer command are in the [WebGPU guide](../webgpu.md).
The acceptance bundle contains the Tensor wheel, original-TIRx WGSL artifacts,
module closure, consumer script and evidence auditor. The dedicated workflow
builds on Linux/Windows and executes Windows-produced WGSL on a separate Linux
software consumer. Physical GPU results and workflow outcomes must be recorded
before marking the corresponding gates passed.
