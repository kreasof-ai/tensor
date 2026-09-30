# Phase 5 WebGPU implementation and hardware validation

**Phase 5 is complete.** The [Linux-to-Windows RX 6700 XT transfer](webgpu-rx6700xt-transfer.md)
passes all 33 inference/composition checks on physical AMD Vulkan hardware.
The strict audit was rerun against the original producer bundle and reports
`physical_second_gpu=true`, `two_hosts=true`, and `phase5_hardware_gate="passed"`.
The returned consumer result is timestamped 2026-09-30 18:35:37 UTC
(2026-10-01 in Asia/Jakarta). Source/artifact hashes match, compiler imports are
blocked, and adapter limits and measured timings are retained with the evidence.

**2026-10-01 update:** [Windows RX 6700 XT validation](webgpu-rx6700xt.md)
passes all 33 inference/composition checks on physical AMD hardware through
Vulkan in an isolated wheel consumer, plus 28 native-enabled contract/audit
tests with zero skips. Timings and raw evidence are recorded in that report.
D3D12 passes six FP32 affine checks before rejecting missing `shader-f16`.
The subsequent [AMD scaling sweep](latency-scaling.md#windows-rx-6700-xt-at-the-same-workload-sizes)
also passes all 17 A10G workload sizes and two repeats with matching WGSL hashes.
This initial AMD run built its suite locally and left the two-host transfer gate
open. The subsequent Linux-produced bundle run above closes that requirement.

2026-09-30 implementation baseline: the native WebGPU provider and agreed
inference lowering are implemented. Initial WebGPU validation used Mesa
llvmpipe; the same machine's A10G was subsequently enabled for Vulkan with
matching local NVIDIA driver libraries. The [matched A10G comparison](latency-scaling.md)
passes all 17 scaling workloads and two repeats on physical hardware. That
same-device comparison leaves the separate AMD/Apple portability gate open.

The full local suite passed **164 tests, zero skips, in 184.75 seconds**, with
CUDA, native Torch execution, direct Triton controls and WebGPU enabled. Nine
additional evidence-audit tests and one packed-dispatch regression passed after
that run. No CUDA kernel schedules or NVRTC default changed.

The subsequent A10G comparison added explicit large-buffer negotiation and
revalidated **180 unique checks**: 128 passed with native WebGPU enabled,
and all 52 CUDA opt-in checks passed in a separate run. No unchecked cases
remained. The [scaling evidence](data/latency-scaling.json) records that
regression summary, 17 same-input CUDA/Vulkan profiles and two matched repeats.

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
software consumer. That [workflow passed](https://github.com/kreasof-ai/tensor/actions/runs/36740570767)
on source `1d21ce91193b4f01dbf9c58a5034b786ac28f6d3`: both GPU-free producers,
all 33 Windows-to-Linux consumer checks and the evidence audit succeeded. See
[CI metadata](data/webgpu-ci.json) and the [transfer result](data/webgpu-transfer-windows-linux.json).
The existing [NVRTC/Torch workflow also passed](https://github.com/kreasof-ai/tensor/actions/runs/36739977375)
on Linux and Windows for the implementation's source parent; the follow-up only
normalized transfer-manifest paths. Its [metadata](data/webgpu-nvrtc-regression-ci.json)
records both jobs. Physical AMD Vulkan execution and a latency baseline are now
recorded in the [RX 6700 XT report](webgpu-rx6700xt.md). The subsequent
[physical AMD two-host transfer](webgpu-rx6700xt-transfer.md) passes the strict
audit and completes Phase 5 acceptance.
