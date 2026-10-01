# Larger-shape latency scaling

All baselines now execute the **same operations, shapes, dtypes and input
values on the physical NVIDIA A10G**. Tensor wgpu uses Vulkan; the other
providers use CUDA. CPU software-adapter timings are excluded from this
comparison. The 17-profile sweep and two targeted repeats passed numerical
checks, including WebGPU GEMM 4096³ and attention S=8192.

A subsequent [Windows RX 6700 XT sweep](#windows-rx-6700-xt-at-the-same-workload-sizes)
passes all 17 sizes with byte-identical WebGPU shaders and the same allocating-call
timing protocol. Its input values, operating system, host and driver differ;
the cross-system WebGPU comparison is reported separately below.

The outer CUDA `torch.compile` wrapper becomes a smaller fraction of latency
as GPU work grows. At 64M pointwise elements, Tensor through `torch.compile`
is **1.02×** matched TileLang. At GEMM 4096³ the ratio is
1.01×, but both remain slower than Triton and eager PyTorch.
The current portable wgpu lowering is slower still: 4.92× Tensor through
`torch.compile` for 64M pointwise, 14.68× for GEMM 4096³, and
64.44× for non-causal S=8192 attention.

Measured on 2026-09-30 with driver 595.91.07, the C++ Torch executor and the
pinned environment from the [direct comparison](direct-backend-comparison.md),
plus wgpu 0.29.0 / wgpu-native 27.0.2.0. NVIDIA's matching Vulkan libraries
were extracted locally to enable the existing compute-only container.

![Matched A10G latency scaling and full Tensor call ratios](data/latency-scaling.png)

[SVG figure](data/latency-scaling.svg).
[Full raw observations, repeat, numerical errors and fingerprints](data/latency-scaling.json).
[Scaling benchmark](../../benchmarks/inference/scaling_backend_benchmark.py).
[WebGPU workload preparation](../../benchmarks/inference/webgpu_scaling.py).
[Plot exporter](../../scripts/plots/plot_latency_scaling.py).

## Individual-call latency

Every table reports **one warm allocating call followed by completion
synchronization**, in milliseconds. Each median uses 45 individual samples,
after 20 warmups, with rotating provider order. Output allocation is inside
the timer and output destruction is outside it. Input upload/download, AOT
compilation and cold pipeline creation are excluded. The same input tensors
are copied into WebGPU storage before measurement; their hashes are recorded.

| Baseline | A10G API | Entry point |
|---|---|---|
| Tensor through torch.compile | CUDA | PyTorch adapter with native C++ executor |
| Direct Tensor C++ (figure/raw data) | CUDA | Native executor without outer torch.compile wrapper |
| Matched TileLang | CUDA | Native TVM-FFI, identical Tensor cubin |
| Triton | CUDA | Compiled-kernel runner |
| Eager PyTorch | CUDA | Eager operators / library kernels |
| Tensor wgpu | Vulkan | Tensor's native WebGPU allocating API and owned queue |

WebGPU uses its own allocator and queue-completion operation. It has no
PyTorch integration in this measurement. These are comparable completed-call
latencies for the same workload, rather than isolated shader timings or
identical host wrappers. CUDA-only host batches and graph timings are retained
in the raw data, alongside ordinary NVCC TileLang, warmed Triton JIT and Inductor.

| FP32 pointwise elements | Tensor through torch.compile | Matched TileLang | Triton | Eager PyTorch | Tensor wgpu / Vulkan | Tensor CUDA / TileLang |
|---|---:|---:|---:|---:|---:|---:|
| 129 | 0.060 | 0.018 | 0.027 | 0.033 | 0.563 | 3.25× |
| 1M | 0.074 | 0.041 | 0.048 | 0.061 | 0.443 | 1.80× |
| 4M | 0.152 | 0.121 | 0.127 | 0.265 | 1.444 | 1.26× |
| 16M | 0.469 | 0.432 | 0.439 | 1.006 | 2.591 | 1.09× |
| 64M | 1.680 | 1.641 | 1.668 | 3.951 | 8.267 | 1.02× |

Here M means 2²⁰ elements. Every provider computes `relu(a*2+b)`.
The small anchor varies between runs: its repeat measured
0.085 ms for Tensor through `torch.compile`,
0.024 ms for TileLang and
0.595 ms for wgpu. Both observations are retained;
hollow plot markers show the repeat. A smaller input is not guaranteed to
have a lower observed host-call latency in these separate samples.
At 64M, wgpu's 8.267 ms is 2.09× eager PyTorch.

| FP16 linear + bias + ReLU, M=N=K | Tensor through torch.compile | Matched TileLang | Triton | Eager PyTorch | Tensor wgpu / Vulkan | Tensor CUDA / TileLang |
|---|---:|---:|---:|---:|---:|---:|
| 512 | 0.062 | 0.027 | 0.033 | 0.036 | 0.859 | 2.32× |
| 1024 | 0.108 | 0.073 | 0.074 | 0.073 | 2.993 | 1.48× |
| 2048 | 0.504 | 0.470 | 0.373 | 0.366 | 20.285 | 1.07× |
| 4096 | 10.629 | 10.549 | 4.635 | 2.352 | 156.017 | 1.01× |

All providers compute `relu(linear(a, w, bias))`, including bias/ReLU at every
shape. At 4096, Tensor through `torch.compile` is
2.29× slower than Triton and
4.52× slower than eager PyTorch.
Wgpu measures 156.017 ms, or
66.34× eager PyTorch. The targeted repeat measured
10.623, 10.544,
4.428, 2.364 and
155.991 ms for Tensor, TileLang, Triton, eager and wgpu
respectively. The large-GEMM behavior persists across those runs.

| FP16 attention, B=1 H=8 D=64 | Tensor through torch.compile | Matched TileLang | Triton | Eager PyTorch | Tensor wgpu / Vulkan | Tensor CUDA / TileLang |
|---|---:|---:|---:|---:|---:|---:|
| S=1024, non-causal | 0.122 | 0.086 | 0.084 | 0.074 | 3.690 | 1.42× |
| S=2048, non-causal | 0.276 | 0.238 | 0.220 | 0.205 | 13.828 | 1.16× |
| S=4096, non-causal | 0.809 | 0.770 | 0.699 | 0.716 | 49.585 | 1.05× |
| S=8192, non-causal | 2.993 | 2.876 | 2.695 | 2.509 | 192.876 | 1.04× |
| S=1024, causal | 0.096 | 0.063 | 0.075 | 0.074 | 2.605 | 1.52× |
| S=2048, causal | 0.195 | 0.159 | 0.164 | 0.210 | 8.449 | 1.22× |
| S=4096, causal | 0.508 | 0.464 | 0.442 | 0.484 | 26.932 | 1.09× |
| S=8192, causal | 1.675 | 1.605 | 1.464 | 1.479 | 100.019 | 1.04× |

Every provider uses B=1/H=8/D=64 at all four sequence lengths, with no
dropout or custom mask. Wgpu measures 192.876 ms for S=8192
non-causal attention and 100.019 ms for causal attention:
76.86× and 67.61× eager PyTorch respectively.

The top plot panels include wgpu medians and interquartile bands for every
matched shape. Bottom panels include Tensor CUDA / wgpu ratios as well as
CUDA baselines, using a logarithmic scale so the larger gaps remain visible.
Ratios below one favor Tensor through `torch.compile`.

## WebGPU lowering and correctness

The workloads match, while schedules remain backend-specific. Wgpu GEMM uses
32×32 output tiles, K=16 and scalar FP32 accumulation. Attention retains the
streaming online-softmax algorithm with 8×16 query/key tiles and FP32
accumulators. There is no vendor matrix-instruction lowering or autotuning.
Large CUDA schedules cannot simply be copied when they exceed WebGPU's
workgroup-storage limits.

All 17 WebGPU outputs passed against eager PyTorch with `atol=0.002`,
`rtol=0.02`, matching the benchmark's CUDA acceptance tolerances. Pointwise
outputs were exact. Per-case maximum errors and artifact/input hashes are
retained in the raw data; FP16 error at large output magnitudes is assessed
with both relative and absolute tolerances. The repeat reused identical
input values and WGSL hashes.

The scalar shader schedules and WebGPU allocation/submission path differ
from the optimized CUDA implementations. These completed-call measurements
do not isolate their individual costs. No wgpu GPU timestamp measurement is
reported, and CUDA graph timings below must not be interpreted as wgpu time.

## Host overhead versus GPU execution

Excluding the noisy 129-element anchor, the CUDA host submission difference
between `tensor_compile` and `tensor_direct` has a median of
**46 µs**, ranging from 43 to 48 µs.
Their serialized-call difference has a median of 39 µs, but individual
cases have substantial timing variation, including negative differences.
These differences between independent sample medians are not an exact
additive decomposition of CPU and GPU work.

For 64M pointwise elements, captured direct Tensor GPU time is about
1.616 ms; for 1M it is about
26 µs. The relative cost of the outer wrapper shrinks
as GPU work grows. The CUDA GEMM 4096 slowdown remains when host dispatch
is removed:

| Captured CUDA GPU time, GEMM 4096³ | Main sweep | Targeted repeat |
|---|---:|---:|
| Direct Tensor | 10.501 ms | 10.501 ms |
| Matched TileLang | 10.477 ms | 10.516 ms |
| Triton compiled launcher | 4.484 ms | 4.353 ms |
| Eager PyTorch | 2.357 ms | 2.359 ms |

Direct Tensor captured GPU time increases about 23.1× from GEMM 2048³
to 4096³ for 8× the arithmetic work. Identical-cubin TileLang reproduces the
large-GEMM slowdown. Improving only the outer wrapper cannot resolve it;
these measurements do not identify the precise instruction or memory bottleneck.

## Method and reproduction

The sweep uses seed 42, contiguous inputs, inference mode and a non-default
CUDA stream. Every provider executes the same pointwise, fused linear or BHSD
attention operation. Backend schedules are fixed throughout the sweep.
The wgpu adapter must be a discrete GPU whose name matches CUDA's selected
GPU; a software-adapter result cannot enter the comparison.

All CUDA allocating and fixed-output graphs are checked against eager;
matched TileLang outputs are bitwise equal to Tensor. Both `torch.compile`
Tensor and Inductor outputs are checked. All 34 matched cubin variants in
the sweep and four in the repeat have verified binary hashes, argument ABI
and launch metadata. WebGPU uses copies of the same input tensors and records
19 numerical checks across the sweep and repeat, plus WGSL/source hashes.

CUDA host measurements use nine batches of 20 calls after 100 warmups;
the completed-batch variant synchronizes once per batch. Serialized timing
rotates all nine providers across nine batches of five calls, after 20
warmups. It synchronizes the appropriate stream/queue after each call and
releases each output outside the timer. CUDA GPU timing captures ten calls
and uses nine CUDA-event samples with warmed graphs. The tiny anchor's
CUDA event timing includes appreciable graph-launch/event overhead.

The WebGPU session explicitly negotiates 256 MiB buffer bindings for the
64M FP32 case; smaller limits fail instead of silently reducing the workload.
It retains the 32 KiB workgroup-storage ceiling. Shader compilation and
uploads finish before any timed call. NVRTC remains the CUDA compiler default,
and consumer execution remains toolkit free.

On the pinned producer environment with the optional Torch/native adapters
and WebGPU extra, use a working NVIDIA Vulkan ICD. The
[headless Vulkan setup](../guides/webgpu.md#headless-nvidia-vulkan-in-a-compute-container)
records how this compute-only workspace loaded the matching driver locally.

```sh
export PYTHONPATH="$PWD/packages/tensor-torch/src"
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
export CUDA_HOME=/path/to/cuda-12.9
export VK_DRIVER_FILES="$PWD/build/nvidia-vulkan-595.91.07/icd.json"
export LD_LIBRARY_PATH="$PWD/build/nvidia-vulkan-595.91.07/driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export WGPU_BACKEND_TYPE=Vulkan
python benchmarks/inference/scaling_backend_benchmark.py --webgpu \
  --cache build/phase4-completion-cache --out build/latency-scaling-a10g/full.json

python benchmarks/inference/scaling_backend_benchmark.py --webgpu \
  --case pointwise-129 --case gemm-4096-4096-4096 \
  --cache build/phase4-completion-cache --out build/latency-scaling-a10g-repeat/repeat.json

# Matplotlib 3.11.2 in the measured environment; the report includes both APIs.
python scripts/plots/plot_latency_scaling.py docs/research/data/latency-scaling.json \
  --out build/latency-scaling/scaling
```

Physical A10G execution establishes this same-device comparison. Phase 5's
separate AMD/Apple portability acceptance gate remains open; see the
[Phase 5 validation report](phase5-webgpu.md).

## Windows RX 6700 XT at the same workload sizes

Measured on **2026-10-01 (Asia/Jakarta)**, the physical RX 6700 XT through
Vulkan passed **all 17 scaling workloads and two targeted repeats**. The
operations, dimensions, dtypes and every WGSL SHA-256 match the A10G sweep
above. The AMD run used Windows 11, driver `32.0.21043.19003` / Vulkan
description `26.6.2`, Python 3.12.13, NumPy 2.5.3, wgpu 0.29.0 and
wgpu-native 27.0.2.0, with the same Tensor consumer source as the
[smaller RX 6700 XT validation](webgpu-rx6700xt.md).

![Same WebGPU workload sizes on A10G and RX 6700 XT](data/webgpu-rx6700xt-scaling.png)

[SVG figure](data/webgpu-rx6700xt-scaling.svg).
[AMD raw observations and repeats](data/webgpu-rx6700xt-scaling.json).
[Producer suite and shader hashes](data/webgpu-rx6700xt-scaling-suite.json).
[Standalone benchmark](../../benchmarks/inference/webgpu_scaling_benchmark.py).
[Plot exporter](../../scripts/plots/plot_webgpu_scaling.py).

Every entry is one warm **allocating call plus queue completion**, with
45 individual samples after 20 warmups. Output creation is inside the timer;
output release, uploads/downloads, CPU reference calculation, artifact build
and first pipeline creation are outside it. The consumer negotiates 256 MiB
allocation/storage-binding limits, allowing the full 64M FP32 workload without
reducing its size. It runs in the six-distribution compiler-free wheel
environment used for the earlier AMD validation.

The RX and A10G runs have **different input values**. AMD uses NumPy PCG64
with `SeedSequence([42, case_index])`; A10G used Torch's CUDA random generator.
AMD correctness uses NumPy FP32 references, with attention evaluated in
64-query chunks to bound host memory. All checks pass `atol=0.002, rtol=0.02`.
Pointwise results are exact; maximum GEMM absolute errors range from 0.0625
to 0.25, and attention errors from 0.0001221 to 0.0019531. Relative tolerance
and output magnitude are included in GEMM acceptance.

These are measurements of two complete systems. The RX run times one
WebGPU provider; the A10G run rotated it with CUDA providers. Hardware,
CPU, OS, driver, allocator and interleaving differ, so the ratios do not
isolate GPU architecture or shader execution time. No AMD CUDA, Torch,
TileLang or Triton baseline was measured. Here **RX/A10G > 1 means the RX
system's completed call took longer**; both columns use Tensor wgpu/Vulkan.

| FP32 pointwise elements | A10G wgpu | RX 6700 XT wgpu | RX/A10G |
|---|---:|---:|---:|
| 129 | 0.563 ms | 0.526 ms | 0.93× |
| 1M | 0.443 ms | 0.852 ms | 1.92× |
| 4M | 1.444 ms | 5.535 ms | 3.83× |
| 16M | 2.591 ms | 11.555 ms | 4.46× |
| 64M | 8.267 ms | 35.299 ms | 4.27× |

M means 2²⁰ elements. Every workload computes `relu(a*2+b)`.

| FP16 linear + bias + ReLU, M=N=K | A10G wgpu | RX 6700 XT wgpu | RX/A10G |
|---|---:|---:|---:|
| 512 | 0.859 ms | 1.245 ms | 1.45× |
| 1024 | 2.993 ms | 5.163 ms | 1.72× |
| 2048 | 20.285 ms | 28.780 ms | 1.42× |
| 4096 | 156.017 ms | 206.811 ms | 1.33× |

This retains transpose-B, bias/ReLU, 32×32 output tiles and K=16.

| FP16 attention, B=1 H=8 D=64 | A10G wgpu | RX 6700 XT wgpu | RX/A10G |
|---|---:|---:|---:|
| S=1024, noncausal | 3.690 ms | 5.178 ms | 1.40× |
| S=2048, noncausal | 13.828 ms | 19.836 ms | 1.43× |
| S=4096, noncausal | 49.585 ms | 73.172 ms | 1.48× |
| S=8192, noncausal | 192.876 ms | 268.707 ms | 1.39× |
| S=1024, causal | 2.605 ms | 3.225 ms | 1.24× |
| S=2048, causal | 8.449 ms | 11.989 ms | 1.42× |
| S=4096, causal | 26.932 ms | 35.106 ms | 1.30× |
| S=8192, causal | 100.019 ms | 134.228 ms | 1.34× |

Attention uses the same streaming online softmax and 8×16 query/key tiles.
The largest shapes fit and execute correctly. Their latency remains that of
the portable scalar/SIMT lowering; this run adds no subgroup or matrix acceleration.

The two AMD repeats regenerated identical inputs and produced identical output
hashes and shader hashes: 129-element pointwise measured **0.544 ms**, and
GEMM 4096³ measured **204.921 ms**, versus 0.526 and 206.811 ms in the main sweep.
The large-GEMM result therefore persists in the repeat. Interquartile bands
and all individual samples are retained in the figure and raw result.

To reproduce in PowerShell after preparing the wheel consumer described in
the [Windows validation report](webgpu-rx6700xt.md#reproduce-on-windows), use a
fresh suite directory and confirm that device 0 is the physical Vulkan adapter:

```powershell
.venv/Scripts/python.exe benchmarks/inference/webgpu_scaling_benchmark.py --build build/webgpu-rx6700xt-scaling
$env:OPENBLAS_NUM_THREADS = '6' # CPU reference only; outside GPU timing
build/webgpu-rx6700xt-consumer/Scripts/python.exe benchmarks/inference/webgpu_scaling_benchmark.py --consume build/webgpu-rx6700xt-scaling --device 0 --compare docs/research/data/latency-scaling.json --repeat-case pointwise-129 --repeat-case gemm-4096-4096-4096 --out build/webgpu-rx6700xt-scaling.json

uv run --no-project --python 3.12 --with matplotlib==3.11.2 --with numpy==2.5.3 python scripts/plots/plot_webgpu_scaling.py docs/research/data/latency-scaling.json build/webgpu-rx6700xt-scaling.json --out build/webgpu-rx6700xt-scaling-plot
```

The benchmark rejects changed consumer sources, artifact hashes, workload
shapes/dtypes, A10G WGSL hashes and software adapters. This follow-up is a
local producer/consumer performance sweep; it does not close Phase 5's
separate two-host acceptance gate. D3D12 remains outside this comparison
because it lacks `shader-f16` in the measured Windows configuration.

## RX 6700 XT compute and bandwidth utilization

The current lowering leaves substantial headroom: the largest GEMM achieves
approximately **5.03% of advertised FP32 peak**, while the largest pointwise
case achieves **5.94% effective useful-data bandwidth utilization**. These
are derived from the recorded completed allocating-call medians, not a new
GPU-counter measurement or a whole-model training MFU measurement.

AMD specifies **13.21 TFLOP/s FP32 vector**, **26.43 TFLOP/s FP16 vector**,
and **384 GB/s GDDR6 bandwidth** for the
[RX 6700 XT](https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6700-xt.html).
The compute denominator below is FP32 because the emitted WGSL explicitly
converts FP16 matrix operands to `f32` and calls `fma` on FP32 accumulators.
The packed FP16 vector peak is an alternate denominator, not the arithmetic
path currently emitted. Advertised peaks use peak clock assumptions; effective
clocks during these measurements were not captured.

| Workload | Useful throughput | Compute utilization vs FP32 peak |
|---|---:|---:|
| FP16 linear 512³ | 0.216 TFLOP/s | 1.63% |
| FP16 linear 1024³ | 0.416 TFLOP/s | 3.15% |
| FP16 linear 2048³ | 0.597 TFLOP/s | 4.52% |
| FP16 linear 4096³ | 0.665 TFLOP/s | 5.03% |
| Attention S8192, noncausal, B1/H8/D64 | 0.511 TFLOP/s | 3.87% |
| Attention S8192, causal, B1/H8/D64 | 0.512 TFLOP/s | 3.88% |

For GEMM, useful FLOPs are `2*M*N*K`, counting an FMA as two operations and
excluding bias/ReLU. Attention counts the two dominant matrix products:
`4*B*H*S*S*D` for noncausal, and `2*B*H*S*(S+1)*D` for the useful causal
triangle. Softmax, scaling/normalization and masked/padded tile work are not
included. Utilization is `useful FLOPs / (call seconds * peak FLOP/s)`.
These are useful-work MFU approximations; they are not the percentage of
GPU cycles busy or occupied. Against the full 26.43 TFLOP/s FP16 vector peak,
the largest GEMM is **2.51%** and S8192 attention is approximately **1.94%**.

Pointwise is better assessed by useful data movement. Reading two FP32 inputs
and writing one FP32 output requires `12*N` bytes per call, so effective MBU
is `12*N / (call seconds * 384e9)`:

| Pointwise elements | Useful IO bandwidth | Effective useful-IO MBU |
|---|---:|---:|
| 1M | 14.77 GB/s | 3.85% |
| 4M | 9.09 GB/s | 2.37% |
| 16M | 17.42 GB/s | 4.54% |
| 64M | 22.81 GB/s | 5.94% |

At 64M, that is 805,306,368 bytes / 35.299 ms, compared with the advertised
384 GB/s. The corresponding pure useful-data transfer lower bound is about
2.10 ms; this bound omits allocation, submission, initialization and additional
traffic, so it is not a promised optimized latency.

**Actual DRAM MBU is unknown.** Neither actual GDDR6 byte traffic nor isolated
GPU execution time was measured. Tiled GEMM and attention reload operands,
while caches may satisfy those loads. Dividing only their one-read/one-write
tensor sizes by time would report minimum useful-IO rates, not actual DRAM
utilization, and would be especially misleading for attention. AMD's cache-based
effective bandwidth figure is not the GDDR6 peak used here. The same allocation
and host overhead included in call timing also lowers these effective metrics;
the results do not identify the precise kernel or driver bottleneck.

The [derived utilization data](data/webgpu-rx6700xt-utilization.json) contains
all 17 workloads and both repeats, FLOP/byte counts, both compute denominators,
and timing-report/specification provenance. Actual DRAM MBU is explicitly null.
Reproduce the derivation without rerunning the GPU:

```powershell
build/webgpu-rx6700xt-consumer/Scripts/python.exe benchmarks/inference/webgpu_utilization.py docs/research/data/webgpu-rx6700xt-scaling.json --out build/webgpu-rx6700xt-utilization.json
```
