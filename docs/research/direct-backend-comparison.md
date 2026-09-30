# Direct Tensor, TileLang and Triton comparison

Measured on 2026-09-30: NVIDIA A10G (`sm_86`), driver 595.91.07, Python 3.12,
PyTorch 2.14.0+cu130, TileLang 0.1.14, TVM-FFI 0.1.12 and Triton 3.8.0.
Tensor uses its optional C++ executor and NVRTC 12.9 cubins. Native TileLang
uses TVM-FFI with NVCC 12.9.86. These are cached inference microbenchmarks
on one GPU, with fixed schedules and no autotuning.

Tensor's direct runtime has lower submission overhead than both baselines.
The matched TileLang control executes **identical cubins**, and its GPU time
agrees with Tensor within 1.8% for every profile. Triton improves GEMM GPU
execution, while attention results vary by shape and causal mode.
The outer `torch.compile` wrapper remains a substantial cost: complete Tensor
calls through that API are slower than direct TileLang and Triton calls.

[Raw samples, source/binary hashes and environment](data/direct-backend-comparison.json).
[Benchmark implementation](../../tools/direct_backend_benchmark.py).
[Independent Triton kernels](../../tools/direct_triton_kernels.py).

## Aggregate results

Each entry is the geometric mean of baseline latency divided by **direct
Tensor latency**, with all 20 profiles weighted equally. Values above 1 favor
Tensor. This profile mix includes 12 attention cases; it is not a whole-model
speedup or a representative distribution of application workloads.

| Measurement | TileLang, identical cubin | TileLang, ordinary NVCC | Triton compiled launcher | Triton warmed JIT |
|---|---:|---:|---:|---:|
| Allocating host submission | 1.55× | 1.52× | 2.01× | 3.87× |
| Allocating calls with stream completion | 1.27× | 1.26× | 1.48× | 2.33× |
| Prepared host submission | 1.92× | 1.90× | 2.12× | 4.75× |
| Prepared GPU execution | 1.00× | 0.99× | 0.98× | 0.98× |

For the 129-element pointwise operation, prepared submission is **4.20 µs**
for Tensor, **7.81 µs** for matched TileLang, **8.59 µs** for Triton's compiled
launcher and **18.66 µs** for warmed Triton JIT dispatch. GPU execution is
about **1.16 µs** for every provider. Allocating calls with completion take
8.08, 13.10, 17.16 and 30.81 µs respectively.

Complete Tensor calls through `torch.compile` have **2.66× higher latency**
than direct matched TileLang, **2.29× higher latency** than Triton's compiled
launcher and **1.45× higher latency** than warmed Triton JIT, geometrically
across these profiles. Tensor's outer wrapper adds little captured GPU work,
but it affects ordinary host calls. For pointwise-129, the complete call takes
51.33 µs versus 8.08 µs through the direct Tensor graph. The direct result
therefore describes the C++ execution path, rather than the current public
`torch.compile` entry point's full latency.

The retained eager and default Inductor baselines give complete Tensor calls
1.55× higher latency than eager and 1.74× speedup over Inductor in this run.
Captured GPU execution gives direct Tensor 1.52× speedup over eager and 1.24×
over Inductor. Host and GPU ratios answer different questions.

## Kernel differences

Prepared GPU speedup of Tensor relative to the independent fixed-schedule
Triton kernels is 1.00× for pointwise, 0.85× for GEMM/MLP, and 1.01× for
attention, using geometric means within each family. A near-equal overall
mean hides individual wins and losses.

| Workload | Tensor GPU | Matched TileLang GPU | Triton GPU | Eager PyTorch GPU |
|---|---:|---:|---:|---:|
| GEMM + bias + ReLU, 512³ | 9.93 µs | 9.93 µs | 7.99 µs | 9.40 µs |
| Attention, B=1 H=8 S=1024 D=64, non-causal | 68.83 µs | 68.85 µs | 57.67 µs | 49.01 µs |
| Attention, B=1 H=8 S=1024 D=64, causal | 44.75 µs | 44.81 µs | 51.92 µs | 49.54 µs |

This table uses capture of allocating functions, including eager PyTorch;
allocations and Python dispatch occur during capture and are excluded from
the event timings. The raw report also contains separate fixed-output capture
measurements, which closely agree. Eager attention uses PyTorch's default
SDPA selection; no attention backend is forced in this comparison.

Triton's large GEMM executes about 1.24× faster than Tensor's kernel, although
its allocating completed calls take 17.86 µs versus Tensor's 10.74 µs. Long
non-causal attention is GPU-bound: completed calls take 69.70 µs for Tensor,
59.03 µs for Triton and 50.38 µs for eager PyTorch. Further wrapper changes
cannot remove that kernel execution gap.

## Controls and methodology

The suite reuses Phase 4's four FP32 pointwise sizes, three FP16 fused linear
shapes, one two-layer MLP, and six FP16 BHSD attention shapes with both causal
modes. Shapes include partial tiles. All inputs are contiguous, with seed 42,
inference mode and a non-default CUDA stream. All allocating and prepared
graphs are checked numerically against eager before timing.

* `tensor_direct` executes the lowered FX graph's C++ regions without the
  outer Dynamo wrapper. `tensor_compile` uses ordinary `torch.compile`.
* `tilelang_matched` loads the artifact's original TIRx and uses native
  `JITKernel` host code generation and TVM-FFI allocation/launch. Its device
  compiler callback returns Tensor's exact cubin. The lowered entrypoint,
  argument ABI, grid, block and shared-memory size must match the artifact
  before any execution. All 42 allocating/prepared kernel variants have
  verified cubin SHA-256 equality; outputs match Tensor bit for bit.
* `tilelang_default` uses the same original TIRx and unmodified native NVCC
  device compilation. This includes device compiler differences as well as
  runtime overhead. It is a same-schedule baseline, not an independently
  tuned TileLang implementation.
* `triton` uses the native `CompiledKernel[grid]` runner after compilation.
  `triton_jit` uses the usual warmed `@triton.jit` dispatch. Both execute the
  same independently implemented Triton kernels. Neither baseline uses Tensor.

Repeated TileLang lowering can exchange independent reduction scratch-buffer
locations. Generated CUDA text therefore does not always match byte for byte:
36 of the 42 matched wrapper variants match the reference lowering. The
identical-cubin control checks the externally visible ABI and launch metadata,
loads the verified original binary, and checks exact output equality. It does
not rely on regenerated scratch-buffer names or offsets. Ordinary NVCC
allocating/prepared variants may consequently have different cubin hashes.

Host submission measures 200 calls per batch, with completion outside the
timer; it measures enqueue cost, not time until output readiness. Allocating
completed timing measures 100 queued calls followed by stream completion,
amortized per call. It does not synchronize after every invocation. Prepared
calls bind fixed outputs, including both MLP stages, outside timing. All host
measurements warm each provider and rotate provider order over nine batches.

GPU measurements capture 50 calls and use CUDA events around replay, with
nine rotated batches. Graphs are warmed before timing. This measures steady
GPU execution with fixed addresses and removes Python submission overhead.
Every raw batch is retained; reported latency is its median. No cold compile
time or whole-model inference time is inferred from these measurements.

Triton pointwise uses 256 elements and four warps; GEMM uses 32×64×32 tiles,
four warps and three stages; attention uses 32×64 query/key tiles, four warps
and one stage. These tile sizes mirror Tensor's schedules, but layouts and
instructions are chosen by different compilers. Triton disables FP fusion and
uses FP32 accumulation with FP16 linear outputs and attention probabilities.
Tensor NVRTC and TileLang's default NVCC use C++20 without fast-math overrides.
There is no autotuning or claim that these are the fastest possible kernels.
The implementations follow the algorithms described in Triton's official
[matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)
and [fused attention](https://triton-lang.org/main/getting-started/tutorials/06-fused-attention.html)
tutorials.

## Reproduction and validation

Use the installed native Tensor adapter, pinned Torch/TileLang/Triton versions,
Tensor's local NVRTC bundle, and a CUDA 12.9 toolkit plus host compiler for the
native TileLang baseline. The toolkit is an optional benchmark dependency;
Tensor consumers retain their existing compiler-free execution path.

```sh
PYTHONPATH="$PWD/packages/tensor-torch/src" \
TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" \
CUDA_HOME=/path/to/cuda-12.9 \
  python tools/direct_backend_benchmark.py \
    --cache build/phase4-completion-cache --out build/direct-comparison/full.json

TENSOR_DIRECT_CUDA=1 python -m pytest tests/test_direct_backends.py -q
```

`--quick` selects six profiles. The full run checks all 20 profiles for every
provider. Seven additional GPU regressions pass with zero skips in 4.82 seconds,
covering NaN propagation, all three GEMM tails, short causal attention blocks,
sequence tails, head dimensions 64/128, multiple batches/heads, non-default
streams, compiled-runner output allocation and CUDA graph replay. The tests
remain opt-in so Triton is not required by consumer or Windows builds.

These results support further work on the outer Dynamo wrapper and GEMM/long
non-causal attention schedules. NVRTC remains the default compiler; this
comparison introduces no compiler backend or consumer dependencies. Direct
PTX remains experimental work after v1.
