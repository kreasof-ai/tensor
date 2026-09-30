# Phase 4 — Inference-first PyTorch adapter acceptance

Accepted 2026-09-30 at source commit `c700d3315c23a2710ea6f51865b46ca0eed37f81`. The scoped Phase 4 work
is complete: a separate `tensor-torch` wheel, functional and mutable-output
custom operators, an actual FX-to-TIRx-to-NVRTC inference backend, visible
partial fallback, graph breaks, guarded shape specializations, compiler-free
cached execution, and AOTAutograd forward/backward lowering evaluation.
Full compiled training remains experimental and is not a Phase 4 completion claim.

[Usage and contracts](../pytorch.md), [raw metrics](data/phase4-metrics.json),
[acceptance and transfer records](data/phase4-exit.json), and
[CI evidence](data/phase4-ci.json) provide reproducible details.

## Implementation and correctness

Core Tensor still installs with NumPy alone. The adapter reuses the client's
PyTorch and exposes the `torch_dynamo_backends` entry point `tensor`.
The compiler-free consumer imports neither TileLang nor TVM. A cache miss
requires the pinned frontend and local NVRTC 12.9 bundle; it never invokes nvcc.
Direct PTX remains experimental after Tensor v1.

Supported inference profiles are FP32/FP16 pointwise arithmetic and activations,
rank-two FP16 matmul/linear with bias/activation epilogues, chains of linear
layers, and FP16 forward BHSD self-attention with head dimension 64/128.
Attention covers causal/noncausal, tails, and batching without a global score
matrix. Masks, dropout, GQA, and attention backward remain unsupported.
Unsupported nodes execute their original PyTorch FX targets and are reported.
Compiler errors propagate instead of masquerading as compiled regions.

Outputs are allocated by PyTorch. The first concrete binding uses core DLPack
and runtime ABI 1.1 validation. Guarded launches reuse descriptor storage,
select the current PyTorch stream on every invocation, and record allocator
usage without per-call synchronization. Prepared calls retain fixed tensors,
validate storage/metadata before submission, and reject changed bindings.
Custom operators declare their mutation contract and provide symbolic FakeTensor
implementations. `opcheck` verifies registration; separate numerical checks
verify execution. The optional native submission shim builds with Python
headers and links no CUDA toolkit or PyTorch libraries. Installed wheels need
no host compiler. A ctypes fallback remains available.

The full GPU-enabled suite passes **141 tests, zero skips, in 167.15 s**.
Checks include FP16 intermediate rounding, NaN propagation, activation precision,
GEMM tails and bias, multi-layer execution, attention tails/causal masks,
current-stream ordering, allocator lifetime, CUDA graph capture, metadata/storage
invalidation, scalar reuse, alias rejection, graph breaks, mixed supported and
unsupported regions, specialization limits, corrupted cache repair, and fresh
process execution with compiler imports blocked.

An unaligned attention view exposed a misaligned-address error in PyTorch's
chosen Flash provider on this environment. The fallback regression explicitly
selects PyTorch's math provider and verifies shared q/k/v operands. Tensor
preserves the client's provider policy for fallback operations.

AOTAutograd evaluation compiles affine/ReLU forward plus an actual backward
multiply region and checks both input gradients. Detach and threshold-backward
remain reported PyTorch operations. The default backend preserves eager
autograd when gradients are enabled; it never silently detaches a training graph.
This is bounded compiled forward/backward work, rather than capture-only evidence.

## Performance acceptance

Measured on NVIDIA A10G (`sm_86`), driver 595.91.07, Python 3.12, PyTorch
2.14.0+cu130, NVRTC 12.9, and runtime ABI 1.1. Both compiled providers use
`fullgraph=True`, `dynamic=False`, and default mode. Every benchmark region
actually executes Tensor kernels; none uses semantic fallback. Results apply
to this suite and configuration, rather than every model or PyTorch release.

Warm end-to-end measurements include Python/Dynamo submission, output allocation,
and stream completion, amortized over 100 calls with median of seven batches.
GPU times separately replay a 50-call CUDA graph with CUDA events; they exclude
Python submission. Inputs are allocated before timing. Attention also measures
forced PyTorch Flash SDPA. The benchmark records source/binary hashes; the final
run matches the accepted source, uses a fresh artifact cache, and runs without
concurrent GPU tests. Cold times include graph capture, lowering, NVRTC and first
execution, with PyTorch/CUDA already initialized. Cached preparation includes
recapture and first execution; it invokes no Tensor compiler.

| Gate | Target | Measured | Result |
|---|---:|---:|---|
| Warm end-to-end geometric mean vs Inductor | ≤1.10× | 0.756× | Pass |
| Worst end-to-end case vs Inductor | ≤1.25× | 1.116× | Pass |
| Selected fused 4M-element graph vs eager | ≥1.25× faster | 2.39× | Pass |
| Prepared host submission | ≤15 µs | 8.75 µs | Pass |
| Cached graph preparation, maximum | ≤100 ms | 57.30 ms | Pass |
| Cold pointwise graph preparation, maximum | ≤10 s | 2.72 s | Pass |
| Cold GEMM graph preparation, maximum | ≤30 s | 2.87 s | Pass |
| Cold two-layer MLP preparation | ≤30 s | 5.62 s | Pass |

The table below distinguishes device execution from complete graph calls.
Ratios below 1 favor Tensor. Eager speedup above 1 favors Tensor.

| Case | Tensor end-to-end µs | Inductor µs | Tensor/Inductor | Eager speedup | Tensor GPU µs | Inductor GPU µs |
|---|---:|---:|---:|---:|---:|---:|
| pointwise-129 | 70.99 | 63.58 | 1.116 | 0.37× | 1.15 | 1.15 |
| pointwise-257 | 68.10 | 62.84 | 1.084 | 0.39× | 1.17 | 1.17 |
| pointwise-1048576 | 68.71 | 61.73 | 1.113 | 0.67× | 27.30 | 26.34 |
| pointwise-4194304 | 105.29 | 107.33 | 0.981 | 2.39× | 103.22 | 105.57 |
| gemm-33-65-64 | 72.36 | 94.23 | 0.768 | 0.36× | 3.17 | 3.71 |
| gemm-128-128-128 | 73.03 | 101.34 | 0.721 | 0.43× | 2.74 | 3.87 |
| gemm-512-512-512 | 72.25 | 95.77 | 0.754 | 0.38× | 10.04 | 9.36 |
| mlp-128-256-128 | 96.99 | 133.71 | 0.725 | 0.54× | 6.10 | 7.05 |
| sdpa-1-8-128-64-False | 72.02 | 99.36 | 0.725 | 0.33× | 5.20 | 8.66 |
| sdpa-1-8-128-64-True | 72.17 | 100.88 | 0.715 | 0.32× | 5.24 | 8.99 |
| sdpa-1-8-129-64-False | 73.18 | 101.17 | 0.723 | 0.32× | 6.92 | 15.71 |
| sdpa-1-8-129-64-True | 72.53 | 100.94 | 0.719 | 0.33× | 6.70 | 16.36 |
| sdpa-2-4-257-64-False | 72.82 | 114.92 | 0.634 | 0.43× | 10.38 | 13.84 |
| sdpa-2-4-257-64-True | 71.90 | 113.52 | 0.633 | 0.44× | 10.40 | 14.19 |
| sdpa-1-8-512-64-False | 70.42 | 108.37 | 0.650 | 0.43× | 19.82 | 23.49 |
| sdpa-1-8-512-64-True | 71.22 | 111.80 | 0.637 | 0.44× | 19.78 | 22.73 |
| sdpa-1-8-1024-64-False | 71.82 | 99.09 | 0.725 | 0.70× | 68.98 | 49.48 |
| sdpa-1-8-1024-64-True | 70.71 | 98.57 | 0.717 | 0.72× | 44.85 | 49.62 |
| sdpa-1-8-512-128-False | 70.84 | 111.35 | 0.636 | 0.64× | 32.42 | 42.29 |
| sdpa-1-8-512-128-True | 70.26 | 110.98 | 0.633 | 0.56× | 31.13 | 36.35 |

Most small graph calls remain slower than eager because framework/adapter host
cost exceeds their GPU work. The selected large fused pointwise case benefits
from avoiding intermediate memory traffic. Noncausal attention at sequence 1024
still takes about 69 µs on the GPU versus 49 µs for forced Flash SDPA; its
end-to-end Inductor comparison passes because host overhead is also counted.
These are measured limitations, not a claim that every kernel beats PyTorch.
The earlier ctypes implementation failed the worst-case gate. Native submission,
static Dynamo guard reuse and PyTorch's fast allocation path close that gap.
The tested PyTorch release exposes two private optimization helpers; public API
fallbacks are retained, and other releases need their own performance validation.

## Installation and transfer acceptance

[GitHub Actions run 36658315181](https://github.com/kreasof-ai/tensor/actions/runs/36658315181)
passes on Ubuntu and Windows: **100 passed / 41 skipped** and
**90 passed / 51 skipped**, respectively. Hardware-only tests are skipped there.
Both runners build native ABI3 adapter wheels, compile three actual adapter-emitted
FX profiles using NVRTC without a GPU, and install/discover the backend in a
consumer containing no compiler packages. CPU fallback and FakeTensor CUDA
custom-op execution pass on each platform.

The downloaded Linux and Windows profiles each execute affine, GEMM/bias/ReLU,
and causal tail attention on the A10G using an isolated compiler-free Linux
consumer. Artifact hashes match each producer report. The uploaded Linux native
wheel executes all 20 cached graphs with compiler imports blocked; its native
prepared launch path also executes all six transferred profiles. The Windows
wheel is built/imported/tested on Windows, while GPU execution of Windows-produced
images is measured on Linux. No Windows GPU execution or cross-SM claim is made.
The Linux adapter wheel is approximately 20 KB; PyTorch remains a separate client
dependency. No public PyPI upload occurred.

Reproduce the GPU suite with the compiler environment and its local NVRTC bundle:

```sh
TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9" \
CUDA_HOME="$PWD/experiments/p0/out/cuda-12.9" \
TENSOR_P0_CUDA=1 TENSOR_P1_CUDA=1 TENSOR_P2_CUDA=1 \
TENSOR_P3_CUDA=1 TENSOR_P4_CUDA=1 TENSOR_ATTENTION_CUDA=1 \
uv run --locked --extra publish python -m pytest -o addopts='' -q
```

Build and install the adapter wheel, then run:

```sh
python benchmarks/inference/backend_benchmark.py --cache build/torch-benchmark-cache --out build/torch-benchmark.json
python benchmarks/inference/fx_producer.py --target sm_86 --out build/torch-profiles
```

Use `benchmarks/inference/fx_consumer.py` from a clean wheel installation for backend
discovery, FakeTensor contracts, compiler-free cached graphs, and transferred
profile execution. The committed evidence records exact versions, artifact
hashes, targets, measured timings, test counts and CI job identities.
