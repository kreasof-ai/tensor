# Larger-shape latency scaling

The outer `torch.compile` wrapper becomes a smaller fraction of latency as
GPU work grows. It does not explain every large-shape performance gap.
At 64M pointwise elements, Tensor through `torch.compile` is only **1.03×
slower than direct matched TileLang**. At GEMM 4096³, that ratio is 1.00×,
but both execute a kernel that is much slower than Triton and eager PyTorch.

Measured on 2026-09-30 with the same A10G, C++ executor and pinned compiler
environment as the [direct comparison](direct-backend-comparison.md).
All 17 scaling profiles passed numerical checks. A separate repeat of the
small pointwise anchor and largest GEMM confirmed the GEMM slowdown.

![Latency scaling and full Tensor call ratios](data/latency-scaling.png)

[SVG figure](data/latency-scaling.svg).
[Full raw observations, targeted repeat and fingerprints](data/latency-scaling.json).
[Scaling benchmark](../../tools/scaling_backend_benchmark.py).
[Plot exporter](../../tools/plot_latency_scaling.py).

## Individual-call latency

These tables report **one warm allocating call followed by stream
synchronization**, in milliseconds. The reported median uses 45 individual
observations with rotating provider order. Synchronization overhead is included
for every provider. This measures time until the output is ready, rather than
the amortized completed batches reported in the earlier comparison.

TileLang here uses native TVM-FFI allocation/launch with Tensor's identical
cubin. Triton uses its compiled-kernel runner. The raw data also retains ordinary
NVCC TileLang, warmed Triton JIT, direct Tensor and default Inductor results.

| FP32 pointwise elements | Tensor through torch.compile | Matched TileLang | Triton | Eager PyTorch | Tensor / TileLang |
|---|---:|---:|---:|---:|---:|
| 129 | 0.088 | 0.026 | 0.034 | 0.047 | 3.35× |
| 1M | 0.073 | 0.041 | 0.047 | 0.061 | 1.78× |
| 4M | 0.151 | 0.120 | 0.127 | 0.265 | 1.26× |
| 16M | 0.457 | 0.426 | 0.440 | 1.002 | 1.07× |
| 64M | 1.683 | 1.640 | 1.664 | 3.951 | 1.03× |

Here M means 2²⁰ elements. The first pointwise anchor varied between runs:
the targeted repeat measured 0.058 ms for Tensor and 0.017 ms for matched
TileLang, with a similar 3.41× ratio. Its change reflects timing variation,
rather than a claim that 1M elements are intrinsically faster than 129.
Both observations are retained and the repeat appears as hollow plot markers.

| FP16 linear + bias + ReLU, M=N=K | Tensor through torch.compile | Matched TileLang | Triton | Eager PyTorch | Tensor / TileLang |
|---|---:|---:|---:|---:|---:|
| 512 | 0.059 | 0.026 | 0.031 | 0.033 | 2.25× |
| 1024 | 0.105 | 0.072 | 0.071 | 0.070 | 1.46× |
| 2048 | 0.482 | 0.438 | 0.382 | 0.378 | 1.10× |
| 4096 | 10.633 | 10.587 | 4.059 | 2.375 | 1.00× |

At 4096, full Tensor calls are still **2.62× slower than Triton** and
**4.48× slower than eager PyTorch** in the main sweep. The repeat measured
10.607, 10.549, 4.400 and 2.353 ms respectively. This is a persistent GPU
kernel gap despite ordinary run-to-run variation, especially in Triton's
measurement. None of these implementations is autotuned by this benchmark.

| FP16 attention, B=1 H=8 D=64 | Tensor through torch.compile | Matched TileLang | Triton | Eager PyTorch | Tensor / TileLang |
|---|---:|---:|---:|---:|---:|
| S=1024, non-causal | 0.118 | 0.085 | 0.081 | 0.073 | 1.39× |
| S=2048, non-causal | 0.270 | 0.236 | 0.215 | 0.205 | 1.14× |
| S=4096, non-causal | 0.817 | 0.778 | 0.699 | 0.718 | 1.05× |
| S=8192, non-causal | 2.991 | 2.944 | 2.698 | 2.491 | 1.02× |
| S=1024, causal | 0.097 | 0.062 | 0.075 | 0.074 | 1.55× |
| S=2048, causal | 0.187 | 0.156 | 0.161 | 0.209 | 1.20× |
| S=4096, causal | 0.497 | 0.460 | 0.429 | 0.478 | 1.08× |
| S=8192, causal | 1.668 | 1.620 | 1.467 | 1.484 | 1.03× |

At S=8192, Tensor's full call is about 11% slower than Triton for non-causal
attention and 14% slower for causal attention. The same-kernel TileLang ratio
is close to parity because it isolates wrapper cost. It cannot establish
competitiveness against a different GPU kernel.

## Host overhead versus GPU execution

Excluding the noisy 129-element anchor, the main sweep's host submission
difference between `tensor_compile` and `tensor_direct` has a median of
**43 µs**, ranging from 40 to 49 µs. Their serialized-call difference has a
median of 38 µs, ranging from 35 to 80 µs. These are differences between
independently measured medians; they are not an exact additive decomposition
of CPU and GPU work.

For 64M pointwise elements, Tensor's captured GPU time is about 1.62 ms,
so tens of microseconds of wrapper cost have little relative impact. For
the 1M case, about 25 µs of GPU work remains comparable to the wrapper cost.

The GEMM 4096 slowdown remains when host dispatch is removed:

| Captured GPU time, GEMM 4096³ | Main sweep | Targeted repeat |
|---|---:|---:|
| Direct Tensor | 10.536 ms | 10.514 ms |
| Matched TileLang | 10.550 ms | 10.498 ms |
| Triton compiled launcher | 4.034 ms | 4.326 ms |
| Eager PyTorch | 2.363 ms | 2.304 ms |

Direct Tensor GPU time increases about 24.5× from GEMM 2048³ to 4096³, while
the arithmetic work increases 8×. Identical-cubin TileLang reproduces the
behavior. This points to poor scaling of the fixed kernel schedule; this
benchmark does not identify the precise memory or instruction bottleneck.
Improving only the outer wrapper cannot resolve this gap.

## Method and reproduction

The sweep uses seed 42, contiguous inputs, inference mode and a non-default
stream. FP32 pointwise remains `relu(a*2+b)`, FP16 linear retains bias/ReLU,
and FP16 BHSD attention uses default PyTorch SDPA as its numerical/baseline
reference. Shape growth changes no tile sizes or tuning policy.

All allocating and fixed-output direct graphs are checked against eager;
matched TileLang outputs are bitwise equal to Tensor. Both `torch.compile`
Tensor and Inductor outputs are checked too. All 34 matched cubin variants
in the main sweep and four in the repeat have verified binary hashes and
matching argument ABI and launch metadata.

Host measurements use nine rotated batches of 20 calls, after 100 warmups
per provider. The completion variant synchronizes once after each batch and
divides elapsed time by 20; it describes amortized throughput. Serialized
measurements warm 20 calls, then synchronize after every one of 45 timed
calls. Output destruction occurs outside each serialized timer. GPU timing
captures ten calls and uses nine CUDA-event samples with warmed graphs.
The small 129-element anchor's GPU event timing has appreciable graph launch
and event overhead with ten-call replay; it is not used to infer intrinsic
microkernel latency. The larger-shape GPU comparisons remove that ambiguity.

No individual run provides an exact CPU/GPU sum. Batched calls overlap host
submission with GPU work, so their overhead becomes almost invisible once
GPU execution dominates. The serialized measurements retain the additional
time a caller pays before one result is ready.

On the same pinned producer environment, with the optional native adapter,
local NVRTC bundle and native TileLang's NVCC toolkit:

```sh
export PYTHONPATH="$PWD/packages/tensor-torch/src"
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
export CUDA_HOME=/path/to/cuda-12.9
python tools/scaling_backend_benchmark.py \
  --cache build/phase4-completion-cache --out build/latency-scaling/full.json

# Repeat selected cases with the same controls.
python tools/scaling_backend_benchmark.py \
  --case pointwise-129 --case gemm-4096-4096-4096 \
  --cache build/phase4-completion-cache --out build/latency-scaling/repeat.json

# Optional plot dependency: matplotlib 3.11.2 in the measured environment.
python tools/plot_latency_scaling.py docs/research/data/latency-scaling.json \
  --out build/latency-scaling/scaling
```

NVRTC remains Tensor's default compiler and consumer execution remains toolkit
free. These optional research tools add no consumer dependency or backend.
