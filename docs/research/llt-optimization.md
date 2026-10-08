# LLT projection optimization on NVIDIA L40S

The optimized explicit Tensor operators reduce one-token LLT decode from
3.415 ms to 0.439 ms in the measured W768/H12/L12/R64/S1024/V256 fixture.
The matched Torch control takes 0.687 ms. This is CUDA graph execution with
prepared BF16 serving weights, FP32 embeddings/residuals, and a persistent
1024-token prefix. It is a small-vocabulary fixture; these numbers do not
establish full-vocabulary training performance.

The complete retained measurements, candidate schedules, source snapshots and
artifact hashes are in [llt-optimization](data/llt-optimization). Model-scale
experiments are tracked in the
[LLT repository](https://github.com/kreasof-ai/loop-latent-transformer).

## Bottleneck and implementation

CUPTI attribution places about 94% of baseline decode GPU time in matrix
projections. The old fixed 32×64×32 tile has few CTAs for one input row and
loads transposed linear weights with poor global coalescing. The new template
preserves row-major shared storage and transposes at MMA. Small linear inputs
use a coalesced row-reduction GEMV. Both accumulate in FP32 and store BF16/FP16.

An actual `ScheduleSearch` beam, width 4, explored up to 18 candidates per case
with a 180-second budget. Every timed candidate first passed a seeded FP32
reference comparison (absolute tolerance .005, relative .035). The search
includes forward, input gradient, weight gradient, decode and vocabulary
projection shapes. All candidates and nine raw timing samples are retained;
the search does not establish global optimality.

Production dispatch selects the searched GEMV and shape-dependent MMA schedules.
`Operators(gemm_profile="legacy")` retains the original GEMM for controlled
comparisons. Target performance has been qualified on sm89 only. There is no
automatic tuning during normal execution and no Torch numerical fallback.

Independent production rechecks after search, in rotating backend order:

| M×K×N | Legacy µs | Optimized µs | Torch µs | Improvement |
|---|---:|---:|---:|---:|
| 1×3072×768 | 131.07 | 4.81 | 8.91 | 27.23× |
| 1×768×3072 | 34.15 | 4.10 | 6.50 | 8.34× |
| 1024×768×3072 | 308.63 | 36.61 | 33.17 | 8.43× |
| 1024×3072×768 (dX) | 50.79 | 39.22 | 30.21 | 1.30× |
| 3072×1024×768 (dW) | 207.27 | 36.66 | 40.29 | 5.65× |
| 1024×768×50304 | 4944.90 | 433.46 | 391.99 | 11.41× |

These are isolated, repeatedly replayed kernels with hot weights; they must not
be substituted for whole-model timing. The model comparison independently
includes normalization, residuals, activation, attention and cache operations.

## Correctness and nanoGPT coverage

The qualification checks FP16/BF16, all four transpose combinations, and odd
row/channel/contraction tails. Odd row strides expose a pinned TileLang
`cp.async` two-byte transfer failure. The template disables pipelining and
uses synchronous copies for those strides; aligned tiles retain async copies.
No installed dependency is patched. The original search factory is frozen in
`search-sources` and the final corrected factory is frozen in `sources`.

Unweighted RMSNorm no longer computes an unused norm-weight gradient. Existing
learned RMSNorm gradients remain supported. Affine LayerNorm forward, input
and parameter gradients, hierarchical channel reductions, and optional linear
bias support were added to execute actual nanoGPT. Parameters remain FP32;
LayerNorm reductions accumulate in FP32. Linear bias currently uses a separate
epilogue, introducing an extra BF16 rounding compared with a fused Torch linear.

Small LLT/naive models pass all-parameter gradient comparisons. The actual
nanoGPT adapter preserves the upstream model's parameter names and tied
embedding/classifier identity. An eight-step paired update comparison passes;
its first-step worst parameter-gradient relative L2 is .00661 and maximum loss
difference is .00117. This is numerical qualification on seeded weights, not
trained language quality. The upstream nanoGPT source and adapter identities
are included in the manifest.

## Reproduction

Run GPU phases sequentially from the Tensor repository root:

```sh
export TENSOR_NVRTC_HOME=build/nvrtc-12.9
.venv/bin/python -m benchmarks.llt.optimize profile
for tensor_case in 0 1 2 3 4 5; do
  .venv/bin/python -m benchmarks.llt.optimize search --case "$tensor_case" --candidates 18 --seconds 180
done
.venv/bin/python -m benchmarks.llt.optimized_checks
.venv/bin/python -m benchmarks.llt.optimized_models
.venv/bin/python -m benchmarks.llt.optimized_finalists
TENSOR_LLT_CUDA=1 .venv/bin/pytest packages/tensor-torch/tests/test_llt.py -q \
  -k 'model_operators or aliased_latent_gradient or chunked or affine_layer_norm or channel_bias'
.venv/bin/python -m benchmarks.llt.optimization_manifest
```

Compilation and warmup are excluded from latency. The LLT fixture and nanoGPT
adapter come from the LLT repository; `LLT_CHECKOUT` can select its checkout.
The manifest records their exact source snapshots. Kernel binaries and profiler
traces remain build artifacts rather than committed development logs.
