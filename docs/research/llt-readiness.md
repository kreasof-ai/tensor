# Tensor LLT dependency qualification on L40S

This report qualifies the explicit `tensor_torch.llt` workload profile for
[LLT](https://github.com/kreasof-ai/loop-latent-transformer). The API is described
in the [LLT guide](../guides/llt.md); the [dependency checklist](../plan/llt-readiness.md)
tracks T01–T10. This scope is separate from Tensor 1.0, model quality, and LLT's
architecture performance targets.

Qualified implementation: [ab17948](https://github.com/kreasof-ai/tensor/commit/ab17948fe94dd7c7b857c8f067c465e901329b94). Subsequent documentation-only
commits do not change the qualified source or wheel hashes.

## Implementation and numerical contract

ABI 1.3 adds permanent BF16 ID 13 and `bfloat16_storage`, preserving IDs 1–12 and
all native descriptor layouts. New BF16 artifacts require that capability;
non-BF16 CUDA artifacts retain their old minor requirement. CUDA buffers support
owned/borrowed BF16 and lifetime-safe DLPack export. The runtime has explicit RNE
conversion, raw-byte download, and FP32-decoded NumPy download.

Attention uses online softmax and recomputed tile gradients without a global
sequence-squared score/probability allocation. Shared K=V receives both gradients
and every head's contribution through autograd. Long shared-head backward uses
FP32 head partitions plus deterministic reduction; this workspace is linear in
sequence length and does not replicate the input latent. Separate score/value
dimensions and runtime CUDA rotary offsets implement decoupled positional scores.
Changing token positions does not compile a new rotary artifact. Model
kernels include transposed/batched GEMM, RMSNorm, GELU, embeddings, residuals,
full-vocabulary CE, gradient norms/clipping, and FP32 AdamW. Checkpoint replay is
exact for the tested deterministic model.

Predeclared BF16 attention tolerances are 0.035 absolute/relative; gradients use
0.05. FP16 attention and gradients use 0.005. Full-model gradient relative L2 is
bounded by 0.08; loss drift by 0.15, final parameter RMS difference by 0.03, and
resume parameter error by 1e-6. Unit stress cases publish their own tolerances.
Matched training uses identical initialization, batches, constant LR 0.001,
AdamW, norm clipping, and exact loop checkpoints. LR 0.003 was rejected during
bring-up because the naive fixture accumulated loss drift; tolerances were not
relaxed. [Bring-up rejections](data/llt-readiness/bringup-rejections.json) retain
the failed settings; concurrent preliminary timings were discarded. This synthetic learnable sequence tests backend correctness rather than
trained model quality.

## Method and environment

Linux, Python 3.12.14, Torch 2.14.0/CUDA 13.0, TileLang 0.1.14, TVM FFI 0.1.12,
NVRTC 12.9, NVIDIA L40S `sm_89`, driver 595.91.07. TF32 is disabled. Native adapter
wheels are built for the matching Torch 2.14 minor. Reports retain the producer
base revision plus exact implementation hashes and source snapshots; the commit
containing this report pins the qualified implementation. Wheel and artifact
hashes are retained in [audit.json](data/llt-readiness/audit.json).

Final qualification runs one GPU experiment process at a time. Timing retains
nine CUDA-event and synchronized wall samples after three warmups. Graph results
use ten replays per sample. Eager and graph results are separate. Torch SDPA uses
its default dispatch, so differing score/value dimensions can select a slower
backend; this is not universally a Flash-only comparison. Training includes
forward, backward, clipping and optimizer updates; graph prefill includes model
kernels and layouts but does not persist a decode cache. Complete generation
measures four model decode tokens including cache writes, with an existing prefix.
The attention-only decode sweep measures read-only cache attention.

CUDA peaks count all currently live Torch allocations, including weights,
gradients, master/state buffers, output/workspace, and graph pool allocations.
Unused cuBLAS workspaces are cleared between phases. Driver/context allocations
outside Torch's allocator are excluded; the L40S has approximately 46 GiB usable
capacity. Peak allocations are not a device-wide memory measurement. Classifier
memory measurements include live input and classifier weights, dX/dW, and logits;
they do not include a model optimizer. Every latency regression is accepted only
with disclosure under the predeclared [performance policy](../plan/llt-readiness.md#qualification-performance-policy-fixed-before-final-runs).

## Validation and measured results

65 runtime/LLT/Torch adapter checks plus the additional unclipped-optimizer
regression pass (66 checks total). The existing non-opt-in suite passes
311 tests (386 skipped; optional hardware/profile checks are reported separately).
The BF16 fused-ReLU compiler defect, frontend-alignment validation defect,
zero-output manual-plan restriction, and NaN contamination from unused cache
slots are fixed and exercised by regressions. Raw summaries are in
[validation.json](data/llt-readiness/validation.json).

| Fixture | Steps | Largest loss drift vs Torch | Final parameter RMS difference | Resume error |
|---|---:|---:|---:|---:|
| llt | 1000 | 0.002981 | 0.000621 | 0 |
| naive | 1000 | 0.009289 | 0.000752 | 0 |

All four architecture/positional fixtures pass output and every-parameter
gradient checks; maximum gradient relative L2 across fixtures is 0.006902.
Tensor checkpointed gradients match its uncheckpointed gradients bitwise.
See [gradients.json](data/llt-readiness/gradients.json) and
[training.json](data/llt-readiness/training.json).

A cold attention forward/backward build takes 12.91 seconds including compilation
and initial setup. Cache reload reproduces gradients exactly; all reloads are hits.
See [cold.json](data/llt-readiness/cold.json) for warm samples.

### Attention forward and backward

Each row times the complete forward plus dQ/shared-dC backward. Graph ratios
above 1 are Tensor regressions. Both methods include their gradients and output
allocations; saved tensors and linear split-head workspace are separately recorded.

| Dtype | B / S / rank | Tensor graph ms | Torch SDPA graph ms | Ratio | Tensor / Torch eager peak MiB |
|---|---|---:|---:|---:|---:|
| float16 | 1 / 129 / 32 | 0.029 | 0.035 | 0.83 | 0.24 / 0.43 |
| float16 | 1 / 129 / 64 | 0.037 | 0.041 | 0.91 | 0.46 / 0.84 |
| float16 | 1 / 129 / 96 | 0.047 | 0.044 | 1.07 | 0.69 / 1.26 |
| float16 | 1 / 129 / 128 | 0.052 | 0.041 | 1.25 | 0.92 / 1.67 |
| bfloat16 | 1 / 129 / 32 | 0.029 | 0.035 | 0.84 | 0.24 / 0.43 |
| bfloat16 | 1 / 129 / 64 | 0.037 | 0.041 | 0.92 | 0.46 / 0.84 |
| bfloat16 | 1 / 129 / 96 | 0.047 | 0.044 | 1.07 | 0.69 / 1.26 |
| bfloat16 | 1 / 129 / 128 | 0.052 | 0.041 | 1.26 | 0.92 / 1.67 |
| bfloat16 | 4 / 257 / 64 | 0.042 | 0.050 | 0.83 | 5.68 / 6.18 |
| bfloat16 | 8 / 129 / 32 | 0.033 | 0.036 | 0.91 | 1.86 / 3.38 |
| bfloat16 | 1 / 513 / 64 | 0.051 | 0.061 | 0.83 | 2.84 / 2.96 |
| bfloat16 | 1 / 1025 / 128 | 0.132 | 0.101 | 1.30 | 11.29 / 11.54 |

[Attention raw results](data/llt-readiness/attention.json) retain all eager wall,
eager GPU, graph samples, saved tensor shapes/bytes, and peak allocations.
No saved tensor or workspace has two sequence dimensions. The shared-head
partition path adds linear FP32 workspace at contexts of 256 or more.

### Persistent attention and complete generation

Decode uses 16-row tensor-core tiles. Split candidates 32/64/128/256 were
numerically checked before timing; the selected default uses 64 splits for large
capacities with rank up to 64, otherwise 32. The warp prototype passes numerical
checks but was rejected for latency. Its bring-up exposed and fixed the pinned
NVRTC standard-trait omission. Historical candidate data is retained in
[tensor-core tuning](data/llt-readiness/decode-tuning.json),
[initial split rejection](data/llt-readiness/decode-splits-rejected.json), and
[warp rejection](data/llt-readiness/decode-simt-tuning.json). These are schedule
selection evidence preceding the final source revision, not final exit-gate runs.

| BF16 B / history / rank | Tensor graph ms | Torch graph ms | Ratio | Physical shared cache MiB |
|---|---:|---:|---:|---:|
| 1 / 4097 / 32 | 0.008 | 0.011 | 0.74 | 0.25 |
| 1 / 4097 / 64 | 0.009 | 0.012 | 0.74 | 0.50 |
| 1 / 4097 / 96 | 0.013 | 0.014 | 0.90 | 0.75 |
| 1 / 4097 / 128 | 0.012 | 0.015 | 0.80 | 1.00 |
| 1 / 65537 / 32 | 0.029 | 0.027 | 1.09 | 4.00 |
| 1 / 65537 / 64 | 0.040 | 0.029 | 1.36 | 8.00 |
| 1 / 65537 / 96 | 0.064 | 0.036 | 1.78 | 12.00 |
| 1 / 65537 / 128 | 0.075 | 0.037 | 2.03 | 16.00 |
| 4 / 1025 / 64 | 0.009 | 0.011 | 0.79 | 0.50 |
| 8 / 129 / 32 | 0.008 | 0.007 | 1.04 | 0.06 |

[Decode raw results](data/llt-readiness/decode.json) also include the matching
FP16 sweep, eager timings, temporary allocations, and graph peaks. Read-only
attention at 65K remains slower than Torch on the measured profiles; the worst
BF16 ratio is 2.03 at rank 128. Shorter-context graph attention is usually faster.
Eager host overhead remains material and is not removed by native preparation.

All four model-generation fixtures match full-prefix Tensor inference exactly
for four consecutive tokens after a real 17-token prefill. This small deterministic
fixture is not a trained text-quality result. Its complete calls include cache
append, projections, norms, MLPs, and classifier execution.

| Architecture / positional mode | Cache count | Allocated cache bytes | Four-token eager wall ms |
|---|---:|---:|---:|
| llt / absolute | 1 | 8200 | 20.591 |
| llt / RoPE | 1 | 20488 | 27.511 |
| naive / absolute | 6 | 196656 | 30.299 |
| naive / RoPE | 6 | 245808 | 41.478 |

[Generation raw results](data/llt-readiness/generation.json) include prefix
length, configuration, every output error and all timing samples.

### Full-model scaling and the classifier floor

These compare Tensor and Torch within each identical architecture/configuration.
The LLT and naive architectures use the same width, sequence, loop count and
vocabulary but have different parameter structures; this is not a matched-quality
or parameter-budget study. Eager full-step timings include checkpoint replay,
gradients, clipping, and AdamW. The prefill column measures graph replay.

| W / S / V / loops / rank / architecture | Tensor / Torch train wall ms | Tensor / Torch train peak MiB | Tensor / Torch prefill graph ms |
|---|---:|---:|---:|
| 64 / 33 / 256 / 1 / 32 / llt | 17.522 / 10.797 | 2.07 / 18.48 | 0.134 / 0.172 |
| 64 / 33 / 256 / 1 / 32 / naive | 16.095 / 9.864 | 2.16 / 18.57 | 0.126 / 0.141 |
| 64 / 33 / 256 / 1 / 64 / llt | 17.915 / 10.876 | 2.26 / 18.64 | 0.144 / 0.177 |
| 64 / 33 / 256 / 1 / 64 / naive | 15.877 / 9.584 | 2.16 / 18.57 | 0.126 / 0.141 |
| 64 / 33 / 256 / 2 / 32 / llt | 26.134 / 16.361 | 2.27 / 18.48 | 0.212 / 0.265 |
| 64 / 33 / 256 / 2 / 32 / naive | 28.721 / 16.653 | 2.37 / 18.62 | 0.238 / 0.264 |
| 64 / 33 / 256 / 2 / 64 / llt | 26.184 / 16.529 | 2.48 / 18.70 | 0.226 / 0.273 |
| 64 / 33 / 256 / 2 / 64 / naive | 28.477 / 16.757 | 2.37 / 18.62 | 0.238 / 0.264 |
| 64 / 33 / 256 / 10 / 32 / llt | 91.664 / 58.176 | 2.34 / 18.54 | 0.826 / 1.009 |
| 64 / 33 / 256 / 10 / 32 / naive | 124.477 / 70.554 | 2.44 / 18.68 | 1.134 / 1.243 |
| 64 / 33 / 256 / 10 / 64 / llt | 89.984 / 58.881 | 2.55 / 18.76 | 0.889 / 1.048 |
| 64 / 33 / 256 / 10 / 64 / naive | 124.661 / 69.191 | 2.44 / 18.68 | 1.134 / 1.243 |
| 64 / 33 / 256 / 20 / 32 / llt | 171.722 / 110.439 | 2.42 / 18.62 | 1.594 / 1.938 |
| 64 / 33 / 256 / 20 / 32 / naive | 245.089 / 137.424 | 2.52 / 18.76 | 2.254 / 2.466 |
| 64 / 33 / 256 / 20 / 64 / llt | 173.691 / 109.911 | 2.63 / 18.85 | 1.717 / 2.016 |
| 64 / 33 / 256 / 20 / 64 / naive | 245.091 / 136.912 | 2.52 / 18.76 | 2.254 / 2.466 |
| 512 / 257 / 4096 / 10 / 64 / llt | 126.437 / 70.471 | 172.84 / 185.41 | 5.657 / 2.066 |
| 512 / 257 / 4096 / 10 / 64 / naive | 126.674 / 70.992 | 181.89 / 195.90 | 5.784 / 2.060 |
| 512 / 257 / 50257 / 2 / 64 / llt | 59.519 / 33.515 | 870.34 / 1084.14 | 2.602 / 1.214 |
| 512 / 257 / 50257 / 2 / 64 / naive | 29.283 / 22.799 | 884.83 / 1097.64 | 2.340 / 0.900 |

[Full-model raw results](data/llt-readiness/systems.json) contain all 40 cases,
eager prefill wall/GPU timing, training GPU timing, and graph memory peaks.
Tensor eager training is slower than the corresponding Torch fixture in every
measured case (about 1.28–1.80×). Tiny-model prefill graphs are faster; width-512
prefill graphs are about 2.14–2.81× slower. These are accepted, published profile
regressions, not claims of end-to-end speed superiority.

The Torch reference recreates an approximately 16 MiB cuBLAS workspace during
small-model measurements. Its workspace counts in peak allocation, explaining
much of the apparent small-model memory advantage. Large-vocabulary differences
also reflect fused Tensor optimizer updates versus Torch temporary buffers.
Backend memory improvements must not be attributed solely to LLT absorption.
Across matched Tensor LLT/naive configurations, no case establishes LLT's combined
50% total-training-memory reduction and 20% step-latency reduction target.

The streaming classifier replaces checkpoint-per-chunk autograd accumulation
with explicit Tensor FP32 accumulation, bounding logits without a full-size
temporary gradient sum for every chunk. Full matrix-gradient outputs round through
BF16 consistently with the model policy; summing rounded chunk gradients can
differ slightly from a single full GEMM, within the tested tolerance.

| Rows / chunk rows | Classifier peak MiB | Eager wall ms |
|---|---:|---:|
| 129 / full | 258.39 | 2.313 |
| 129 / 32 | 249.00 | 11.489 |
| 1025 / full | 349.41 | 6.495 |
| 1025 / 32 | 252.51 | 73.729 |

At 1,025 rows, chunking reduces this classifier peak by 27.7% but costs about
11× time; weights and their gradient remain the memory floor. It is an optional
memory tradeoff, not the default path. The original checkpoint alternative
increased memory and was rejected:
[rejected measurements](data/llt-readiness/classifier-checkpoint-rejected.json).
[Final classifier measurements](data/llt-readiness/loss-memory.json) include nine
samples for full/chunked 129 and 1,025 row cases.

### Distribution and steady state

Both training fixtures show zero allocation growth after 20 warmup steps followed
by 100 checked steps; see [leak.json](data/llt-readiness/leak.json). A clean installed
consumer contains only the runtime/adapter wheels, NumPy, Torch and Torch's runtime
dependencies. Compiler imports are blocked. Both model backward/optimizer paths
and the streaming classifier pass using cache hits and matching native plans.
See [consumer.json](data/llt-readiness/consumer.json). The [audit](data/llt-readiness/audit.json)
validates 468 unique artifact hashes/SM targets/ABI requirements, source snapshots,
every sample count, sustained training/resume, and wheel source identity.

## Execution coverage and limitations

PyTorch owns tensor allocation, layout/concatenation, low-precision gradient casts,
autograd scheduling and tied-gradient sums, checkpoint control, RNG, optimizer
parameter groups/hyperparameter uploads, and serialization. Explicit Tensor
operators report every actual kernel and layout normalization copy; their semantic
fallback list is empty in qualification. Generic FX coverage is narrower and is
not claimed to lower the full LLT graph automatically.

The supported profile is single-GPU FP16/BF16 contiguous positive-extent BHSD,
first-order autograd, no dropout/general additive mask, score dimensions divisible
by 16 up to 256 and value dimensions up to 128. Tested ranks are 32/64/96/128;
other allowed dimensions still need workload-specific performance validation.
Persistent caches support variable batch lengths, capacity checks, poisoned unused
slots, reset/reuse, and captured append; the host must check overflow after replay.
Paged serving, higher-order derivatives, general stochastic checkpointing, other
SM targets, and distributed training are outside this dependency gate.

The RoPE model stores concatenated latent/positional keys and separate latent
values, duplicating the latent once; reported cache sizes include this cost.
Further packed positional storage is an optimization, not an uncounted saving.
Folded weights and cached prefixes are invalidated on parameter version changes.
The pinned TileLang fragment extraction limitation remains; the reproducer fails
as expected and the shared-memory workaround passes decode. No upstream compiler
fix is claimed.

## Reproduction

```bash
bash benchmarks/llt/run.sh
```

This runs operator checks, cold/cache load, gradients, 1,000-step training, real
prefix generation, attention/decode, classifier-memory, and full-model scaling.
It assumes the locked producer environment, NVRTC bundle, and native adapter have
been installed. Build the runtime and native adapter wheels into `build/llt-wheels`,
install only those wheels, NumPy, and Torch into a clean Python environment, and
copy `benchmarks/llt/consumer.py` plus `model.py` (as `model_fixture.py`) outside the
source tree. Run the consumer against the warmed qualification artifact cache,
then `python -m benchmarks.llt.audit`. The executable
`bash benchmarks/llt/installed-consumer.sh` reproduces this wheel/consumer check. The consumer blocks compiler imports and
verifies installed package paths, every cache hit, and matching native plans.
Raw data and source snapshots live entirely in this Tensor repository.
