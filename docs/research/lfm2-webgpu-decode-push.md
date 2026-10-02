# Further LFM2 WebGPU decode optimization on RX 6700 XT

On October 3, 2026, the ordinary `tensor_llm.LFM2` WebGPU runner improves
2.6B Q4_0 decode from **128.0 to 139.9 tokens/s** at prefix 128 (+9.3%).
The matched llama.cpp Vulkan reference stays near 169 tokens/s, reducing the
decode gap from **1.32x to 1.21x**. Parity still requires about 17.5% less
time per token. The 230M Q4_0 workload also improves; F16 forced decode is
essentially unchanged at this prefix.

The baseline is `3eef0c2a65410050325cd9df8ba3fe5d07c0c006`, including the four
recent optimization/documentation commits. This is a fresh before/after run,
following the [previous 2.6B report](lfm2-2.6b-q4_0-matched-run.md).
[Raw evidence](data/lfm2-webgpu-decode-push.json) contains every matched timing
sample, validation metric, implementation fingerprint, kernel sweep and profile.

## Changes kept

For subgroup Q4_0 decode with both matrix dimensions at least 2048 and depth
divisible by 256, each output row uses 32 lanes instead of 16. Each lane still
unpacks four bytes and consumes eight FP32 activations, but the K loop advances
by 256 rather than 128 elements. The workgroup remains 128 threads. Narrower
matrices and the portable profile retain the previous default schedule.

Each packed word supplies two four-element floating-point dot products. The
FP16 block scale multiplies each dot result, replacing eight separately scaled
scalar products. This changes FP32 summation order, so it is validated against
the existing numerical limits. Activations remain FP32; no Q8 activation
quantization or relaxed accuracy gate is introduced.

The FFN down projection adds the residual directly in its output epilogue.
This uses the same FP32 addition as the separate add kernel and removes one
dispatch per layer. Single-row decode also passes the hidden state directly
to the final normalization, removing the last-row copy. Decode dispatches fall
from 285 to 254 for 2.6B, 147 to 132 for 230M Q4_0, and 175 to 160 for 230M F16.
Prefill plans retain 307/155/155 dispatches.

The producer exposes explicit schedule overrides for reproducible sweeps. The
8-lane, 64/256-thread and four-accumulator alternatives remain opt-in tuning
candidates; the default selects only the measured 32-lane floating-dot route
for the larger matrices.

## Matched measurements

Windows 11, Ryzen 5 5600, RX 6700 XT 12 GB, AMD Vulkan driver 26.6.2, Python
3.12.13, NumPy 2.5.3, wgpu 0.29.0. Both Tensor runs use the native prepared-plan
encoder. The llama.cpp b11310 reference and its configuration are unchanged:
all layers on Vulkan, six CPU threads, flash attention and F16 KV caches.

Context is 512, prefill chunks are 32 tokens, and every forced decode call
returns completed host FP32 logits. Each cell is a median of five repetitions
after one warmup. Tensor and llama.cpp run sequentially in alternating order;
loading and sampling are excluded. The before and after sessions are separate.

| Model | Prefix | Tensor before, tok/s | Tensor after, tok/s | Change | llama.cpp after, tok/s |
|---|---:|---:|---:|---:|---:|
| 2.6B QAD Q4_0 | 32 | 131.4 | 143.9 | +9.5% | 169.3 |
| 2.6B QAD Q4_0 | 128 | 128.0 | 139.9 | +9.3% | 169.5 |
| 2.6B QAD Q4_0 | 384 | 118.7 | 128.7 | +8.4% | 168.8 |
| 230M Q4_0 | 32 | 458.5 | 496.1 | +8.2% | 777.5 |
| 230M Q4_0 | 128 | 420.0 | 460.0 | +9.5% | 770.4 |
| 230M Q4_0 | 384 | 371.0 | 382.0 | +3.0% | 763.5 |
| 230M F16 | 32 | 285.3 | 279.8 | -1.9% | 482.1 |
| 230M F16 | 128 | 268.5 | 268.8 | +0.1% | 487.0 |
| 230M F16 | 384 | 236.5 | 237.5 | +0.4% | 476.5 |

The 230M Q4_0 native reference also varies by up to 2.9% between sessions;
those smaller-model gains should be interpreted with that variation in mind.
The 2.6B native decode reference changes by less than 0.1% at prefix 128.

Full generation uses the same prompt and exactly the same greedy tokens before
and after. Medians include tokenization, reset, prefill, sampling and completion:

| Model | GPU greedy before, seconds | GPU greedy after, seconds | Less time |
|---|---:|---:|---:|
| 2.6B QAD Q4_0 | 0.8643 | 0.8052 | 6.8% |
| 230M Q4_0 | 0.1079 | 0.0933 | 13.5% |
| 230M F16 | 0.1559 | 0.1489 | 4.5% |

Prefill is unchanged: at prefix 128, 2.6B measures 232.0 to 231.8 tok/s,
230M Q4_0 1702.1 to 1720.7, and 230M F16 1773.6 to 1770.2. These are decode
kernel and plan changes. They do not change the generic GEMM or attention
shaders measured in [latency scaling](latency-scaling.md), so that sweep was
not repeated for this change.

## Kernel selection and profiling

`webgpu_decode_tune.py` streams every matching layer's actual packed matrices
in a prepared plan, avoiding a single repeatedly cached matrix. Its candidates
are compared to a float64 CPU calculation using FP32 activations and decoded
weights. A continuous one-second GPU warmup follows each compilation, then
seven timing batches follow one discarded batch. Each batch launches ten plans
and includes completion. Warmup was added after the preliminary smaller-model
sweep exposed GPU clock ramp-up.

For 2.6B, the final dot sweep measures the complete 30-layer paired FFN at
2.623 ms for the original schedule, 2.427 ms for 32 lanes with scalar products,
and 2.190 ms for 32 lanes with floating dots. The down projection improves
only slightly, 1.387 to 1.338 ms. The 22 short-convolution input projections
improve from 0.705 to 0.605 ms at the selected 128-thread workgroup. Narrower
230M shapes do not show a consistent benefit from this larger-row schedule,
so their default stays at 16 lanes.

Eight lanes regress the 2.6B down projection to about 2.06 ms. Four independent
accumulators improve the scalar FFN slightly but lose to floating dots. Larger
workgroups do not improve the selected dot FFN enough to justify a change.
All 114 candidates in the warmed scalar/dot sweeps pass their numerical gate;
correctness alone does not make them good default schedules.

Seven-repeat timestamp profiles at decode position 128 confirm GPU execution
falls from **6.361 to 5.839 ms** (-8.2%). Instrumented FFN time falls from
2.496 to 2.118 ms. Linear plus fused residual projections total 3.136 ms after,
versus 3.261 ms before. The whole prefill pass at position 96 is essentially
identical: 137.788 versus 137.749 ms. These profiles use the measured Vulkan
timestamp period of 10 ns; the matched forward benchmark is the acceptance
measurement because per-dispatch instrumentation can perturb execution.

## Accuracy and consumer checks

All **19 fixtures pass for each of the three models**. The gates remain relative
RMS below 0.01, cosine above 0.9999, identical argmax and finite logits.
Maximum relative RMS after is 0.004446 for 2.6B, 0.004991 for 230M Q4_0 and
0.001905 for F16. Reset remains bitwise deterministic. GPU and host greedy
generation agree and preserve the before-run token sequences.

Independent NumPy logits were reused from the previously passed reports for
these exact GGUF files. The replay verifies model SHA-256, context, the full
ordered token/reset fixture sequence, the original accuracy gates and finite
arrays. It records every reference-array and original report hash. Before and
after use identical hashes. llama.cpp logits are recomputed in each run.
This is cached independent reference validation, not a fresh NumPy execution.
Omitting `--numpy-fixtures` retains the original fresh-reference behavior.

The full Tensor LLM suite plus fixture-replay tests passes **81 tests**, with
four CUDA tests skipped on this Radeon. Focused tests cover Q4 subnormal and
negative scales, unaligned packed blocks, partial output rows, floating-dot
and accumulator variants, portable reductions and forced subgroup fallback.
Residual fusion is bitwise checked against a separate add for F32, F16, Q4_0
and Q6_K.

A newly built Tensor LLM wheel is installed alongside the existing matching
native Tensor runtime wheel in a clean consumer. TVM, TileLang and Torch are
absent. All three models load their AOT bundles, report native encoding, and
match both GPU and host greedy generation from the measured runs. The raw
evidence records the wheel hashes. The standard local WebGPU bundles were
updated to these artifacts; existing external bundles require rebuilding.

## Reproduction

From the repository root, build a matching subgroup bundle, then run the
unchanged matched benchmark. Use the existing local 2.6B model and the raised
buffer limit for its 215 MB Q6_K embedding:

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
$model='D:/LLM/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf'
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model $model --out build/lfm2-2.6b-q4_0-webgpu --context 512 --provider webgpu --webgpu-profile subgroup
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_run.py --model $model --bundle build/lfm2-2.6b-q4_0-webgpu --reference build/llama-vulkan-b11310 --out build/lfm2-decode-repeat --max-buffer-size 268435456 --numpy-fixtures build/lfm2-2.6b-q4_0-run-prefill
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_decode_tune.py --model $model --out build/lfm2-decode-tune-repeat --preset dot
```

For 230M, use the F16 or Q4_0 file under `build/lfm2-230m-models`, its matching
bundle/reference-fixture directory, and omit the larger buffer limit. The
tuner uses explicit original schedule parameters so future default changes
do not silently alter its baseline. Fresh before measurements additionally
require the baseline revision's matching Tensor LLM package and AOT bundle.
