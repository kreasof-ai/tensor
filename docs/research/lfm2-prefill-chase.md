# LFM2.5-230M prefill: transferring the CLBlast chase

[Research index](README.md) · [FP32 GEMM experiment](webgpu-outer-product-gemm.md) ·
[Previous runtime/decode result](lfm2-230m-runtime-search.md)

On 2026-10-03, transferring the compiler scheduling work into native-F16
LFM2.5-230M inference on the RX 6700 XT improves completed prefill by
**15.9% at 32 tokens and 2.18–2.19× at 128/384 tokens**. The larger gain is
primarily from larger prefill chunks; selected projection schedules add another
9–16%. The existing fused FFN remains the winner. Blanket explicit unrolling
regresses full-model throughput and is not the retained profile.

Tensor exceeds the earlier 32-chunk llama.cpp baseline on long prompts, but
larger chunks also accelerate llama.cpp. Against its best tested chunk size,
the remaining prefill gaps are **1.07×, 1.12× and 1.50×** at 32, 128 and
384 tokens. This establishes substantial progress, not universal native parity.

## Independent adaptive-bundle repeat

All rates are tokens/s, measured on completed forwards returning host FP32
logits. Reset, loading, compilation, initial calls and sampling are excluded.
The same checkpoint and forced token IDs run on every backend, with context
512, three warmups and seven timed samples. Runner order rotates sequentially
on the GPU. Native llama.cpp b11310 uses ordinary Vulkan arithmetic, all layers
offloaded, F16 KV caches and Flash Attention. Its batch and microbatch sizes
both match the tested chunk, rather than fixing its baseline at 32.

| Prompt tokens | Tensor before, chunk 32 | Tensor adaptive | Gain | llama.cpp chunk 32 | Best tested llama.cpp |
|---:|---:|---:|---:|---:|---:|
| 32 | 2,536 | **2,939** | **1.16×** | 3,129 | **3,140** (chunk 64) |
| 128 | 2,635 | **5,755** | **2.18×** | 4,184 | **6,431** (chunk 128) |
| 384 | 2,557 | **5,599** | **2.19×** | 4,499 | **8,394** (chunk 128) |

The short-prompt native chunk-32/64 difference is small; picking the largest
observed median is a bounded comparison, not proof of the native global optimum.
At 128 tokens the new Tensor forward takes **22.24 ms**, versus **19.90 ms**
for native chunk 128. At 384 tokens it takes **68.59 ms**, versus **45.75 ms**.

The adaptive Tensor bundle contains rows **1, 32 and 128**. Forward selects
the largest compiled prefill chunk fitting the remaining tokens, uses the
smallest prefill chunk for shorter tails, and uses the decode plan for a
single-token tail. It therefore avoids making a short prompt execute a padded
128-row projection. Chunk selection is a simple policy, not a latency-optimal
search for every possible prompt length; intermediate-length throughput was
not benchmarked. All partial-chunk and cache-boundary correctness fixtures pass.

## Ablations: kernels versus chunking

The initial sweep independently measures nine Tensor variants and three native
chunk sizes. Its selected fixed-row results agree with the adaptive repeat:

| Prompt | Existing schedules, chunk 32 | Existing schedules, chunk 128 | Selected schedules, chunk 32 | Selected schedules, chunk 128 |
|---:|---:|---:|---:|---:|
| 32 | 2,524 | 1,341 | **2,929** | 1,462 |
| 128 | 2,623 | **5,314** | 3,019 | **5,788** |
| 384 | 2,565 | **5,171** | 2,928 | **5,650** |

Larger chunks alone approximately double long-prompt throughput. They reuse
weights across more rows and reduce complete model submissions and repeated
last-row/logit work. The selected schedules improve the same 128-row chunk by
8.9–9.3%, while improving the 32-row chunk by 14.2–16.1%. This is not the
3.37× gain measured against generic FP32 GEMM: LFM2 already had tuned F16 FFNs.

Applying explicit unrolling across all projections and FFNs gives only
**2,419 / 3,516 tokens/s** at a 128-token prompt for chunk 32/128, respectively,
versus existing-schedule **2,623 / 5,314**. The grammar produces numerically
valid kernels but poorer full-model performance. It remains an experimental
`prefill_unrolled` profile for reproduction, rather than a default choice.

## Compiler and inference integration

The staged outer-product helper now accepts an activation load/rounding
expression, allowing an FP32 activation ABI with F16 shared staging and native
F16 weights. LFM2 supplies its existing integer nearest-even half-rounding
expression, including subnormals. Accumulation and output stay FP32.
The existing paired source transformation retains gate/up/SwiGLU fusion;
this experiment does not add an extra activation materialization or upload.

Three producer profiles make the comparison explicit:

- `prefill_chunked`: retain the current searched F16 schedules at larger rows.
- `prefill_unrolled`: expand bounded marked loops, including larger FFN unrolls.
- `prefill_outer`: retain only shape-specific replay winners, falling back to
  the current searched schedules elsewhere.

The independent replay examines **246 valid candidates** over five operation
groups and rows 32/64/128. Timings stream every matrix in the group with
distinct output buffers, rather than repeatedly timing just one hot weight.
Each candidate passes a float64 oracle. Control and the best three rotate on
fresh replay; each selected winner additionally passes **612 combined checks**
across all affected weights, three held-out seeds and scales .01, 1 and 1e-5.
Nine operation/row/shape selections exceed the **3% replay-improvement gate**.

The retained changes are explicit unrolling for the 32-row convolution input
projection and outer products for selected attention/convolution projections.
At 32 rows, the 1024→3072, 1024→1024 and 1024→512 group medians improve
from **152.10→128.77**, **104.21→77.21**, and **94.09→70.77 µs**.
No candidate wins the fused FFN; the existing partitioned fused schedule is
retained. The small FFN-down improvements do not exceed the replay gate.
Selection is specialized to the measured F16 shapes, not a universal schedule.

Experimental profiles accept multiple sorted prefill rows. The standard
WebGPU and CUDA row profiles retain their existing defaults. The new inference
loop also preserves those single-prefill-row execution sequences.

## Accuracy and decoding

Both full-model experiments pass all **19 independent NumPy fixtures**,
including lengths 31/32/33, 127/128/129, cached decoding through position 511,
partial chunks and deterministic reset. Every Tensor runner meets the existing
1% relative-RMS, cosine > .9999, finite-output and argmax gates. All Tensor
profiles reproduce the same **39-token greedy response**, and their GPU and
host greedy paths agree.

| Final runner | Maximum relative RMS | Minimum cosine | Matching argmax |
|---|---:|---:|---:|
| Tensor before | 0.1937% | 0.99999815 | 19/19 |
| Tensor adaptive | **0.2205%** | **0.99999762** | **19/19** |
| llama.cpp chunk 32/128 | 4.7380% | 0.99895662 | 19/19 |
| llama.cpp chunk 64 | 4.7619% | 0.99889999 | 19/19 |

These remain same-checkpoint throughput comparisons with different arithmetic
precision. Ordinary native Vulkan does not satisfy Tensor's stricter 1% gate.
The gap is not a precision-equivalent parity claim.

Decode does not materially improve:

| Prefix | Tensor before decode | Tensor adaptive decode | llama.cpp chunk-32 decode |
|---:|---:|---:|---:|
| 32 | 437.1 | 438.7 | 484.8 |
| 128 | 437.6 | 437.4 | 487.4 |
| 384 | 429.0 | 433.2 | 478.0 |

Decode projection WGSL is byte-identical across the variants. The adaptive
bundle allocates cache capacity 640 rather than 576 to support row 128, so
capacity-dependent attention shaders differ. The modest decode differences
do not establish a new decode optimization. Dispatch counts remain **132**
for decode and **155** per prefill chunk; long prefill needs fewer chunks.

The operator suite passes 12 native GPU cases, including fused/nonfused mixed
activation outer products and tails. The related CPU regression subset passes
60 tests. Logs and numerical checks are retained in the raw evidence.

## Remaining work

A diagnostic timestamp profile of row 128 at active prefix 256 measures a
**23.50 ms whole compute pass**. Instrumented per-dispatch medians sum to
24.20 ms, so their fractions are approximate and include instrumentation effects.
Matrix projections account for about **77.6%** of that sum; fused FFN plus its
down projection alone account for **41.5%**, attention **12.7%**, and RMS
normalization **7.3%**. Convolution is about 1.2%. The existing fused FFN and
down schedules are consequently still major targets, followed by attention
and parallel prefill normalization. More aggressive unrolling alone did not
solve their compute cost. This profile is diagnostic, not the acceptance timer.

## Evidence and reproduction

[Raw replay, sweep, adaptive comparison, shader/source snapshots and test logs](data/lfm2-prefill-chase.json).
Hardware is RX 6700 XT, Ryzen 5600, Windows, AMD Vulkan 26.6.2 and wgpu 0.29.0,
using the native prepared-plan encoder. The model is the existing native-F16
checkpoint with SHA256
`4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f`.
GPU jobs run sequentially. CPU references use `OPENBLAS_NUM_THREADS=6`.

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
$env:OPENBLAS_NUM_THREADS='6'
.venv/Scripts/python.exe benchmarks/lfm2/prefill_chase_search.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-prefill-chase/search
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-prefill-chase/baseline32 --provider webgpu --context 512 --webgpu-profile decode_searched
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-prefill-chase/adaptive --provider webgpu --context 512 --webgpu-profile prefill_outer --prefill-chunks 32 128
.venv/Scripts/python.exe benchmarks/lfm2/prefill_chase_compare.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --root build/lfm2-prefill-chase --reference build/llama-vulkan-b11310 --fixtures build/lfm2-230m-f16-webgpu-validation --out build/lfm2-prefill-chase/adaptive-comparison --bundles baseline32 adaptive --repeats 7
```

Use `--prefill-chunk 64` or `128` with `prefill_chunked`, `prefill_unrolled`
and `prefill_outer` to build the fixed-row ablations. The stored first-sweep
source snapshots preserve the code before adaptive selection was added.
The summary script also expects that complete variant sweep. To use the
accepted experiment, load the adaptive bundle through the ordinary `LFM2`
runner or `python -m tensor_llm generate`; it selects rows from the bundle.
Context 512 and this 230M F16 checkpoint are the measured acceptance scope.
