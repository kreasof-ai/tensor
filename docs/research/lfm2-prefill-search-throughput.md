# Searched LFM2 kernels: full-model prefill and decoding throughput

The searched Tensor schedules now run in the full LFM2.5-230M F16 model on
this RX 6700 XT. They improve warmed prefill by **45–47%**. Replaying tinygrad's
searched FFN schedules improves its full-model prefill by **19–28%**. Neither
search changes the decode kernels. llama.cpp remains faster in both phases.

These are measured end-to-end rates, replacing the estimates derived from the
[isolated projection comparison](lfm2-tensor-search-comparison.md).
The subsequent [decode search](lfm2-230m-decode-search.md) retains these prefill
schedules and improves decoding through a separate opt-in profile.
[Raw evidence](data/lfm2-prefill-search-throughput.json) contains every timing
sample, all accuracy metrics, bundle manifests, replayed schedules, source
snapshots and unchanged decode shader hashes.

## Prefill

Tokens per second; higher is better. The searched columns use the schedules
from the completed Tensor/tinygrad searches, with the fusion adjustment below.

| Prompt tokens | Tensor before | Tensor searched, Vulkan | tinygrad before, OpenCL | tinygrad searched, OpenCL | llama.cpp Vulkan |
|---:|---:|---:|---:|---:|---:|
| 32 | 1,706 | **2,466** | 971 | **1,154** | **3,124** |
| 128 | 1,774 | **2,612** | 1,296 | **1,617** | **4,170** |
| 384 | 1,745 | **2,549** | 1,405 | **1,803** | **4,491** |

At 384 tokens, Tensor improves **1.46×** and tinygrad **1.28×**. Tensor's
prefill gap to llama.cpp falls from **2.57× to 1.76×**. Tensor's searched
full-model prefill is **1.41×** the searched tinygrad adapter's rate, although
tinygrad's isolated searched FFN GEMMs were faster. Submission, fusion,
attention and the other projections contribute to the whole-model result.

## Decoding

Tokens per second for 64 identical forced tokens after each prompt. Sampling
is excluded; each token call completes and returns FP32 logits to the host.

| Prompt tokens | Tensor before | Tensor searched profile | tinygrad before | tinygrad searched profile | llama.cpp Vulkan |
|---:|---:|---:|---:|---:|---:|
| 32 | 281.9 | **283.6** | 40.1 | **40.0** | **482.4** |
| 128 | 266.6 | **269.4** | 40.0 | **40.0** | **482.9** |
| 384 | 238.3 | **236.9** | 39.8 | **39.8** | **479.7** |

Decode is effectively unchanged: these searches optimized 32-row prefill
GEMMs, while decode uses the separate one-row GEMV path. All Tensor decode
shaders are byte-identical between the baseline and searched bundles. The
small rate differences do not demonstrate a decode improvement. At 384
tokens, the remaining native decode gap is **2.03×**.

## What was integrated

Tensor adds the explicit producer option `--webgpu-profile searched` on top of
the existing subgroup profile. Selection is restricted to native F16 weights,
32 rows, and the 230M FFN shapes. Other profiles retain their defaults.

- Gate/up: the isolated gate winner uses an M16/N8 tile, 128 threads, 16 K
  partitions, vec4 dots, unroll 4 and blocked K. The actual model fuses gate,
  up and SwiGLU. A validated M8/N8 variant performs better for that fusion,
  which doubles accumulators and shared partials. The fused path uses M8.
- Down: M4/N8, 64 threads, 16 K partitions, vec4 dots, unroll 4 and striped K.
- Accumulators and output remain FP32. Prefill operands retain the independent
  oracle's nearest-even FP16 rounding, including subnormals.

tinygrad replays the exact cached beam-8 projection ASTs from its earlier
30-minute experiment. No new search occurs during inference. It materializes
FP32 FFN inputs and outputs so those ASTs match; the residual is completed
before normalization to keep convolution/state writes ordered. Both gate/up
and down match the saved schedules. All other kernels use the ordinary
heuristics. The raw report records the matched AST hashes and applied opts.

The before/after tinygrad rows both retain **native FP16 weight storage**.
They supersede the historical full-model rows whose adapter accidentally
expanded F16 matrices to FP32; that correction was already documented in the
[earlier report](lfm2-tinygrad-comparison.md).

## Correctness and comparison scope

Both Tensor profiles and both tinygrad modes pass all **19 independent NumPy
fixtures**, including partial chunks, convolution and attention state, cached
decode through a 511-token prefix, matching argmax, and bitwise reset.
The unchanged gates are relative RMS below 1%, cosine above 0.9999 and finite
logits. The separate fused-FFN check passes four independent inputs/scales,
including ordinary and FP16-subnormal inputs, with a propagated float64
reference error bound and NaN output sentinels.

| Runner | Maximum relative RMS vs NumPy | Minimum cosine | Matching argmax |
|---|---:|---:|---:|
| Tensor before | 0.1905% | 0.99999821 | 19/19 |
| Tensor searched | 0.1937% | 0.99999815 | 19/19 |
| tinygrad before | 0.2209% | 0.99999756 | 19/19 |
| tinygrad searched | 0.2013% | 0.99999799 | 19/19 |
| llama.cpp default Vulkan arithmetic | 4.7380% | 0.99895662 | 19/19 |

The native row uses llama.cpp's ordinary inference precision, including its
default Vulkan matmul accumulation and flash attention. Its accuracy metrics
are recorded separately; it does **not** pass the stricter Tensor/NumPy gate.
This is a same-checkpoint throughput comparison, rather than a claim of
identical internal arithmetic. The earlier isolated kernel comparison forced
llama.cpp FP32 accumulation; those microsecond figures are a different test.

tinygrad is a benchmark-only LFM2 adapter using its generic OpenCL compiler.
It scans fixed, masked 576-entry KV buffers; Tensor bounds attention to the
active prefix, and llama.cpp uses its native flash attention. This measures
the available complete implementations on this Windows/RDNA2 machine. It
does not represent tinygrad's specialized native AMD LLM kernels.

## Measurement protocol and reproduction

Windows, Ryzen 5 5600, RX 6700 XT, Vulkan driver 26.6.2, OpenCL driver
3652.0 (PAL,LC); Tensor uses wgpu 0.29.0 and its native prepared encoder.
The source base is `f9258441d599463f3802d958ec35ef1473fcb57d` plus the source
snapshots in the raw evidence. tinygrad is pinned to
`91b8cb5fa6c031c5a7440159d955f66952c5e2e9`; llama.cpp b11310 to
`f872b591121761ac7b2af18283bd99bdc092a63a`.
The shared model SHA256 is
`4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f`.

Each rate is token count divided by the median of **seven** measured completed
durations after **three** warmups. All five runners rotate order and execute
sequentially on the GPU. Context is 512, prompt chunks are 32 for every runner,
and prefill ends with host FP32 logits. Decode uses 64 shared forced tokens.
Model loading, Tensor AOT compilation, and first-call compilation/JIT capture
are excluded from warmed timings. Reset occurs outside the prefill timer.
The report retains observed first calls, but shared driver/compiler caches
make those observations unsuitable for a clean cold-start comparison.

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-prefill-search/baseline --context 512 --provider webgpu --webgpu-profile subgroup
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-prefill-search/searched --context 512 --provider webgpu --webgpu-profile searched
$env:PYTHONPATH=Join-Path (Get-Location) 'build/tinygrad-comparison/upstream'
$env:DEV='CL'
$env:BEAM='0'
$env:JITBEAM='0'
$env:BEAM_ESTIMATE='0'
$env:PARALLEL='0'
$env:WGPU_BACKEND_TYPE='Vulkan'
$env:CACHEDB=Join-Path (Get-Location) 'build\lfm2-prefill-search\reproduce-cache.db'
.venv/Scripts/python.exe benchmarks/lfm2/tinygrad_compare.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --bundle build/lfm2-prefill-search/searched --baseline-bundle build/lfm2-prefill-search/baseline --reference build/llama-vulkan-b11310 --fixtures build/lfm2-230m-f16-webgpu-validation --out build/lfm2-prefill-search/comparison --tinygrad-search-root build/tinygrad-comparison/long-search-30m --repeats 7 --decode 64
.venv/Scripts/python.exe benchmarks/lfm2/prefill_schedule_check.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-prefill-search/fused-check
.venv/Scripts/python.exe benchmarks/lfm2/prefill_search_summary.py --root build/lfm2-prefill-search --out docs/research/data/lfm2-prefill-search-throughput.json
```

The checkpoint, pinned native release, independent NumPy fixtures and tinygrad
search caches must already exist at these paths. Their hashes and selected
schedules are retained in the linked experiment reports.
