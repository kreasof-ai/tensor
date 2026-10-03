# LFM2.5-2.6B QAD Q4_0: approaching Vulkan decode parity

[Research index](README.md) · [Previous repeat](lfm2-2.6b-q4_0-revisit.md) ·
[Raw measurements, sources and generated WGSL](data/lfm2-2.6b-q4_0-parity.json)

On 2026-10-03, the RX 6700 XT reaches **164.2 tokens/s decode against
llama.cpp Vulkan's 169.8** at prefix 128. This is **96.7% of native throughput**,
up 12.3% from the current adaptive control. At prefix 384, the gain is 19.4%.
Prefill reaches **522/517 tokens/s** for 128/384 tokens, a further **51%** gain
over the same larger-chunk control. **Actual parity is still unproven**:
native leads decode by 3–5% and long-prompt prefill by 3.73–4.20×.

The opt-in `quant_searched` producer profile combines exact packed Q4/Q6
decode schedules, fused attention scores/values and staged Q4 prefill outer
products. The optional C helper now also creates, submits and releases fresh
command handles. Default producer profiles remain available.

## Completed prefill and decoding

All rates are tokens/s. The local checkpoint is the same **QAD Q4_0** file
as the previous repeat, SHA256
`a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03`.
It has 30 layers, width 2048, FFN width 10752, 32 query heads, eight KV heads
and vocabulary 128000. Its 166 Q4_0 matrices and one Q6_K tied embedding/output
remain packed; inference does not retain expanded weights. The Q6_K matrix is
215,040,000 bytes and needs `max_buffer_size=268435456`.

Each forward completes and returns host F32 logits. Context is 512, decode
uses the same 64 forced tokens after each prefix, and three warmups precede
seven measured samples. Runner order rotates sequentially on the GPU.
Compilation, loading, reset and sampling are excluded. The final comparison
loads three Tensor runners and one native runner. Native b11310 uses Vulkan
for all layers, Flash Attention, F16 K/V, six CPU threads and batch/ubatch 128.

| Prompt tokens | Tensor chunk 32 control | Tensor adaptive control | Tensor searched | Search / adaptive | llama.cpp chunk 128 |
|---:|---:|---:|---:|---:|---:|
| 32 | 231.1 | 231.1 | **291.7** | **1.26×** | 933.6 |
| 128 | 232.0 | 345.2 | **522.1** | **1.51×** | 1,945.0 |
| 384 | 230.4 | 341.6 | **517.3** | **1.51×** | 2,174.0 |

| Prefix tokens | Tensor chunk 32 control | Tensor adaptive control | Tensor searched | Search / adaptive | llama.cpp | Tensor / native |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 150.9 | 151.0 | **165.6** | **1.10×** | 170.1 | **97.4%** |
| 128 | 146.9 | 146.2 | **164.2** | **1.12×** | 169.8 | **96.7%** |
| 384 | 134.9 | 134.6 | **160.6** | **1.19×** | 168.9 | **95.1%** |

Both adaptive bundles use rows 1/32/128 and allocate 1,633,086,092 bytes.
The chunk-32 control allocates 1,603,438,732 bytes. Searched decode uses 246
dispatches versus 254; prefill still uses 307 per chunk. Thus the 51% prefill
improvement isolates schedule changes at identical chunk sizes. Against the
chunk-32 control, long-prompt prefill improves approximately 2.25×.

Separate healthy repeats measured searched decode at 159–167 tokens/s with
native at approximately 169–171. The earlier kernel-only comparison loaded
two Tensor and three native runners and reached 159.3 tokens/s at prefix 128;
the identical-shader runtime ablation below reached 165.1. These separate
sessions are retained, but their difference does not isolate a runtime gain.
A six-runner attempt caused severe memory pressure, including native falling
to roughly 13 tokens/s, and was excluded. The final report reduces residency
and retains a native rate consistent with the other healthy repeats.

## What search transferred into inference

Decode discovery tests **155 candidates**, including alternate row ownership,
explicit K unrolling and independent F32 accumulation chains. Groups stream
every affected matrix with distinct outputs; finalists rotate against the
control in fresh timestamp measurements, then face three held-out input scales.
Selected Q4 gains range from 1.09× for FFN down to 1.80× for the narrow 512-column
projection. The paired gate/up candidate gains only 1.014× and is not selected.

A second **52-candidate Q6_K search** groups adjacent decoded values into
four-wide floating dots. It retains native packed scales and signed integer
fields, uses F32 input/accumulation and introduces no activation quantization.
The selected 256-thread, four-step unroll reaches **574 µs** for the full
215 MB output matrix, versus **800 µs** for the original control: **1.39×**.

Attention discovery tests 22 schedules at five positions, including poisoned
inactive K/V entries. The selected fused score/value schedule uses 64 channels
and eight value partitions. It removes one dispatch from each attention layer,
accounting for the eight fewer decode dispatches.

Prefill discovery evaluates **150 candidates** across five projection families
at rows 32/128. The compiler outer-product helper now accepts a packed RHS load
expression and a value transform. Q4/Q6 decode to F32 before nearest-even F16
rounding into shared tiles; the activation ABI and output remain F32, with F32
accumulation. Paired gate/up/SwiGLU stays fused. Selected schedules vary shared
layout, tile dimensions, register microtiles and explicit unrolling.

The first discovery control omitted the production wide tile for outputs of
at least 5120 columns. A **corrected independent replay** restores that tile
and tests 39 control/finalist combinations. Its gains are **1.22–1.78×** across
the ten groups, and all selected schedules exceed the 3% replay threshold.
There are **816 held-out checks** covering every affected matrix at three
scales. The original discovery is archived with this caveat; retained gain
claims use the corrected replay and the full-model comparison.

## Native submission ablation

Two copies of the same searched bundle compare Python encoder/pass/finish/submit
wrappers with the new native submission helper. Both already use native dispatch
encoding, ordered compute/copy readback and owned host arrays. WGSL is identical;
all 19 before/after logit arrays are **bitwise equal**.

| Prefix | Python wrappers | Native fresh submission | Gain | llama.cpp chunk 32 |
|---:|---:|---:|---:|---:|
| 32 | 165.0 | **166.7** | 1.010× | 170.5 |
| 128 | 163.9 | **165.1** | 1.007× | 170.5 |
| 384 | 159.4 | **161.1** | 1.011× | 169.9 |

The helper resolves function pointers from the already-loaded wgpu 0.29 library,
validates the complete record stream before mutation, captures validation errors,
and releases every fresh command/pass/encoder handle. A validation error stops
encoding before invalid commands can be submitted. Resource/session checks and
map completion remain in the owning Python plan. Wheels without the new helper
retain their existing encoding/submission path.

## Accuracy and remaining work

All **19 independent NumPy fixtures** pass finite-output, RMS below 1%, cosine
above 0.9999 and matching argmax gates. Searched maximum relative RMS is **0.7414%**,
minimum cosine **0.99999064**, with **19/19 matching argmax**. Reset is bitwise
stable. All three Tensor profiles produce the same 96 generated token IDs with
both host and GPU greedy sampling; this is a bounded continuation check.

Native ordinary Vulkan arithmetic has maximum relative RMS **10.88%** against
the same NumPy fixtures, although all 19 argmax values match. Its precision
contract differs from Tensor's. These are throughput comparisons with explicit
numerical gates, rather than evidence of identical native/Tensor arithmetic.

Validation includes 70 GPU kernel cases covering packed fields, Q4/Q6 floating
dots, unrolling, fused FFN, half rounding and generic outer-product tails;
24 CPU contract/chunk-policy cases; and six runtime cases covering Python,
native encoding, native submission, malformed ABI, ordered owned snapshots and
validation recovery. Sources, helper binary hash, WGSL, fixture hashes, native
release/logs, timestamp traces and sample timings are archived in the raw report.

At prefix 128, completed decode needs approximately **6.09 ms/token** against
native **5.89 ms/token**. A separate instrumented trace records roughly 5.14 ms
whole GPU time. Per-kind median sums put paired FFN at 2.09 ms, linear at 1.48 ms,
FFN down/residual at 1.21 ms, RMS plus add/RMS at **0.386 ms across 60 dispatches**,
and attention at 0.116 ms. Timestamp instrumentation and medians are diagnostic;
their sums are not an accounting identity for uninstrumented completed latency.

The remaining decode target is about **0.20 ms/token** in this repeat.
Normalization/projection fusion is a concrete next experiment; it must pay for
duplicated normalization and shared-memory/occupancy costs. Prefill remains
primarily projection compute: rows-128 linear plus FFN account for roughly
230 ms of the 248 ms instrumented GPU pass. A larger search alone does not
establish that either remaining gap can be closed.

## Reproduce

Use the pinned local model and `build/llama-vulkan-b11310` from the preceding
report. Rebuild manifests after any implementation change. These commands run
from the repository root, with the optional C extension built in place:

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
$env:OPENBLAS_NUM_THREADS='6'
$env:TENSOR_BUILD_WEBGPU_NATIVE='1'
.venv/Scripts/python.exe setup.py build_ext --inplace
$model='D:/LLM/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf'
$root='build/lfm2-2.6b-parity'
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model $model --out "$root/baseline32" --provider webgpu --webgpu-profile subgroup --context 512 --prefill-chunk 32
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model $model --out "$root/chunked" --provider webgpu --webgpu-profile prefill_chunked --context 512 --prefill-chunks 32 128
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model $model --out "$root/adaptive" --provider webgpu --webgpu-profile quant_searched --context 512 --prefill-chunks 32 128
.venv/Scripts/python.exe benchmarks/lfm2/prefill_chase_compare.py --model $model --root $root --reference build/llama-vulkan-b11310 --fixtures build/lfm2-2.6b-q4_0-run-prefill --out "$root/comparison-final" --bundles baseline32 chunked adaptive --native-chunks 128 --allow-decode-changes --max-buffer-size 268435456 --repeats 7
.venv/Scripts/python.exe benchmarks/lfm2/runtime_compare.py --model $model --bundle "$root/adaptive" --reference build/llama-vulkan-b11310 --fixtures build/lfm2-2.6b-q4_0-run-prefill --out "$root/submission-ablation" --ablation submission --max-buffer-size 268435456 --repeats 7
```

Run discovery jobs sequentially on this GPU: `quantized_decode_search.py` for
the first decode search and again with `--encoding 14` for Q6; `prefill_chase_search.py`
with `--encoding 2 --rows 32 128`, then `--replay` pointing to its report;
and `decode_fusion_search.py --heads 32 --kv-heads 8 --depth 64 --capacity 640
--include-current`. The projection searches require `--model` and `--out`;
attention discovery requires only `--out`. Discovery source
snapshots and the exact candidates/results are retained in the raw report.
