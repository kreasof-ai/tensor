# LFM2.5-2.6B QAD Q4_0: runtime and larger prefill chunks

[Research index](README.md) · [Previous quantized decode result](lfm2-webgpu-decode-push.md) ·
[230M F16 prefill transfer](lfm2-prefill-chase.md)

On 2026-10-03, a fresh RX 6700 XT comparison reaches **146 tokens/s decode**
at prefix 128 versus llama.cpp Vulkan's **169–170**. This is about 86% of
native throughput, a **1.16×** gap. Larger prefill chunks improve Tensor's
128/384-token prefill by **49%/48%**, to **346/339 tokens/s**. Native also
benefits from larger chunks and still leads substantially on prefill.

The existing local checkpoint is **QAD Q4_0**, the same file as the previous
2.6B measurements, rather than the separately published ordinary Q4_0 file:
`D:/LLM/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf`.
SHA256 is `a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03`.
Its 166 Q4_0 matrices, one Q6_K tied embedding/output and 99 F32 tensors remain
packed. The new outer-product selection is F16-only and does not apply to
these weights; this repeat introduces no new projection schedules.

## Completed prefill and decoding

All rates are tokens/s. Every forward returns completed host FP32 logits.
The same forced token IDs run on each backend, at context 512. Three warmups
precede seven measured samples. Runner order rotates sequentially on the GPU;
reset, model loading, compilation and sampling are excluded. Forced decode is
64 tokens after each prefix. llama.cpp b11310 runs all layers on Vulkan, with
Flash Attention, F16 K/V and six CPU threads.

Tensor's `baseline32` bundle uses the current subgroup profile with rows
1/32. The adaptive `prefill_chunked` bundle uses rows 1/32/128 and the same
projection schedules. Both have 254 decode dispatches and 307 per prefill
chunk. The adaptive bundle allocates 1,633,086,092 bytes versus 1,603,438,732
for the baseline. The raised WebGPU buffer limit is 256 MiB for the 215 MB
Q6_K output weight. Decode projection WGSL is byte-identical; cache capacity
and attention shaders differ between bundles.

| Prompt tokens | Tensor prefill, chunk 32 | Tensor adaptive prefill | Gain | llama.cpp chunk 32 | Best tested llama.cpp prefill |
|---:|---:|---:|---:|---:|---:|
| 32 | 230.3 | **230.4** | 1.00× | 927.3 | **936.9** (chunk 64) |
| 128 | 231.9 | **345.8** | **1.49×** | 1,094.3 | **1,944.5** (chunk 128) |
| 384 | 229.8 | **339.0** | **1.48×** | 1,137.6 | **2,185.6** (chunk 128) |

The remaining prefill gap to the best tested native chunk is approximately
**4.07× / 5.62× / 6.45×**. These are bounded chunk-size comparisons, not a
global optimization of either backend. Tensor now needs **370.20 ms** for
128 tokens, versus native's **65.83 ms**. At 384 tokens the corresponding
latencies are **1,132.71 ms / 175.69 ms**. The F16 230M result does not establish
quantized 2.6B prefill parity.

| Prefix tokens | Tensor decode, chunk-32 bundle | Tensor decode, adaptive bundle | llama.cpp decode, chunk-32 reference | Native / adaptive |
|---:|---:|---:|---:|---:|
| 32 | 149.5 | **150.2** | 169.7 | **1.13×** |
| 128 | 144.7 | **145.7** | 169.7 | **1.16×** |
| 384 | 133.7 | **134.1** | 169.1 | **1.26×** |

Native decode across its three chunk configurations is 168.7–169.7 tokens/s.
The small adaptive/baseline decode differences do not demonstrate a kernel
improvement. Decode at prefix 128 now takes **6.86 ms/token**, versus about
**5.89 ms/token** native. Parity requires approximately **14.1% less completed
Tensor latency** at this prefix.

The historical result was 139.9 tokens/s versus 169.5, with one warmup and
five samples. The new adaptive result is 4.2% above that historical Tensor
rate, but the separate-session comparison is not a controlled runtime
ablation. The next experiment isolates that cause with identical shaders.

## Isolating the runtime benefit

`runtime_compare.py` loads two copies of the same current chunk-32 bundle.
The before runner restores the previous queue-wide fence, separate readback
copy submission and public mapping flush. The after runner uses the retained
combined compute/copy submission and ordered map completion. All **19 fixtures
are bitwise equal** between these two runners.

| Prefix | Original readback decode | Current readback decode | Gain | llama.cpp decode |
|---:|---:|---:|---:|---:|
| 32 | 143.1 | **150.0** | **4.9%** | 169.5 |
| 128 | 138.7 | **145.7** | **5.0%** | 169.9 |
| 384 | 128.3 | **133.9** | **4.4%** | 169.2 |

At prefix 128, completed latency falls from **7.21 to 6.86 ms/token**.
Prefill changes by less than 0.5%. Median GPU-greedy generation of the same
96-token bounded continuation falls from **804.86 to 779.09 ms**, a **3.2%**
latency reduction. These timings include tokenization, reset, prefill,
sampling and completion. The response reaches the 96-token measurement cap;
it is not an EOS-completed-response measurement.

## Accuracy

Both experiments reuse the independently computed NumPy logits for this
exact checkpoint and ordered fixture sequence. The cached-reference loader
verifies the original passed report, model SHA256, context 512, token/reset
sequence, finite arrays and original numerical gates. The evidence retains
all reference and result array hashes. Native logits are recomputed.

Both Tensor chunk variants pass **19/19 fixtures**, with maximum relative RMS
**0.4446%**, minimum cosine **0.99999747**, and matching argmax in all cases.
The chunk variants are also **bitwise equal on all 19 fixtures**. Tests cover
partial chunks, 31/32/33 and 127/128/129 lengths, cached decoding through
position 511 and deterministic reset. Both GPU and host greedy paths produce
the same 96 token IDs across both bundles and the runtime ablation. The cached
reference helper's seven CPU regression tests also pass; modified benchmark
scripts compile and `git diff --check` passes.

Ordinary native Vulkan uses different arithmetic precision. Its maximum
NumPy-relative RMS is **14.2719% / 13.9521% / 10.8808%** for chunk sizes
32/64/128, respectively; all native fixture argmaxes match. Native does not
meet Tensor's 1% relative-RMS gate here. This is a same-checkpoint throughput
comparison, with the precision difference recorded rather than treated as
precision-equivalent parity or a model-quality evaluation.

## Remaining costs

A five-repeat diagnostic profile uses Vulkan's measured 10 ns timestamp
period. Decode is profiled at prefix 128; prefill at active prefix 256.
Per-dispatch instrumentation can perturb execution and these timings are
diagnostic, rather than the completed-forward acceptance measurements.

| Rows | Whole GPU pass | Instrumented dispatch sum | Projection fraction of dispatch sum | Fused FFN + down fraction |
|---:|---:|---:|---:|---:|
| 1 | 6.10 ms | 6.15 ms | **85.4%** | **55.5%** |
| 32 | 139.65 ms | 139.83 ms | **93.4%** | **70.3%** |
| 128 | 378.48 ms | 419.78 ms | **95.6%** | **73.5%** |

Row-128 instrumentation has a substantial sum/whole-pass difference, so those
fractions are approximate. The direction is clear: packed projections still
dominate. Larger chunks help but retain the existing register-matmul family.
The next prefill experiment would extend the staged outer-product family to
packed Q4 loads, decoding into F16 shared operands with FP32 accumulation,
and independently search the larger 2048/10752 FFN shapes. That extension and
any integer-dot activation-quantization route require new kernel and
full-model numerical validation; their gains are not established by this run.

## Evidence and reproduction

[Raw comparison, runtime ablation, shader/source snapshots, profiles and array hashes](data/lfm2-2.6b-q4_0-revisit.json).
Windows, Ryzen 5600, RX 6700 XT, AMD Vulkan 26.6.2, wgpu 0.29.0,
with the native prepared-plan encoder. Repository head before this repeat:
`3be6cf9`. Only benchmark helpers are extended for the larger buffer limit;
the inference/compiler implementation is unchanged.

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
$env:OPENBLAS_NUM_THREADS='6'
$model='D:/LLM/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf'
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model $model --out build/lfm2-2.6b-revisit/baseline32 --context 512 --provider webgpu --webgpu-profile subgroup
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model $model --out build/lfm2-2.6b-revisit/adaptive --context 512 --provider webgpu --webgpu-profile prefill_chunked --prefill-chunks 32 128
.venv/Scripts/python.exe benchmarks/lfm2/prefill_chase_compare.py --model $model --root build/lfm2-2.6b-revisit --reference build/llama-vulkan-b11310 --fixtures build/lfm2-2.6b-q4_0-run-prefill --out build/lfm2-2.6b-revisit/comparison --bundles baseline32 adaptive --repeats 7 --max-buffer-size 268435456
.venv/Scripts/python.exe benchmarks/lfm2/runtime_compare.py --model $model --bundle build/lfm2-2.6b-revisit/baseline32 --reference build/llama-vulkan-b11310 --fixtures build/lfm2-2.6b-q4_0-run-prefill --out build/lfm2-2.6b-revisit/runtime-ablation --repeats 7 --max-buffer-size 268435456
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_profile.py --model $model --bundle build/lfm2-2.6b-revisit/adaptive --out build/lfm2-2.6b-revisit/profile-adaptive.json --repeats 5 --timestamp-period-ns 10 --max-buffer-size 268435456 --prefill-position 256
.venv/Scripts/python.exe benchmarks/lfm2/quantized_revisit_summary.py --root build/lfm2-2.6b-revisit --reference build/llama-vulkan-b11310 --out docs/research/data/lfm2-2.6b-q4_0-revisit.json
```

The cached fixtures are generated by the original `webgpu_run.py` benchmark;
omit its `--numpy-fixtures` option to create a fresh independent reference on
this exact file. Intermediate prompt lengths and longer contexts are outside
the measured throughput scope.
