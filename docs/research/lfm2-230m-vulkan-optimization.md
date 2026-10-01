# LFM2.5-230M Vulkan optimization

The current runner improves full-logit decode at a 128-token prefix from
89.7 to 141.3 tok/s for F16 and from 89.8 to 183.5 tok/s for Q4_0. Prefill
improves from 212.6 to 1,569.7 tok/s and from 197.2 to 1,322.3 tok/s,
respectively. Q4 now provides a decode speed advantage as well as lower
weight-buffer memory. Native llama.cpp remains faster.

This report uses the same Windows RX 6700 XT, driver 26.6.2, 230M checkpoints,
512-token context, F16 caches and pinned llama.cpp b11310 release as the
[initial demonstration](lfm2-230m-vulkan.md). It records the October 1, 2026
optimization pass, rather than replacing the initial measurements.

## Measurements

The before runner uses the original installed wheels in
`build/lfm2-230m-consumer` and copied bundles in `build/lfm2-230m-before`.
The after runner uses the updated source and matching rebuilt bundles.
Fresh [F16 before](data/lfm2-230m-f16-optimization-before.json),
[Q4 before](data/lfm2-230m-q4_0-optimization-before.json),
[F16 after](data/lfm2-230m-f16-optimized.json) and
[Q4 after](data/lfm2-230m-q4_0-optimized.json) retain every sample and fixture.

Both runners and native llama.cpp receive identical checkpoint files and token
IDs. The prefill chunk limit is 32; each sample has 64 forced single-token
decode calls after a 32-, 128- or 384-token prefix. Every forward call includes
device completion and full host FP32 last-token logits. Reset, load,
tokenization and sampling are excluded from these forward timings. Results
are medians of five repetitions after one warmup. Tensor/native order alternates;
all GPU benchmark processes run sequentially. Native uses six CPU threads,
all layers on Vulkan, flash attention and F16 K/V caches. The before and after
processes are sequential runs, not interleaved samples of both Tensor versions.

| Format | Prefix | Prefill before → after tok/s | Prefill speedup | Decode before → after tok/s | Decode speedup | Native decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| F16 | 32 | 211.3 → 1,430.0 | 6.77× | 89.9 → 144.3 | 1.60× | 491.8 |
| F16 | 128 | 212.6 → 1,569.7 | 7.38× | 89.7 → 141.3 | 1.57× | 490.9 |
| F16 | 384 | 178.4 → 1,515.8 | 8.50× | 91.6 → 131.4 | 1.43× | 489.1 |
| Q4_0 | 32 | 202.1 → 1,237.1 | 6.12× | 89.8 → 191.4 | 2.13× | 808.4 |
| Q4_0 | 128 | 197.2 → 1,322.3 | 6.70× | 89.8 → 183.5 | 2.04× | 805.5 |
| Q4_0 | 384 | 180.7 → 1,289.0 | 7.13× | 89.6 → 167.3 | 1.87× | 785.5 |

At prefix 128, the remaining decode gap is about 3.5× for F16 and 4.4× for Q4;
the prefill gap is about 2.7× and 4.6×. These are bounded public-forward
measurements, not default `llama-bench` or model-wide quality estimates.

## What changed

1. **Packed decode.** Q4 assigns 16 logical lanes per row and eight output rows
   per workgroup. Each lane extracts both nibbles from a byte and reuses its
   block scale. Q6 output projection traverses 256-value blocks, reuses its
   half scale and packed low/high planes across eight groups, and accumulates
   privately. Weights retain their GGUF encodings; there is no full-weight
   FP16 expansion.
2. **Register-tiled prefill.** A 16×32 output tile uses 128 threads and 32-wide
   K tiles. Each thread retains four FP32 accumulators across the whole
   reduction. Transposed shared RHS storage supports adjacent output-column
   loads. FP16 operand rounding remains explicit. This replaces the generic
   fragment materialization and per-K-tile accumulation through shared memory
   for these LFM2 projections.
3. **Attention and normalization.** Attention computes cache products only for
   occupied positions, then performs full-prefix FP32 softmax and value
   accumulation with workgroup tree reductions. Fixed loop bounds keep barriers
   uniform; masked positions are initialized and receive zero probability.
   Decode RMS uses a parallel reduction. Residual addition and the following
   FFN normalization share one decode kernel.
4. **Selective fusion.** Matching gate/up matrices share a projection kernel
   with a SwiGLU epilogue for prefill and quantized decode. F16 decode uses
   separate projections: its fused pair was slower on this adapter. Decode
   dispatch counts change from 183 to 169 for F16 and 141 for Q4; both prefill
   plans use 155. Greedy plans append one sampling dispatch.
5. **Queue and generation.** Ordered queue writes replace per-chunk host fences
   and reset fences on WebGPU. GPU argmax chooses the lowest token ID on a tie,
   writes the next decode token directly, and reads back one int32 instead of
   65,536 floats. `forward()` keeps its full-logit contract. The Python
   `generate(..., gpu_greedy=False)` option selects host greedy sampling.

The profile still uses `shader-f16` and ordinary portable WGSL. No subgroup,
cooperative matrix, vendor shader or CUDA runtime dependency was added.
Model-buffer totals remain 447.8 MiB F16 and 149.5 MiB Q4, excluding pipelines,
cached bind groups, uniforms and readback allocations.

## Profiling and rejected changes

The benchmark-only profiler requests native timestamp features on its own
device. It records whole-pass and individual-dispatch GPU timestamps, plus
host upload, encoding/submission and completion/readback wall time. Native
timestamps are ticks, requiring multiplication by the adapter period;
[wgpu documents this conversion](https://wgpu.rs/doc/wgpu/struct.Queue.html#method.get_timestamp_period).
`vulkaninfo` reports 10 ns/tick for this RX 6700 XT. Profiling flags are not
required by installed bundles. Per-dispatch timestamps perturb scheduling, and
host completion includes GPU execution, so these measurements are diagnostic
and must not be added together as independent costs.

| Whole compute-pass GPU time | F16 before → after | Q4 before → after |
|---|---:|---:|
| One-token decode, position 128 | 7.73 → 4.03 ms | 7.92 → 2.87 ms |
| 32-token prefill, position 96 | 200.34 → 20.21 ms | 191.65 → 28.49 ms |

The [F16 profiles](data/lfm2-230m-f16-profile-before.json) and
[final F16 profile](data/lfm2-230m-f16-profile-after.json), plus
[Q4 before](data/lfm2-230m-q4_0-profile-before.json) and
[Q4 after](data/lfm2-230m-q4_0-profile-after.json), retain per-dispatch records.
Decode originally spent substantial time on projections, attention and RMS;
prefill was dominated by projections. Projections remain the main GPU cost.

The [rejected F16 fusion profile](data/lfm2-230m-f16-profile-fused.json) showed
1.90 ms for paired decode FFN projections, compared with 1.15 ms for the
original separate gate/up projections. Keeping that fusion reduced the
128-prefix decode result to 129.1 tok/s; selecting separate F16 projections
raised it to 141.3 tok/s. For Q4, the paired projection profile improves from
0.76 ms for the original gate/up pair to 0.46 ms fused. These comparisons use
instrumented timings and separate runs; they establish the selection on this
adapter, rather than a general rule for all GPUs.

A parallel prefill RMS reduction was also rejected: it caused a retained
129-token logit fixture to exceed the unchanged 1% RMS gate. Small reduction
changes can move FP16 operand rounding boundaries through this hybrid model.
Prefill retains its original RMS summation; no tolerance was relaxed.

## Correctness and generation

Each final format passes 19 independent NumPy full-vocabulary logit checks
covering chat, cached continuations, 31/32/33 and 127/128/129 chunk boundaries,
511-plus-one context capacity, and reset. Every greedy argmax agrees with
NumPy and native llama.cpp on these fixtures. Worst NumPy-relative RMS is
0.224% F16 and 0.269% Q4, against the unchanged 1% gate; cosine stays above
0.99999. Reset is bitwise identical. This does not establish identical logits
or long-form generations with llama.cpp.

The [65-test suite](data/lfm2-230m-optimized-tests.xml) additionally covers
quantized projection block fields and subnormal scales, exact gathers, argmax
ties/tails/infinities, queued forward/reset behavior, GPU versus host greedy
generation, multi-chunk prompts, capacity and resource lifetime. CUDA GPU
tests were not rerun on this AMD machine.

The [fresh installed-wheel consumer](data/lfm2-230m-optimized-clean-consumer.json)
reproduces both saved logits and the complete 39-token answer to “What is 2 + 2?”
without Torch, TileLang, TVM, Triton, GGML or llama.cpp installed or imported.
Generation timing samples in the final reports include prompt tokenization,
reset, prefill and sampling but exclude loading. GPU-versus-host greedy
differences on this short fixture are variable; the GPU sampler is not credited
with a reliable standalone speedup. Most of the improvement comes from the
forward kernels. Separate before/after default-generation records use the
same prompt, token budget and completion protocol.

The complete default-generation call, including reset, tokenization, prefill,
sampling and final device completion, takes the following median time over five
repetitions after one warmup. Loading is excluded. Both versions produce the
same 39-token answer and stop at EOS.

| Format | Before → after | Speedup | Samples |
|---|---:|---:|---|
| F16 | 0.623 → 0.284 s | 2.19× | [before](data/lfm2-230m-f16-generate-before.json), [after](data/lfm2-230m-f16-generate-after.json) |
| Q4_0 | 0.617 → 0.271 s | 2.27× | [before](data/lfm2-230m-q4_0-generate-before.json), [after](data/lfm2-230m-q4_0-generate-after.json) |

## Reproduction and next work

Model download and producer commands remain in the
[package README](../../packages/tensor-llm/README.md). Rebuild both bundles after
changing model or kernel source; the manifests enforce implementation hashes.
The current consumer is `build/lfm2-230m-optimized-consumer`; the original
consumer and copied bundles are retained for comparisons.

```powershell
$env:OPENBLAS_NUM_THREADS='6'
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_run.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --bundle build/lfm2-230m-f16-webgpu --reference build/llama-vulkan-b11310 --out build/lfm2-230m-f16-webgpu-validation
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_run.py --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --reference build/llama-vulkan-b11310 --out build/lfm2-230m-q4_0-webgpu-validation
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_profile.py --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --out build/lfm2-230m-q4_0-profile-after.json --timestamp-period-ns 10 --repeats 7
$env:TENSOR_WEBGPU='1'
$env:TENSOR_LFM2_WEBGPU='1'
.venv/Scripts/python.exe -m pytest tests/providers/test_webgpu.py tests/providers/test_webgpu_audit.py packages/tensor-llm/tests/test_gguf.py packages/tensor-llm/tests/test_contracts.py packages/tensor-llm/tests/test_webgpu.py
build/lfm2-230m-optimized-consumer/Scripts/python.exe -I benchmarks/lfm2/webgpu_consumer.py --root (Get-Location).Path --out build/lfm2-230m-optimized-clean-consumer.json
```

The [verification record](data/lfm2-230m-optimized-verification.json) retains
bundle, wheel and source hashes. Checkpoints, binaries, environments and full
logits remain under ignored `build/`. Further work should tune the remaining
projection layouts and native readback/encoding overhead, then test an optional
subgroup path against this portable baseline. Context beyond 512 and other
quantizations or adapters require separate acceptance measurements.
