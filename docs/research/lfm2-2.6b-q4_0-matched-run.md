# LFM2.5-2.6B Q4_0: matched WebGPU run

This runs the [230M native submission](lfm2-230m-native-submission.md) measurement
protocol unchanged against the larger LFM2.5-2.6B QAD Q4_0 checkpoint, to find out
whether the 230M results carry to a model with 15x the non-embedding parameters.
No kernel, compiler or runtime change is credited here. Two harness
accommodations were required and are documented below; both are provably
numerics-preserving or opt-in.

The short answer: **decode scales well and the relative gap to native llama.cpp
narrows; prefill scales poorly and the relative gap widens.**

## Checkpoint and configuration

| | 230M Q4_0 | 2.6B QAD Q4_0 |
|---|---:|---:|
| File size | 149.1 MB | 1,593.9 MB |
| Non-embedding parameters | 0.163 B | 2.435 B |
| Embedding parameters | 0.067 B | 0.262 B |
| Layers (attention / conv) | 14 (6 / 8) | 30 (8 / 22) |
| Width / FFN | 1024 / 2560 | 2048 / 10752 |
| Heads / KV heads / head dim | 16 / 8 / 64 | 32 / 8 / 64 |
| Vocabulary | 65,536 | 128,000 |
| Encodings | Q4_0 82, Q6_K 1, F32 49 | Q4_0 166, Q6_K 1, F32 99 |

Both checkpoints quantize `token_embd.weight` to Q6_K and leave the 1-D
normalization weights in FP32, so the 2.6B run introduces no new quantization
path. The architecture contract is satisfied unchanged: `head_dim` is 64,
`ff % 32 == 0` and `conv == 3`. The declared context length is 128,000; the
measured profile still uses context 512.

Checkpoint: `D:\LLM\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-QAD-Q4_0.gguf`,
SHA-256 `a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03`.
This is Liquid AI's published QAD (quantization-aware distillation) Q4_0
checkpoint, not a locally produced quantization.

## Environment

Windows 11 10.0.26200, AMD Radeon RX 6700 XT 12 GiB (WDDM driver
32.0.21043.19003; the 230M report records this adapter as driver 26.6.2, which
this run does not independently map), Ryzen 5 5600, Python 3.12.13, NumPy
2.5.3, wgpu 0.29.0 / native 27.0.2. The pinned llama.cpp b11310 Vulkan release
(`f872b591…`) is the unchanged reference.

The bundle is 36 compiled kernels in the `subgroup` profile, capacity 576,
rows `(1, 32)`, target `webgpu-portable-v1`, and the device reports
`prepared_encoding: native`, so this is the same encoder configuration as the
230M "after" column. Device-owned buffers total 1.603 GB. Decode launches 285
dispatches per token; prefill launches 307 per 32-token chunk.

## Two harness accommodations

**Buffer limit opt-in.** `token_embd.weight` is Q6_K and 215,040,000 bytes,
which exceeds the WebGPU provider's conservative 128 MiB default. The adapter
actually reports 4 PiB `max-buffer-size` and 2 GiB `max-storage-buffer-binding-size`,
so this is a provider default, not a hardware limit. `webgpu_run.py` gained a
`--max-buffer-size` pass-through; the default stays `None`, so the 230M protocol
and its recorded limits are byte-for-byte unchanged. This run used
`--max-buffer-size 268435456`.

**Lazy FP16 operand cache.** The independent NumPy reference pre-rounded every
2-D weight to FP16 into a second full map. That is 876 MB of duplication at
230M and roughly 10.8 GB at 2.6B, which does not fit 32 GB of host RAM with the
checkpoint and both engines resident. The replacement materializes one rounded
weight at a time; each weight is touched at most once per 32-token chunk, so a
single-entry cache is sufficient. Verified bitwise identical to the eager map on
the 230M checkpoint across chat, single-token decode and an 80-token multi-chunk
prefill (`max|delta| = 0` in every case).

## Matched runner results

Protocol is identical to the 230M report: context 512, 32-token prefill chunks,
the same 64 forced decode tokens, medians over five repetitions after one warmup,
runner order alternating, host FP32 logits, loading and sampling excluded. Runs
execute sequentially with no concurrent compilation or GPU benchmarking.

| Format | Prefix | Tensor prefill, tok/s | Tensor decode, tok/s | llama.cpp prefill, tok/s | llama.cpp decode, tok/s |
|---|---:|---:|---:|---:|---:|
| Q4_0 | 32 | 194 | 124 | 941 | 171 |
| Q4_0 | 128 | 195 | 121 | 1,102 | 171 |
| Q4_0 | 384 | 193 | 113 | 1,144 | 171 |

Evidence: [matched run](data/lfm2-2.6b-q4_0-matched-run.json).

The 96-token generated answer takes **0.931 s GPU-greedy** and **0.944 s
host-greedy** (medians of five, including reset, tokenization, prefill, greedy
sampling and completion, excluding loading). GPU and host greedy output agree.
Unlike the 230M run, the model does not reach EOS within the harness's
`max_tokens=96` and the answer is truncated, so this is a throughput figure
rather than a time-to-answer figure.

## What scales and what does not

Implied weight-stream rate at prefix 128, using file bytes per token
(1.594 GB):

| | Tensor | llama.cpp | Tensor / llama.cpp |
|---|---:|---:|---:|
| 230M decode | 66 GB/s | 120 GB/s | 56% |
| 2.6B decode | 193 GB/s | 273 GB/s | 71% |

Decode is the interesting result. The 2.6B model has 10.7x the bytes of the
230M model but decodes only 3.7x slower, so the backend moves from being
overhead-bound at 230M to being substantially memory-bound at 2.6B, and the
relative gap to llama.cpp narrows from 1.80x to 1.42x. Nothing about the stack
degrades with size. Both absolute rates should be read as upper bounds on a
288 GB/s card — the llama.cpp figure is above 94% of theoretical peak, which
implies some L2 reuse, so the ratio is the meaningful signal rather than either
absolute number.

Prefill goes the other way. Using non-embedding parameters (2.435 B):

| | Tensor | llama.cpp | Tensor / llama.cpp |
|---|---:|---:|---:|
| 230M prefill | 534 GFLOP/s | 1.99 TFLOP/s | 27% |
| 2.6B prefill | 951 GFLOP/s | 5.37 TFLOP/s | 18% |

Tensor's prefill efficiency does improve in absolute terms (+78%), but llama.cpp
improves more, so the gap widens from 3.72x to 5.64x. Tensor prefill is also
suspiciously flat across prefix lengths (194 / 195 / 193 tok/s), the same
signature the 230M run showed: at these context lengths prefill is limited by
per-token projection cost and launch behaviour, not by attention. This is
consistent with the 230M report's own priority list, which names prefill
schedules as the remaining lever, and it is where a 2.6B checkpoint hurts most.

## Accuracy

All 19 independent full-vocabulary fixtures pass the unchanged gates: chat,
cached continuations, chunk boundaries, context capacity and bitwise reset.
Worst NumPy-relative RMS is **0.445%** (gate: below 1%), minimum cosine is
**0.99999747** (gate: above 0.9999), and all 19 argmax values agree. The
`reset_chat` fixture is bitwise identical to the opening `chat` fixture.

The llama.cpp comparison is reported and not gated, as in the 230M work, because
the two stacks use different activation quantization. Its divergence is larger
here — worst RMS 13.1%, minimum cosine 0.99725, against 10.4% / 0.99454 at
230M. That is a statement about accumulated quantization error over 15x more
parameters, not about a failure: the independent FP32-operand NumPy reference
is the gate, and it passes comfortably.

## Reproduction

```powershell
$env:TENSOR_WEBGPU='1'
$env:TENSOR_LFM2_WEBGPU='1'
.venv/Scripts/python.exe -m pytest tests/providers/test_webgpu.py tests/providers/test_webgpu_audit.py tests/providers/test_webgpu_lowering.py packages/tensor-llm/tests/test_gguf.py packages/tensor-llm/tests/test_contracts.py packages/tensor-llm/tests/test_webgpu.py
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model 'D:\LLM\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-QAD-Q4_0.gguf' --out build/lfm2-2.6b-q4_0-webgpu --context 512 --provider webgpu --webgpu-profile subgroup
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_run.py --model 'D:\LLM\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-QAD-Q4_0.gguf' --bundle build/lfm2-2.6b-q4_0-webgpu --reference build/llama-vulkan-b11310 --out build/lfm2-2.6b-q4_0-run --max-buffer-size 268435456
```

The 116-test suite passes unchanged. Full end-to-end wall time was 844 s, of
which the NumPy reference dominates at roughly 10.8 GB resident.

## Limits of this run

One checkpoint in one quantization on one consumer AMD GPU, at context 512. No
F16 2.6B build was available, so the F16-versus-Q4_0 contrast that carries the
230M report is not reproduced here and no claim is made about it. CUDA tests
remain unrun on this host. Schedule tuning, per-dispatch profiling and
before/after attribution were not repeated; the 230M artifacts remain the
authority on which individual optimizations paid off. The prefill gap and the
decode gap move in opposite directions with model size, so conclusions drawn
from a single model size do not generalize — that is the main reason to record
this run.
