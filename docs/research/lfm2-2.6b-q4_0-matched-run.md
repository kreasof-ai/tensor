# LFM2.5-2.6B Q4_0: matched WebGPU run and schedule corrections

This runs the [230M native submission](lfm2-230m-native-submission.md) measurement
protocol unchanged against the larger LFM2.5-2.6B QAD Q4_0 checkpoint, to find out
whether the 230M results carry to a model with 15x the non-embedding parameters.
It then records two schedule corrections found by profiling that model, which
improve decode and prefill without changing any precision gate.

Two harness accommodations were required and are documented below. Both are
provably numerics-preserving or opt-in.

**Headline:** the 2.6B decode gap to native llama.cpp narrows from 1.42x to
1.31x and the prefill gap from 5.64x to 4.75x, with all 19 correctness fixtures
passing unchanged. The corrections are worth **+8.1% decode and +19.0% prefill**
at prefix 128, and they are bit-identical to the previous kernels.

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
| Q4_0 | 32 | 231 | 134 | 939 | 171 |
| Q4_0 | 128 | 233 | 131 | 1,104 | 171 |
| Q4_0 | 384 | 227 | 122 | 1,145 | 171 |

Evidence: [matched run](data/lfm2-2.6b-q4_0-matched-run.json).

The 96-token generated answer takes **0.851 s GPU-greedy** and **0.855 s
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
relative gap to llama.cpp narrows. Nothing about the stack degrades with size.
Both absolute rates should be read as upper bounds on a 288 GB/s card — the
llama.cpp figure is above 94% of theoretical peak, which implies some L2 reuse,
so the ratio is the meaningful signal rather than either absolute number.

Prefill goes the other way. Using non-embedding parameters (2.435 B):

| | Tensor | llama.cpp | Tensor / llama.cpp |
|---|---:|---:|---:|
| 230M prefill | 562 GFLOP/s | 1.99 TFLOP/s | 28% |
| 2.6B prefill | 1,132 GFLOP/s | 5.37 TFLOP/s | 21% |

## Decode correction: single-workgroup reduction width

The first decode profile of the 2.6B plan ([before](data/lfm2-2.6b-q4_0-profile-before.json))
showed 7.836 ms of GPU time per token, of which `rms` and `add_rms` accounted
for 1.135 ms — 14.5% of decode — while moving about 750 KB. The
`kind=='rms' and r==1` branch launches `T.Kernel(r, threads=64)`, so with `r=1`
that is **one workgroup of 64 threads** reducing a 2,048-element row on a
40-CU device, walking 32 dependent iterations twice. `add_rms` reuses the same
source, so it inherited the geometry.

`rms_width(c)` selects the largest of 256/128/64 that divides `c`, giving
`c=2048` a full 256-thread wave and 8 iterations instead of 32.
`add_rms` derives its index from the same helper; it previously hardcoded
`i * 64`, which would have silently dropped its residual store.

| | before | after |
|---|---:|---:|
| `rms` (31 dispatches) | 0.572 ms | 0.206 ms |
| `add_rms` (30 dispatches) | 0.563 ms | 0.204 ms |
| whole decode | 7.836 ms | 6.520 ms |

[Profile after](data/lfm2-2.6b-q4_0-profile-after.json). The measured matched
gain is 8.1% at prefix 128, against 9.3% predicted from the profile, with
llama.cpp unchanged at 171 tok/s across both runs.

## Prefill correction: staging-loop divisions and output tile

A 12-configuration tile sweep on the real 2.6B FFN weights
([sweep](data/lfm2-2.6b-q4_0-tile-sweep.json)) found the CLBlast direction does
not transfer. Every larger tile is slower — `(32,128,64)` is 2.3x worse than
`(16,32,64)` on `ffn_down` — and only `tile_m=32` helps, on `ffn_gate`
(1.14x). Two changes survived:

**Unsigned divisions in the FP16 rounding.** The generated WGSL emitted
`exponent / 113u < 1u` and `exponent / 102u < 1u` for what is semantically
`exponent < 113`. Every staged prefill operand runs through `round_half` twice,
so the staging loop carried two emulated unsigned divides per element. Testing
the exponent against a signed cast removes both: the rebuilt WGSL contains zero
unsigned divides. This is bit-identical — all 24 swept configurations return the
same maximum absolute error as before (1.64e-07 at `dot_width=4`, 2.87e-07 at
`dot_width=1`).

**Shape-dependent output tile.** `projection_tile` selects `(32,32,64)` when
`o >= 5120`. `tile_m=32` halves the workgroup count, which only pays while
enough column tiles remain to fill the device: `o=10752` gives 336 workgroups
(8.4 per CU) and gains 12%, while `o=2048` gives 64 (1.6 per CU) and regresses.
The guard is therefore on column count, not depth. The 230M checkpoint has no
projection above 3,072 columns and is unaffected by construction.

### Rejected: transposed weight layout

The inner loop builds each 4-wide weight operand from `rhs[base]`,
`rhs[base+32]`, `rhs[base+64]`, `rhs[base+96]` — stride 64 bytes, four scalar
LDS reads that cannot become one vector load, while the activation operands are
contiguous. Storing the shared weights as `[n][k]` makes them adjacent. It
measured **1.4–1.9x slower and failed the `ffn_down` correctness gate**.
Threads index the weight column as `tx % nr`, so under `[n][k]` all 16 distinct
columns sit 64 elements apart and collide on a single LDS bank, while `[k][n]`
keeps consecutive columns adjacent. The existing layout is correct for this
access pattern; making the vector load work needs an XOR swizzle, not a layout
swap. Reverted; the reasoning is recorded in `webgpu_schedule_tune.py`.

## Corrected results

Matched runner, same protocol, before → after at each prefix:

| Prefix | Prefill, tok/s | Decode, tok/s | vs llama.cpp (prefill / decode) |
|---:|---:|---:|---|
| 32 | 194 → **231** | 124 → **134** | 4.07x / 1.28x |
| 128 | 195 → **233** | 121 → **131** | 4.75x / 1.31x |
| 384 | 193 → **227** | 114 → **122** | 5.05x / 1.41x |

At prefix 128 that is **+19.0% prefill and +8.1% decode**, narrowing the prefill
gap from 5.64x to 4.75x and the decode gap from 1.42x to 1.31x. Generation falls
from 0.931 s to 0.851 s. On the 230M checkpoint the same change gives +5.5%
prefill and +6.3% decode for Q4_0, and +4.2% decode with flat prefill for F16;
see [post-schedule Q4_0](data/lfm2-230m-post-schedule-q4_0.json) and
[post-schedule F16](data/lfm2-230m-post-schedule-f16.json).

## Accuracy

All 19 independent full-vocabulary fixtures pass the unchanged gates: chat,
cached continuations, chunk boundaries, context capacity and bitwise reset.
Worst NumPy-relative RMS is **0.445%** (gate: below 1%), minimum cosine is
**0.99999747** (gate: above 0.9999), and all 19 argmax values agree. The
`reset_chat` fixture is bitwise identical to the opening `chat` fixture. The
`round_half` change is bit-identical by construction, and the reduction-width
change reassociates only the sum-of-squares, which moves the decode-path
fixtures in the last few bits while leaving every gate unchanged.

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
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_run.py --model 'D:\LLM\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-QAD-Q4_0.gguf' --bundle build/lfm2-2.6b-q4_0-webgpu --reference build/llama-vulkan-b11310 --out build/lfm2-2.6b-q4_0-run-prefill --max-buffer-size 268435456
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_profile.py --model 'D:\LLM\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-QAD-Q4_0.gguf' --bundle build/lfm2-2.6b-q4_0-webgpu --out build/lfm2-2.6b-q4_0-profile-after.json --repeats 7 --timestamp-period-ns 10 --max-buffer-size 268435456
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_schedule_tune.py --model 'D:\LLM\LiquidAI\LFM2.5-2.6B-GGUF\LFM2.5-2.6B-QAD-Q4_0.gguf' --out build/lfm2-2.6b-q4_0-tile-sweep --preset tiles
```

The 116-test suite passes. A cold checkout has no `build/`, so the two 230M
bundles must be rebuilt with `producer.py` before the suite is green: the
kernel change invalidates their implementation hashes by design, and
`test_real_model_reset_reference_and_capacity` also compares against a stored
bitwise baseline that must be regenerated by re-running `webgpu_run.py` for
230M F16 and Q4_0. End-to-end wall time was 1,109 s, of which the NumPy
reference dominates at roughly 10.8 GB resident.

## Limits of this run

One checkpoint in one quantization on one consumer AMD GPU, at context 512. No
F16 2.6B build was available, so the F16-versus-Q4_0 contrast that carries the
230M report is not reproduced here and no claim is made about it. CUDA tests
remain unrun on this host.

Prefill is still 4.75x behind llama.cpp and both schedule levers are now spent:
tiles are exhausted and the staging divisions are gone. The remaining gap is
structural in `register_matmul_schedule`, which issues one scalar LDS read per
element with no vectorisation, and whose bank-conflict-free weight layout is
exactly what blocks the 4-wide operand from becoming a real `ds_read_b128`.
Closing it needs a swizzled shared layout with vector-typed arrays. Decode's
remaining headroom is small: after the reduction-width fix, non-GEMV dispatches
sit at about 2.1 us each, which is the per-dispatch floor, so further decode
gains require cutting dispatch count by fusing neighbours rather than tuning
kernels. The prefill and decode gaps move in opposite directions with model size,
so conclusions drawn from a single model size do not generalize.
