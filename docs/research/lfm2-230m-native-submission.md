# LFM2.5-230M: native submission and vector prefill

This third pass improves completed decode at prefix 128 by **13.4% F16** and
**18.8% Q4_0** over the preceding [compiler optimization](lfm2-230m-webgpu-compiler-optimization.md).
Prefill improves by 11.2% and 5.2%. It adds an optional native prepared-plan
encoder and compiler-owned four-wide prefill dot scheduling. Both formats pass
the unchanged full-logit gates, and both native and ordinary Python wheels run
without producer compilers installed.

## Matched runner results

Environment remains Windows, RX 6700 XT 12 GiB, Ryzen 5 5600, driver 26.6.2,
Python 3.12.13, NumPy 2.5.3, TileLang 0.1.14 and wgpu 0.29.0/native 27.0.2.
The optional C extension was built with the installed MSVC 14.51 toolchain.
Checkpoints and the pinned llama.cpp b11310 reference are unchanged.

Before uses the previous installed Python wheel and copied subgroup bundles.
After uses native encoding and selected vector prefill schedules. Runs execute
sequentially without concurrent compilation/GPU benchmarks. Both return completed
host FP32 logits; loading and sampling are excluded. Prefixes use 32-token chunks,
context 512, and the same 64 forced decode tokens. Tensor/native order alternates;
medians use five repetitions after one warmup. Native llama.cpp uses Vulkan, all
layers offloaded, F16 caches, flash attention and six CPU threads.

| Format | Prefix | Prefill before → after, tok/s | Decode before → after, tok/s | Native prefill / decode, tok/s |
|---|---:|---:|---:|---:|
| F16 | 32 | 1,519 → 1,713 | 239 → 275 | 3,144 / 483 |
| F16 | 128 | 1,590 → 1,768 | 232 → 263 | 4,225 / 483 |
| F16 | 384 | 1,573 → 1,737 | 206 → 235 | 4,533 / 478 |
| Q4_0 | 32 | 1,502 → 1,605 | 399 → 476 | 3,700 / 805 |
| Q4_0 | 128 | 1,558 → 1,639 | 375 → 445 | 6,103 / 802 |
| Q4_0 | 384 | 1,538 → 1,623 | 319 → 368 | 6,991 / 780 |

Evidence: [F16 before](data/lfm2-230m-native-f16-before.json),
[F16 after](data/lfm2-230m-native-f16-after.json),
[Q4 before](data/lfm2-230m-native-q4_0-before.json),
[Q4 after](data/lfm2-230m-native-q4_0-after.json).
At prefix 128, native decode is still 1.84× faster F16 and 1.80× Q4_0;
prefill is 2.39× and 3.72× faster. Native throughput varies slightly between runs;
the reported improvement compares fresh Tensor before/after samples.

The unchanged 39-token answer takes **0.176 → 0.155 s F16** and
**0.125 → 0.096 s Q4_0**. These medians include reset, tokenization, prefill,
greedy sampling and completion, excluding loading. GPU/host greedy output agrees.

## Native encoder and attribution

The optional `tensor.providers._webgpu_native` extension encodes owned prepared
nodes in one C call. Each record holds pipeline/bind-group handles and packed
dispatch dimensions. Function pointers come from the already-loaded pinned wgpu
library, with no second loader or linked wgpu dependency. The plan retains all
objects, checks each distinct resource before entering C, and clears records on
close. The extension checks record structure before issuing commands. It captures
native validation errors around the whole call through wgpu's error handler;
invalid-dispatch tests confirm errors are raised and subsequent valid launches work.
Consecutive identical pipeline bindings remain skipped.

This encodes fresh WebGPU commands each launch. It does not reuse consumed command
buffers or implement captured graph replay. A Python fallback remains available
when the extension is absent. Building the extension is opt-in through
`TENSOR_BUILD_WEBGPU_NATIVE=1`; ordinary builds remain Python wheels. The measured
native wheel is CPython 3.12 Windows x64. Upgrading wgpu or transferring the
extension to another platform requires separate validation.

Fresh seven-sample timestamp/host profiles show the following results:

| Format | Decode encode/submit, before → after | Prefill encode/submit, before → after | Whole prefill GPU, before → after |
|---|---:|---:|---:|
| F16 | 0.999 → 0.494 ms | 0.820 → 0.407 ms | 19.727 → 17.887 ms |
| Q4_0 | 0.903 → 0.399 ms | 0.919 → 0.401 ms | 22.161 → 20.750 ms |

Evidence: [F16 before profile](data/lfm2-230m-native-f16-profile-before.json),
[F16 after profile](data/lfm2-230m-native-f16-profile-after.json),
[Q4 before profile](data/lfm2-230m-native-q4_0-profile-before.json),
[Q4 after profile](data/lfm2-230m-native-q4_0-profile-after.json).
Timestamp ticks use the verified Vulkan period of 10 ns. Per-dispatch instrumentation
perturbs scheduling; completed-call measurements above remain the acceptance result.
All decode WGSL is byte-identical to before. Instrumented decode sums are almost
unchanged: F16 3.083 → 3.080 ms and Q4 1.764 → 1.768 ms. Whole-pass Q4 timestamps
vary between runs, so no GPU decode speedup is attributed to this change.

An additional [F16 encoder ablation](data/lfm2-230m-native-f16-encode-ablation.json)
and [Q4 encoder ablation](data/lfm2-230m-native-q4_0-encode-ablation.json)
alternate Python/native encoding on identical new shaders, buffers and forced
tokens, checking bitwise equality of final logits. Native decode runs at 264
versus 232 tok/s F16 and 441 versus 347 tok/s Q4 in that isolated protocol.
The ablation omits interleaved llama.cpp work, so its absolute throughput should
not be substituted into the matched table.

## Compiler scheduling

`register_matmul_schedule` now supports scalar or four-wide FP32 dot operations,
LHS shared padding/transposition and explicit inner-loop unrolling. FP16 operand
rounding is retained; vector dots group accumulation into four-element sums,
changing summation order and requiring full-model validation.

The [F16 schedule sweep](data/lfm2-230m-native-f16-schedule-tune.json) and
[Q4 schedule sweep](data/lfm2-230m-native-q4_0-schedule-tune.json) compare eight
configurations on actual FFN-down/gate weights, checking independent FP16-operand
matrix references. Twenty launches plus completion per sample, seven samples
after warmup, use identical inputs across variants. Four-wide dots reduce
FFN-down median from 0.553 to 0.487 ms F16 and 0.568 to 0.488 ms Q4. Maximum
absolute errors remain below 1e-7 on these scaled inputs.

Padding/transposition offer little consistent improvement. Explicit unrolling
adds no reliable gain. The selected schedule retains the 16×32 output tile, K=64,
default shared layout and serial inner loop. F16 prefill projections use four-wide
dots; Q4 uses them for large-K FFN-down while retaining scalar dots for smaller-K
matrices. Other quantizations are not credited with measured gains.

## Acceptance and reproduction

Each selected format passes 19 independent full-vocabulary fixtures: chat, cached
continuations, chunk boundaries, context capacity and bitwise reset. Worst
NumPy-relative RMS is **0.190% F16 / 0.499% Q4_0**; minimum cosine is
0.99999821 / 0.99998770. Every fixture argmax agrees. The gates remain RMS below
1%, cosine above 0.9999 and identical argmax. Long-form/native bitwise equality
is not established. The prior rejected activation integer-dot experiment remains
excluded; this pass adds no activation quantization.

The [90-test suite](data/lfm2-230m-native-tests.xml) additionally covers native
validation error capture, malformed records, Python/native resource lifetimes,
shared layouts/vector dots with M/N tails and unrolling, plus all previous packed
weight, compiler, generation and runtime checks. CUDA GPU tests remain unrun on
this AMD host. The native extension is not required to produce WGSL.

Fresh [native-wheel](data/lfm2-230m-native-clean-consumer.json) and
[pure-Python fallback](data/lfm2-230m-native-fallback-clean-consumer.json) environments
both reproduce saved logits and the complete generated answer, with producer/
reference packages absent and imports blocked. Both install eight packages; the
native encoder arrives inside the Tensor wheel. The portable no-subgroup profile
also passes the retained 19-fixture replay for
[F16](data/lfm2-230m-native-f16-portable-validation.json) and
[Q4_0](data/lfm2-230m-native-q4_0-portable-validation.json).

Build matching bundles/wheels with the [package commands](../../packages/tensor-llm/README.md).
The producer needs the installed MSVC build tools only when opting into the native
encoder. The consumer needs neither MSVC nor TileLang/TVM. For current validation:

## Superseded by later schedule corrections

The measurements above were taken before two kernel changes that improve this
checkpoint too. They are not folded into the tables here, because those tables
document one specific optimization pass and the evidence JSON
(`data/lfm2-230m-native-{f16,q4_0}-{before,after}.json`) records the code state
that pass actually measured.

The later changes are a decode reduction that widens the single-workgroup
`r=1` RMS launch from 64 to 256 threads, a prefill fix that removes two emulated
unsigned integer divides from the FP16 staging loop, and a prefill output tile
of `(32,32,64)` selected for projections with 5120 or more columns. All are
bit-identical or reassociate only the sum-of-squares, and all 19 fixtures keep
passing the unchanged gates. The 230M checkpoint has no projection above 3,072
columns, so the tile change does not apply to it by construction.

Fresh matched runs on the same machine and protocol, context 512, prefix 128:

| Format | Prefill, tok/s | Decode, tok/s |
|---|---:|---:|
| F16 | 1,768 → 1,777 | 263 → **274** |
| Q4_0 | 1,639 → **1,729** | 445 → **473** |

Evidence: [post-schedule F16](data/lfm2-230m-post-schedule-f16.json),
[post-schedule Q4_0](data/lfm2-230m-post-schedule-q4_0.json). The derivation,
including a rejected weight-layout experiment, is in the
[2.6B matched run](lfm2-2.6b-q4_0-matched-run.md), where the same changes are
worth +19.0% prefill and +8.1% decode.

Rebuilding the bundles is required after a kernel change: the implementation
hash guard rejects them, and `test_real_model_reset_reference_and_capacity`
compares against a stored bitwise baseline that must be regenerated.

## Reproduction

```powershell
$env:TENSOR_WEBGPU='1'
$env:TENSOR_LFM2_WEBGPU='1'
.venv/Scripts/python.exe -m pytest tests/providers/test_webgpu.py tests/providers/test_webgpu_audit.py tests/providers/test_webgpu_lowering.py packages/tensor-llm/tests/test_gguf.py packages/tensor-llm/tests/test_contracts.py packages/tensor-llm/tests/test_webgpu.py
build/lfm2-230m-native-consumer/Scripts/python.exe -I benchmarks/lfm2/webgpu_consumer.py --root (Get-Location).Path --out build/lfm2-230m-native-clean-consumer.json
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_schedule_tune.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --out build/lfm2-230m-f16-schedule-tune
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_encode_bench.py --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --out build/lfm2-230m-q4_0-native-encode-ablation.json
```

The [verification record](data/lfm2-230m-native-verification.json) retains sources,
native/pure wheels, bundles and evidence hashes. Frozen previous wheels/bundles
and full logits remain under ignored `build/`. Remaining priorities are decode
projection layouts/occupancy, more selective epilogue fusion, and prefill schedules
that can narrow the still-large native GEMM gap. Native submission now costs
roughly 0.4–0.5 ms per decode; further improvements need measured GPU work savings
as well as lower completion overhead.
