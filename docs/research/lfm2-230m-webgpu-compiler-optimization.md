# LFM2.5-230M: WebGPU compiler and Vulkan optimization

On the RX 6700 XT, this second optimization pass improves completed decode at a
128-token prefix by **1.64× F16** and **2.04× Q4_0**, compared with the previous
[optimized runner](lfm2-230m-vulkan-optimization.md). Both formats pass the same
19 independent full-vocabulary logit gates. These changes are in the compiler,
provider and installed LFM2 runner, with an explicit optional subgroup profile.

## Environment and comparison

Measured October 1, 2026 on Windows, Ryzen 5 5600, 32 GiB RAM, RX 6700 XT 12 GiB,
AMD driver 26.6.2, Python 3.12.13, NumPy 2.5.3, TileLang 0.1.14 and wgpu 0.29.0
(wgpu-native 27.0.2). Model revisions/checksums and the pinned llama.cpp b11310
reference are unchanged from the [original report](lfm2-230m-vulkan.md).

The frozen before runner uses its installed wheel environment and copied bundles.
Before/after runs execute sequentially without other benchmark GPU work. Each
format checks 19 fixtures, then times one warmup and five repetitions at each
prefix length. Tensor/native order alternates. Both return completed host FP32
last-token logits; loading and sampling are excluded. Context is 512, prefill
chunks are 32, and each prefix is followed by the same 64 forced decode tokens.
Native llama.cpp uses Vulkan, all layers offloaded, F16 caches, flash attention
and six CPU threads. The retained samples show native throughput varying slightly
between runs; all speedups below compare the fresh Tensor before/after medians.

| Format | Prefix | Prefill before → after, tok/s | Decode before → after, tok/s | Decode speedup | Native prefill / decode, tok/s |
|---|---:|---:|---:|---:|---:|
| F16 | 32 | 1,421 → 1,503 | 143 → 241 | 1.69× | 3,184 / 493 |
| F16 | 128 | 1,572 → 1,591 | 142 → 233 | 1.64× | 4,226 / 495 |
| F16 | 384 | 1,513 → 1,573 | 132 → 209 | 1.59× | 4,534 / 489 |
| Q4_0 | 32 | 1,228 → 1,487 | 193 → 401 | 2.08× | 3,713 / 815 |
| Q4_0 | 128 | 1,330 → 1,551 | 184 → 375 | 2.04× | 6,092 / 811 |
| Q4_0 | 384 | 1,295 → 1,539 | 164 → 322 | 1.97× | 7,041 / 790 |

Evidence: [F16 before](data/lfm2-230m-compiler-f16-before.json),
[F16 after](data/lfm2-230m-compiler-f16-after.json),
[Q4 before](data/lfm2-230m-compiler-q4_0-before.json),
[Q4 after](data/lfm2-230m-compiler-q4_0-after.json).

The unchanged 39-token greedy answer takes 0.279 → 0.175 s F16 and
0.250 → 0.124 s Q4_0. These medians include reset, tokenization, prefill, sampling
and completion, excluding model loading. GPU and host greedy output agree.
At prefix 128, native decode remains 2.12× faster for F16 and 2.16× for Q4_0;
native prefill remains 2.66× and 3.93× faster respectively.

## Compiler and runtime changes

`T.gemm` now lowers to 2×2 private accumulator microtiles, with configurable
`tensor.webgpu.gemm_microtile` values 1/2/4. It retains transpose and clear/accumulate
semantics and falls back to one along tile axes that cannot divide evenly.
The reusable producer helper `register_matmul_schedule` owns thread distribution,
shared tiles/padding, barriers, private accumulators across all K tiles and guarded
M/N tail stores. LFM2 supplies packed-weight and FP16 operand-rounding expressions
to that compiler schedule. Its selected prefill tile is 16×32 with K=64, after
comparing eight shapes on actual gate/down matrices for both formats. Wider
output tiles were generally slower. The full-run F16 prefill gain is small;
individual F16 projection groups do not all improve.
The [F16 tile sweep](data/lfm2-230m-compiler-f16-tile-tune.json) and
[Q4 tile sweep](data/lfm2-230m-compiler-q4_0-tile-tune.json) retain timings and
FP16-operand numerical comparisons for all eight configurations.

FP32 sum/max tile reductions offer `tensor.webgpu.reduction="tree"`, including
odd-extent identity padding and `clear=False` accumulation. Ordered reduction
remains the default. Tree order changes rounding; LFM2 prefill RMS retains its
previous ordered sum because the earlier parallel experiment failed accuracy.

Typed compiler WGSL helpers use `unpack2x16float` for packed half conversion and
unsigned shifts for unaligned packed u32 reads. Subgroup operations and the
runtime subgroup-size builtin are reflected into required adapter features.
Packed four-byte integer-dot emission is supported for experiments.
Compiler hashes invalidate the producer's previously emitted artifacts even when
the input template text is unchanged.

The [isolated compiler benchmark](data/lfm2-230m-compiler-lowering-bench.json)
measures a 128×512 times 512×256 GEMM at 0.483 ms with scalar output accumulators,
0.299 ms with 2×2 microtiles (1.62×), and 0.277 ms with 4×4 (1.74×). A 64-row,
1,024-element sum-of-squares reduction takes 0.473 ms ordered versus 0.159 ms
tree (2.97×). Each result is a median of five samples after warmup, ten prepared
launches plus completion/readback per sample; numerical references are checked.
These are measurements of those shapes, not guarantees for every schedule.

Prepared plans validate each distinct resource once per launch, reuse bind groups,
skip consecutive repeated pipeline bindings, and avoid allocating an empty
dynamic-offset FFI array per dispatch. The fast path uses wgpu 0.29's native safe
FFI entrypoint, retains resource lifetime checks and owns its groups. Readback
staging buffers are reused and destroyed with their source buffers; returned
NumPy arrays own their memory. The private wgpu interface is version-pinned and
requires revalidation when upgrading.

## LFM2 kernel changes and profiling

The optional subgroup profile combines four-wide F16 decode dot products,
word-packed Q4 decode with eight output rows per workgroup, subgroup RMS and
segmented projection reductions. Segments check the runtime subgroup size and
fall back to shared-memory trees for smaller physical groups. Q6_K retains its
floating-point decoder with the new half unpacking and reductions.

Attention computes head scores across adjacent channels. Decode splits score
production into a separate coalesced kernel, followed by softmax/value reduction;
the extra score workspace is 36,864 bytes. Active-prefix loops skip unused cache
positions. Prefill similarly distributes score dots across channels. Workgroup
barriers stay outside buffer-dependent loops. The separate stage adds six decode
dispatches, but reduces total measured latency.

Native timestamp profiles use the verified Vulkan timestamp period of **10 ns**,
seven samples after warmup, fresh cache prefixes, and same-pass timestamp writes.
Per-dispatch timestamps perturb scheduling; their sums are diagnostic, while
the uninstrumented completed-call table above is the acceptance measurement.

| Format | Whole decode GPU, before → after | Whole prefill GPU, before → after | Decode encode/submit, before → after |
|---|---:|---:|---:|
| F16 | 4.027 → 2.771 ms | 20.013 → 19.879 ms | 1.282 → 0.929 ms |
| Q4_0 | 2.856 → 1.492 ms | 28.851 → 23.772 ms | 1.120 → 0.977 ms |

Evidence: [F16 before profile](data/lfm2-230m-compiler-f16-profile-before.json),
[F16 after profile](data/lfm2-230m-compiler-f16-profile-after.json),
[Q4 before profile](data/lfm2-230m-compiler-q4_0-profile-before.json),
[Q4 after profile](data/lfm2-230m-compiler-q4_0-profile-after.json).
Instrumented decode projection totals fall from 3.212 to 2.328 ms F16 and
2.094 to 0.964 ms Q4_0. Attention, including the new score stage, falls from
0.522 to 0.284 ms F16 and 0.568 to 0.328 ms Q4_0. Host completion/readback
includes GPU execution and must not be added to whole GPU time.

## Correctness and rejected experiment

Both selected bundles pass all 19 full-vocabulary fixtures, covering chat,
cached continuations, chunk boundaries, 511-plus-one capacity and bitwise reset.
Worst NumPy-relative RMS is 0.207% F16 and 0.340% Q4_0; minimum cosine is
0.99999790 and 0.99999424. Every fixture argmax agrees. The gates remain RMS
below 1%, cosine above 0.9999 and identical argmax. This does not establish
bitwise equality with native llama.cpp or identical arbitrary generations.

The [83-test suite](data/lfm2-230m-compiler-tests.xml) covers generic GEMM
microtiles/transposes/tails, odd-axis sum/max accumulation, subgroup feature
metadata tampering, runtime resource lifetime, reusable readback, packed weight
fields, smaller-subgroup fallback, every half bit pattern, Q8 packing/dot arithmetic,
generation, reset, capacity and queued execution. The
[fresh installed-wheel consumer](data/lfm2-230m-compiler-clean-consumer.json)
reproduces both saved logits and the complete generated answer without Torch,
TileLang, TVM, Triton, GGML or llama.cpp installed/imported. CUDA GPU acceptance
was not rerun on this AMD machine.

The benchmark-only Q8 activation experiment replaces 14 FFN-down decode
projections with activation quantization plus `dot4I8Packed`. Its component
arithmetic passes an independent quantized reference, but **six of the 19 full
logit fixtures fail** the existing gate: worst RMS is 1.621%, minimum cosine
0.99987945. Argmax agreement alone is insufficient. It remains excluded from
production requirements and bundles. The
[rejected experiment](data/lfm2-230m-compiler-int8-rejected.json) is retained.

## Reproduction and remaining gap

Build matching wheels/bundles using the [package instructions](../../packages/tensor-llm/README.md).
`--webgpu-profile subgroup` selects this measured Radeon profile; omitting it
retains the portable default. Unsupported subgroup adapters fail feature checks.
The portable profile also passes all 19 replay fixtures for
[F16](data/lfm2-230m-compiler-f16-portable-validation.json) and
[Q4_0](data/lfm2-230m-compiler-q4_0-portable-validation.json).
No vendor matrix or asynchronous shared-copy lowering is claimed: the native
llama.cpp device report has no usable matrix cores here, and pipeline stage
annotations still lower to synchronous SIMT execution.

```powershell
$env:OPENBLAS_NUM_THREADS='6'
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_run.py --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --reference build/llama-vulkan-b11310 --out build/lfm2-230m-q4_0-webgpu-validation
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_profile.py --model build/lfm2-230m-models/LFM2.5-230M-Q4_0.gguf --bundle build/lfm2-230m-q4_0-webgpu --out build/lfm2-230m-q4_0-compiler-profile-final.json --timestamp-period-ns 10 --repeats 7
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_lowering_bench.py
$env:TENSOR_WEBGPU='1'
$env:TENSOR_LFM2_WEBGPU='1'
.venv/Scripts/python.exe -m pytest tests/providers/test_webgpu.py tests/providers/test_webgpu_audit.py tests/providers/test_webgpu_lowering.py packages/tensor-llm/tests/test_gguf.py packages/tensor-llm/tests/test_contracts.py packages/tensor-llm/tests/test_webgpu.py
build/lfm2-230m-compiler-consumer/Scripts/python.exe -I benchmarks/lfm2/webgpu_consumer.py --root (Get-Location).Path --out build/lfm2-230m-compiler-clean-consumer.json
```

The [verification record](data/lfm2-230m-compiler-verification.json) retains source,
bundle, wheel and evidence hashes. Full logits, checkpoints, copied baselines and
environments remain under ignored `build/`. Remaining priorities are projection
layouts/occupancy, especially F16 output projection and prefill; lower host
encoding/completion cost; and an integer-dot approach that satisfies the unchanged
accuracy contract. Other adapters, longer contexts and quantizations require
their own measurements.
