# LFM2.5-2.6B QAD Q4_0: the 1K prefill experiment

[Research index](README.md) · [Previous packed search](lfm2-2.6b-q4_0-parity.md) ·
[Measurements, sources and generated WGSL](data/lfm2-2.6b-prefill-1k.json)

On 2026-10-03, the RX 6700 XT reaches **1,058 tokens/s at 128 prompt tokens**
and **1,047 tokens/s at 384**, clearing the 1K target at both lengths. This is
about **2.01x** the preceding packed control. The opt-in `prefill_mixed` profile
combines searched mixed arithmetic, a shared-layout improvement and reduced
work in the final layers. Decode remains about **165 tokens/s** at prefix 128.
The 32-token prompt reaches 309 tokens/s; the 1K result applies to the larger
prompts. Native llama.cpp still leads prefill by **1.83-2.08x**.

## Completed forward throughput

This uses the same local **LFM2.5-2.6B QAD Q4_0** checkpoint as the preceding
report, SHA256
`a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03`.
Its 30 layers have width 2048, FFN width 10752, 32 query heads, eight KV heads,
64-dimensional heads and vocabulary 128000. All 166 Q4_0 matrices and the Q6_K
tied embedding/output remain packed. The 215 MB output matrix requires
`max_buffer_size=268435456`.

The machine is an RX 6700 XT 12 GB with AMD Vulkan 26.6.2, Ryzen 5 5600 and
32 GB RAM. Tensor uses wgpu-py 0.29.0 / wgpu-native 27.0.2.0. Native llama.cpp
is pinned to b11310, commit `f872b591121761ac7b2af18283bd99bdc092a63a`.

Each measured forward completes and returns host F32 logits. Context is 512;
three warmups precede seven samples, with runner order rotating sequentially
on this GPU. Loading, AOT compilation, reset and sampling are excluded.
Both Tensor bundles expose rows 1/32/128. Native uses Vulkan for all layers,
Flash Attention, F16 K/V, six CPU threads and batch/ubatch 128. The comparison
retains only two Tensor runners and one native runner to avoid memory pressure.

All rates below are tokens/s from median completed latency. The control is the preceding `quant_searched`
profile, rebuilt and measured alongside the new `prefill_mixed` profile.

| Prompt tokens | Tensor control | Tensor mixed | Gain | llama.cpp Vulkan |
|---:|---:|---:|---:|---:|
| 32 | 291.6 | **309.2** | **1.06x** | 935.4 |
| 128 | 525.6 | **1,058.4** | **2.01x** | 1,940.9 |
| 384 | 519.1 | **1,046.9** | **2.02x** | 2,176.4 |

All seven mixed-profile samples exceed 1K at both target lengths: ranges are
1,051.2–1,061.8 tokens/s at 128 and 1,035.4–1,050.1 at 384.

Decode uses the same 64 forced tokens after each prefix:

| Prefix tokens | Tensor control | Tensor mixed | llama.cpp Vulkan |
|---:|---:|---:|---:|
| 32 | 165.8 | **166.3** | 170.2 |
| 128 | 164.9 | **165.0** | 170.0 |
| 384 | 160.5 | **161.3** | 169.3 |

The control allocates **1,633,086,092 bytes**; the mixed
profile allocates **1,638,382,228 bytes**. Rows-1 plans both
use 246 dispatches. Rows-128 plans use 307 control versus 385 mixed dispatches;
activation quantization adds dispatches while suffix liveness reduces processed
rows. State-only intermediate mixed chunks use 377 dispatches at rows 128 and
298 at rows 32. Decode shader identity is checked before timing.

## What changed

The compiler now supports packed adjacent F16 operands in shared `u32` tiles,
short private `vec2<f16>` FMA chains, and a generic packed-integer matmul
schedule. These are producer-selected schedules. Ordinary GEMM defaults stay
available. Search evaluates complete projection families with every affected
weight matrix streamed to a distinct output, then rotates finalists against
the control and checks three held-out input scales.

The selected mixed profile combines two arithmetic contracts:

- FFN gate/up/SwiGLU and convolution input projections use nearest-even F16
  operands. Each even/odd K lane accumulates eight fused operations in F16,
  then contributes to a complete-K F32 total. Gate/up totals and SwiGLU output
  are F32. This introduces additional rounding compared with the control.
- Other selected rows-128 projections first round activations to F16, then
  represent each 32-element block with a Q8 component and a second Q8 residual
  component. Packed signed integer dots use the original Q4 fields, subtract
  their zero offset, and apply exact decoded F32 weight scales. Totals are F32.
  Both quantization dispatches and projection work are included in timings.

Parallel per-row RMS reductions also replace the serial prefill reductions.
The `prefill_q16` profile exposes the integer-only experiment; `prefill_mixed`
selects the measured combination. Rows-1 decode projections retain the
preceding searched F32 arithmetic and byte-identical generated WGSL.

The final layout replay lowers the paired FFN from **1.951 ms to 1.710 ms**,
a **12.4% latency reduction**, against the previously selected short-half
schedule. One packed word of shared padding and row-first workgroup ordering
are selected. Padding supplies most of the observed gain; alternate workgroup
orders differ only slightly. The padded FFN uses 6,240 bytes of shared storage.
Convolution input replay remains about **0.666 ms**, so its existing layout is
retained. Search records preserve both the original F32-total control and the
previous short-half schedule.

There is also a guarded runtime optimization for the final
attention/convolution/convolution suffix. All final-attention K/V entries are
stored before retaining just the final eight queries and residual rows.
The following two three-tap convolutions need a five-row receptive field;
eight rows preserve the final output and both persistent convolution histories.
Discarded early suffix outputs need not agree, because they have no remaining
consumer. Absolute query positions and the complete K/V cache are retained.
Only the last token enters the final FFN and output head. Its FFN uses the
existing rows-1 F32 decode arithmetic.

Intermediate chunks stop after the final convolution updates persistent state,
skipping unused final projections, FFN and logits. The final chunk always
computes logits, including `read=False` calls. Public chunk sizes remain
1/32/128; the eight-row suffix uses internal tiles. Those small tiles were
selected manually and validated, rather than discovered by the rows-128 search.
This specialization requires the measured geometry, suffix and Q4 matrix
types; unsupported cases retain their ordinary plans.

## Accuracy and limits

All **19 independent NumPy logit fixtures** pass. Maximum relative RMS is
**0.71023%** and minimum cosine is **0.99999259**, with **19/19 matching argmax**.
Reset logits are bitwise stable. Both Tensor profiles produce identical token
IDs in the bounded 96-token generation check, using both host and GPU greedy
sampling. Fixtures cover cached decode, partial prefill boundaries and the
511-token boundary; full-model gates use the same cached oracle as the control.

The numerical acceptance gate remains finite output, relative RMS below 1%,
cosine above 0.9999 and matching argmax against independent NumPy fixtures.
The new arithmetic is approximate; it is not bitwise equivalent to the
preceding prefill contract. Native ordinary Vulkan arithmetic has a different
contract: its maximum relative RMS against these fixtures is about 10.88%,
although all 19 argmax values match. The native column is a throughput comparison
with explicit accuracy results, rather than precision-equivalent parity.

The final targeted suite passes **183 tests**, with four inapplicable
F32/packed-half combinations skipped. This includes compiler resource legality
and both workgroup grid orders, M/N/K tails, fused epilogues and suffix controls.

Independent checks cover explicit per-operation half rounding, packed Q4/Q6
fields, integer-dot math and activation reconstruction, reduction tails and
zero/subnormal values. A float64 causal-window test verifies consecutive chunks
and both convolution histories; GPU extraction tests cover partial chunks and
position controls. A full-query versus cropped-query GPU attention test checks
absolute positions against an independent float64 softmax reference.

The final timestamp diagnostic measures **121.46 ms** for the whole rows-128
GPU pass at position 96. Per-kind median sums are **67.79 ms linear**,
**46.52 ms FFN**, **3.52 ms attention** and **1.36 ms activation quantization**.
Projections account for roughly 94% of the instrumented dispatch sum.
Rows-1 whole GPU time remains about **5.12 ms**. These three-sample diagnostic
traces use Vulkan timestampPeriod 10 ns; per-dispatch instrumentation perturbs
scheduling, so the seven-sample ordinary forward rates establish acceptance.

These results establish the measured checkpoint, RX 6700 XT and context-512
workload. They do not establish longer-context accuracy, general model coverage
or llama.cpp prefill parity. Kernel-level half error bounds describe the new
arithmetic; full-model fixture gates provide its bounded inference acceptance.

## Rejected searches and retained evidence

The raw archive includes all completed searches and intermediate full-model
comparisons. Two-component activation dots first reached roughly 835/821
tokens/s; selected short-half tiles reached 926/912 in a seven-sample comparison.
Last-row FFN execution and the cropped suffix subsequently reached about
994/987 before the final layout search.

Wider F32 shared tiles, floating vector dots, packed F16 dots with F32 totals,
larger integer K panels, splitting gate/up, fixed residual scales and direct
F16 shared staging did not improve enough to replace the selected runtime.
Predecoded F16 caches and expanded signed-byte Q4 caches were measured only
as benchmark alternatives. The retained inference path has no expanded cache.
Some isolated down/narrow projection gains were too small to justify a change.

The initial integer discovery stopped at duplicate quantizer artifact creation;
its partial report is diagnostic evidence. Early half search labels sometimes
omit microtile dimensions; distinct artifact names and parameter dictionaries
identify candidates. Source snapshots and archive notes explain early timing
descriptions that predate the experimental arithmetic. Final comparison
protocols, implementation hashes, generated WGSL, native release/logs, fixture
array hashes, tests and timestamp traces are preserved with the measurements.

## Reproduce

From the repository root, with the optional native WebGPU extension built:

```powershell
$env:WGPU_BACKEND_TYPE='Vulkan'
$env:OPENBLAS_NUM_THREADS='6'
$env:TENSOR_BUILD_WEBGPU_NATIVE='1'
.venv/Scripts/python.exe setup.py build_ext --inplace
$model='D:/LLM/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-QAD-Q4_0.gguf'
$root='build/lfm2-prefill-1k'
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model $model --out "$root/control" --provider webgpu --webgpu-profile quant_searched --context 512 --prefill-chunks 32 128
.venv/Scripts/python.exe benchmarks/lfm2/producer.py --model $model --out "$root/mixed" --provider webgpu --webgpu-profile prefill_mixed --context 512 --prefill-chunks 32 128
.venv/Scripts/python.exe benchmarks/lfm2/prefill_chase_compare.py --model $model --root $root --reference build/llama-vulkan-b11310 --fixtures build/lfm2-2.6b-q4_0-run-prefill --out "$root/comparison-final" --bundles control mixed --native-chunks 128 --max-buffer-size 268435456 --repeats 7
.venv/Scripts/python.exe benchmarks/lfm2/webgpu_profile.py --model $model --bundle "$root/mixed" --out "$root/profile-mixed.json" --repeats 3 --timestamp-period-ns 10 --max-buffer-size 268435456 --prefill-position 96
.venv/Scripts/python.exe benchmarks/lfm2/prefill_1k_summary.py --root $root --reference build/llama-vulkan-b11310 --out docs/research/data/lfm2-2.6b-prefill-1k.json
```

Run GPU jobs sequentially. `q16_prefill_search.py` exposes subgroup activation
quantization, wider integer K panels, split FFNs, fixed residual scales and
signed-byte cache experiments. `prefill_chase_search.py --encoding 2 --rows 128`
exposes `--wide`, `--cached`, `--dots`, `--packed-pairs`, `--half-accum` and
`--half-layouts`, with `--shapes` to select projection families and `--replay`
for a fresh finalist comparison. The archive contains exact invocations' flags,
candidates and source snapshots. Rebuild bundles after implementation changes;
the archival helper rejects stale implementation/compiler/artifact hashes.
