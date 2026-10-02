# LFM2, tinygrad, and correctness-checked schedule search

This experiment runs the same LFM2.5-230M F16 checkpoint through Tensor,
llama.cpp Vulkan, and a benchmark-only tinygrad adapter on the RX 6700 XT.
It also tests whether tinygrad's BEAM search improves the actual FFN
projections. This is evidence for adding a search layer to Tensor's producer;
it does not establish the compiler's hardware ceiling.

[Raw evidence](data/lfm2-tinygrad-comparison.json) retains the reports, timing
samples, rejected-run logs, pinned adapter sources and selected schedules.

## Scope and reproducibility

The Tensor source baseline is `63d6d6f` and tinygrad is pinned to
`91b8cb5fa6c031c5a7440159d955f66952c5e2e9` (0.14.0). The checkpoint SHA256 is
`4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f`.
The machine is Windows, Ryzen 5 5600, RX 6700 XT (gfx1031/RDNA2), AMD
Vulkan driver 26.6.2 and OpenCL driver `3652.0 (PAL,LC)`.
Tensor uses wgpu 0.29.0 and its native prepared-plan encoder. llama.cpp uses
the unchanged b11310 reference, all layers on Vulkan, six CPU threads,
flash attention and F16 KV storage.

Upstream tinygrad's [LLM runner](https://github.com/tinygrad/tinygrad/blob/91b8cb5fa6c031c5a7440159d955f66952c5e2e9/tinygrad/llm/model.py)
already separates prefill and rollout JITs. First-call compilation does not
prevent measuring warmed prefill. It lacks LFM2's short-convolution blocks,
so `benchmarks/lfm2/tinygrad_reference.py` supplies those blocks using generic
Tensor operations and TinyJit. The upstream repository remains unchanged.

The specialized [AMD LLM kernels](https://github.com/tinygrad/tinygrad/blob/91b8cb5fa6c031c5a7440159d955f66952c5e2e9/tinygrad/llm/kernels/amd.py)
require RDNA3/4, and the pinned native AMD runtime excludes Windows. Therefore
this experiment tests the available generic OpenCL and Vulkan WebGPU paths.
It is not a reproduction of tinygrad's optimized native AMD LLM benchmark.
Vulkan WebGPU uses the official pydawn 0.3.0 Dawn DLL, release SHA256
`45a3a9a9e194067fcb6c6f05ca0e78804c6ead50d21cbec2f18084a829d3d07c`.

The adapter keeps F16 matrix storage, FP32 normalization/activations and
accumulation, and FP16-rounded projection operands for multirow prefill.
It uses separate symbolic 1/32-row JITs, a symbolic token position and valid
row count, and graph-ordered writes for convolution state and FP16 KV caches.
Reset must reproduce the initial logits bitwise. Attention scans a fixed
576-entry masked cache, whereas Tensor bounds attention to the active prefix;
that additional work is part of this adapter's implementation. The full-model
gap cannot be attributed solely to matmul lowering or search.

## Full-model OpenCL comparison

All 19 existing independent NumPy fixtures pass for both Tensor and tinygrad
OpenCL with search disabled. They cover convolution/attention state, partial
prefill chunks, cached decode, prefixes through 511 tokens, and reset.
The original gates remain relative RMS below 1%, cosine above 0.9999,
matching argmax, and finite logits. tinygrad's maximum relative RMS is 0.221%.
llama.cpp accuracy metrics are recorded separately, as in the existing Tensor
benchmark; its internal precision contract differs from the NumPy fixture.

Each cell below is tokens/s from five measured repetitions after three warmups.
Context is 512; prefill chunks are 32; decode forces 64 shared tokens. All
runners return completed host FP32 logits, with sampling excluded. Runner
order rotates and GPU execution is sequential. Model loading and initial
graph compilation/JIT capture precede the warmed samples; recurring backend
pipeline/submission work remains included.

| Prompt | Tensor prefill | tinygrad CL prefill | llama.cpp prefill | Tensor decode | tinygrad CL decode | llama.cpp decode |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1,713 | 747 | 3,126 | 284 | 38.6 | 479 |
| 128 | 1,766 | 871 | 4,160 | 270 | 38.8 | 481 |
| 384 | 1,748 | 897 | 4,494 | 240 | 35.4 | 475 |

With a fresh private tinygrad persistent cache, the first completed prompt
costs 16.14 seconds on tinygrad CL, 0.170 seconds on Tensor, and 0.025 seconds
on llama.cpp, excluding model loading. Tensor's producer compilation is
already complete, so these are consumer first-call costs, not equally priced
end-to-end compilation. tinygrad also captures/compiles its JIT on the next
call; the first call is not its complete preparation cost.

## Vulkan WebGPU correctness boundary

The initial generic tinygrad WebGPU adapter selects the physical AMD Vulkan
device. It passes the first 16 fixtures but fails `prefix_511` with **1.3076% relative
RMS**, cosine 0.99992237, and matching argmax. The configuration is rejected;
that run provides no accepted full-model WebGPU throughput result. The
existing gates are unchanged.

A conversion probe isolates a precision difference: `half().float()` on this
Dawn/Vulkan stack differs from NumPy nearest-even conversion on 16,494 of
32,768 random FP32 inputs, with observed results consistent with truncation
toward zero. Materializing the half intermediate does not remove it. The
unmodified WebGPU gate projection also fails the independent rounded-dot
oracle (maximum absolute error 2.47e-5).

The experimental adapter's `rounded_half` makes nearest-even operand and KV
rounding explicit through ordinary Tensor integer/FP32 operations. A probe
of 32,777 values, including half-subnormal ties and overflow boundaries,
matches NumPy. With that compatibility step, all 19 full-model WebGPU
fixtures pass; maximum relative RMS is 0.2213%. This is an adapter precision
correction on the measured backend, not a change to upstream tinygrad. Its
extra operations are part of the corrected adapter's timings.

The pinned WebGPU runtime creates a pipeline, binding layouts, bind group and
command encoder inside each `WebGPUProgram.__call__`, submits one dispatch,
and releases those objects. TinyJit capture does not give it Tensor's prepared
native submission path. This source-level difference is included in the
completed-call results; it does not establish how much time is arithmetic
versus pipeline/driver work. The corrected generic WebGPU adapter should not
be used as a proxy for tinygrad's optimized native AMD inference path.

| Prompt | Tensor prefill | tinygrad WebGPU prefill | llama.cpp prefill | Tensor decode | tinygrad WebGPU decode | llama.cpp decode |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 1,635 | 94.9 | 2,143 | 283 | 3.30 | 472 |
| 128 | 1,747 | 17.6* | 3,373 | 269 | 3.19 | 475 |
| 384 | 1,743 | 101.2 | 4,222 | 240 | 3.22 | 472 |

The same five-repeat protocol is used in this separate cohort. **The 128-token
WebGPU prefill sample is unstable:** completed times are 1.289, 1.212, 9.008,
8.124 and 7.292 seconds. The table preserves the prescribed median; it is not
evidence of a steady compute rate or monotonic length scaling. Its cause has
not been isolated. The 384-token samples are 3.758--3.826 seconds. All raw
samples remain in the evidence rather than dropping slow repetitions.

With explicit rounding, the isolated no-search WebGPU gate/down projections
also pass the independent dot oracle. Completed-call times are 1,930/4,221 us,
versus Tensor 247/450 us in that cohort. No WebGPU BEAM speedup is claimed.

## Why search belongs in the producer

An unrestricted full-model `JITBEAM=2` run was stopped after several minutes
of search, with 761 compiled-cache entries and five completed BEAM searches.
It never reached warmed measurements and provides no full-model speedup
claim. Its private cache and log remain available under
`build/tinygrad-comparison/cl-f16-beam2`.

The more useful bounded experiment isolates the actual F16 gate/down weights
from block 0 with 32-row inputs. `BEAM=0,JITBEAM=2` isolates capture-time
search; upcast product is capped at 32, local product at 256, and
`BEAM_ESTIMATE=0` times full workgroups. Compilation is single-process on
Windows. The selected result is checked against an independent float64 dot
oracle of the required rounded operands using an absolute-product error
bound. A hot matrix isolates scheduling and submission, not whole-model
weight bandwidth.

| F16 projection, 32 rows | Tensor Vulkan | tinygrad CL BEAM=0 | tinygrad CL JITBEAM=2 | Search/capture cost |
|---|---:|---:|---:|---:|
| Gate, K=1024, O=2560 | 242 us | 326 us | 187 us | 77.9 s |
| Down, K=2560, O=1024 | 445 us | 269 us | 191 us | 99.2 s |

Tensor columns use the contemporaneous search-run medians; the no-search
control measured 249/443 us. BEAM improves the tinygrad projections by
1.75x/1.41x and beats Tensor by 1.30x/2.33x. Both selected programs pass
the unchanged dot oracle. Different backend/runtime stacks and completed-call
overhead remain part of this comparison; these ratios are not pure ALU gains.

A replay using the saved BEAM cache verifies the selected programs again and
measures 183/188 us. Capture now costs 0.22/0.12 seconds rather than a new
78/99-second search. The generated sources and selected actions are archived.

The winners reveal a schedule family absent from the current reusable staged
GEMM helper. The gate partitions K across 8 lanes, holds 10 FP32 partial
accumulators per thread, loads contiguous `half2` weights and `float2`
activations, then combines partial sums through a 5 KiB local buffer after
one barrier. The down projection partitions K across 16 lanes, holds 8 partial
accumulators, and combines through an 8 KiB local buffer. Neither stages the
input tiles in local memory. By contrast, Tensor's helper retains private
accumulators across the full K axis and stages operand tiles with two barriers
per tile. These are source observations, not measured register/occupancy
counters. Add workgroup-local K partitioning/direct loads as a new legal seed
family, then search it alongside the current staged microtile family.

The full-model and projection harnesses preserve their numerical checks and
write raw timing samples. Completed-call projection timing includes submission
and synchronization; it is not a pure GPU timestamp or an occupancy counter.

## Proposed Tensor architecture

Tensor already has `tensor.compiler.tuning.tune`: it selects among precompiled
CUDA candidates, rejects wrong outputs and saved intermediates, and measures
CUDA-event latency. Candidate generation and compilation are caller-owned.
The current API does not supply WebGPU search, a beam, or persistent selection.
The existing WebGPU register scheduler and benchmark sweeps provide initial
legal programs rather than requiring a new front end.

The useful next layer is:

1. **Semantic contract and NumPy oracle.** Specify operand rounding,
   accumulation precision, layout, epilogues, and state transitions. Retain
   multiple randomized/adversarial fixtures and saved intermediates; use
   independent higher-precision dot references where appropriate.
2. **Initial portable tile program.** Start from the current passing schedule
   and exported IR. Keep that schedule as the fallback.
3. **Candidate generator.** Search tile M/N/K, per-thread register microtiles,
   thread-to-output mapping, vector width, reduction partition, staging and
   local-memory layout. Treat bank padding/swizzles, packed dequantization,
   and fusion choices as explicit IR/lowering transformations with their own
   legality checks. Search cannot discover a layout the IR cannot express.
4. **Compile, validate, measure.** Reject resource/legality violations before
   compilation; reject candidates that fail the oracle before timing. Measure
   full shapes on the actual adapter, with warmups, repeated robust samples,
   isolated GPU execution and a bounded wall-clock/candidate budget. Rank
   finalists by both GPU time and prepared-call time; remeasure close winners.
5. **Persistent AOT selection.** Cache winners by semantic IR hash, shape,
   dtype/precision, epilogue, compiler/lowering version, adapter, driver,
   features and limits. Emit ordinary `.tbin` artifacts so the consumer remains
   compiler-free and incurs no search on the first prompt.
6. **Model acceptance.** Repeat all stateful LFM2 fixtures and matched warmed
   prefill/decode measurements before promoting winners. A hot projection
   win may disappear under model bandwidth, dispatch or attention costs.

[tinygrad's search](https://github.com/tinygrad/tinygrad/blob/91b8cb5fa6c031c5a7440159d955f66952c5e2e9/tinygrad/codegen/opt/search.py)
is a useful model for beam expansion, compilation caching and hardware timing.
Its search loop does not run a NumPy oracle on every candidate; Tensor's
explicit precision/state contract should remain the acceptance boundary.
Keep schedule search separate from more general algebraic discovery: begin
with a small legal action space around the existing GEMM/GEMV programs, then
add new transformations when a specific measured bottleneck warrants them.

## Harness entry points

`benchmarks/lfm2/tinygrad_compare.py` accepts the existing model, bundle,
llama.cpp reference and cached NumPy fixture paths. `--weight-mode packed`
is the default; `decoded` is a separately labeled experiment and is not used
for the accepted numbers here. `tinygrad_projection_compare.py` accepts the
model/bundle and `--rows 32` for prefill-only search.

Install pinned tinygrad in a separate consumer environment alongside the
same clean Tensor/tensor-llm wheels used by the existing Vulkan benchmark.
Set `DEV=CL` or `DEV=WEBGPU`; for WebGPU set
`WEBGPU_BACKEND=WGPUBackendType_Vulkan`. Use separate absolute Windows
`CACHEDB` paths for every configuration. Search configuration is recorded in
the projection reports; persistent cache files are build outputs rather than
shipped artifacts.

For the bounded OpenCL projection search, with the isolated consumer Python
on PATH and the existing bundle/NumPy validation outputs available:

```powershell
$env:DEV='CL'
$env:BEAM='0'
$env:JITBEAM='2'
$env:BEAM_ESTIMATE='0'
$env:BEAM_UPCAST_MAX='32'
$env:BEAM_LOCAL_MAX='256'
$env:PARALLEL='0'
$env:CACHEDB=Join-Path (Get-Location) 'build\tinygrad-comparison\reproduce-cl-beam2\cache.db'
python benchmarks/lfm2/tinygrad_projection_compare.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --bundle build/lfm2-230m-f16-webgpu --out build/tinygrad-comparison/reproduce-cl-beam2 --rows 32
```

For full inference, set `JITBEAM=0`, choose a fresh private cache, and use:

```powershell
python benchmarks/lfm2/tinygrad_compare.py --model build/lfm2-230m-models/LFM2.5-230M-F16.gguf --bundle build/lfm2-230m-f16-webgpu --reference build/llama-vulkan-b11310 --fixtures build/lfm2-230m-f16-webgpu-validation --out build/tinygrad-comparison/reproduce-full
```

F16 is deliberate here: it isolates the compute/schedule question from packed
dequantization. Quantized LFM2 and a complete model with BEAM enabled remain
separate experiments; neither has an accepted result in this report.
