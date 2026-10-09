# Tensor 1.0 benchmark and demonstration program

[Path to 1.0](v1.md) · [Plans](README.md) · [Existing measurements](../research/README.md)

Planning baseline: **2026-10-05**. This records the requested 1.0 comparison
program. The comparison targets below remain planned for Tensor; baseline GPU
measurements are linked separately and do not establish Tensor model support. CPU/GPU composition and
SSD offloading remain proposed extras/addons after 1.0. The **2026-10-07** addition
below scopes a minimalist Strata-inspired hybrid MoE demonstration in that
follow-on track; it does not add a core 1.0 release gate.

The **2026-10-09** [unified Tensor LLM engine plan](unified-llm-engine.md)
specifies one extension of `packages/tensor-llm` for both llama.cpp-like batch-1
latency and vLLM-like batch throughput. BENCH-03/04/09 share model loading,
request state, cache ownership and execution infrastructure, with specialized
kernel schedules and scheduling policies rather than separate engines.

The independent [LLM serving harness](../guides/llm-serving-benchmarks.md) now
implements shared token-ID workloads, vLLM/SGLang/llama.cpp protocol adapters,
concurrency sweeps, telemetry and plots. Synthetic HTTP checks and a
[real Qwen3-0.6B L40S pilot](../research/llm-serving-l40s-pilot.md) qualify the
measurement machinery on all three engines. A [bounded 35B L40S stress run](../research/llm-serving-l40s-stress.md)
also completes 32K/16K requests at concurrency 1, 2 and 4 with native FP8
vLLM/SGLang and separately labeled converted GGUF llama.cpp. Broader model/hardware
coverage, numerical quality and Tensor request/resource contracts remain required;
this is partial BENCH-01/LLM-01 implementation.

The program has three outputs: comparable kernel measurements, complete model
inference comparisons, and reproducible demonstrations of sustained training,
image generation, audio-to-audio inference, and the full llama.cpp quantization
lineup. Publish correctness, coverage,
and reproducibility alongside performance. Winning every comparison is not a
release criterion. Unsupported reference cells must remain visible.

## Requested comparison matrix

| Track | Workload | Implementations | Hardware |
|---|---|---|---|
| LLM inference | [Qwen3.5-35B-A3B-FP8](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) | Tensor, vLLM, SGLang, llama.cpp (format audit required) | L40S, H100 |
| LLM inference | [Qwen3.5-27B-FP8](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) | Tensor, vLLM, SGLang, llama.cpp (format audit required) | L40S, H100 |
| Flash Attention | Forward and backward | Tensor with search, TileLang, Triton, upstream FlashAttention | A10G, L4, L40S, H100, B200 |
| Flash MLA | Forward and backward, with exact operator contracts established first | Tensor with search, TileLang, Triton, upstream FlashMLA | A10G, L4, L40S, H100, B200 |
| Gated DeltaNet (GDN) | Forward and backward | Tensor with search, TileLang, Triton, upstream FLA | A10G, L4, L40S, H100, B200 |

Add Tensor's fixed/default schedule to every kernel track as an ablation. It
shows how much search contributes separately from runtime and kernel changes.
The matrix lists evaluation targets, not blanket backend support guarantees.

All LLM case matrices retain **vLLM, SGLang and llama.cpp** across batch-1,
offline batch sweeps and serving, with a result or explicit disposition for each
baseline. The llama.cpp rows require an actual checkpoint/format capability audit;
this table does not assert native support for these FP8 safetensors. A converted
GGUF comparison is a separately labeled, numerically qualified case using
identical GGUF files in Tensor and llama.cpp. It cannot replace an unsupported
identical-FP8 cell or be silently pooled into its speedup summary.

| Demonstration | Comparison | Required outcome |
|---|---|---|
| nanoGPT training over **1,000,000,000 training tokens** | Tensor versus a matched PyTorch baseline | Continuing training, loss/validation curves, completed throughput, memory, checkpoint/resume |
| [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) image generation | Tensor versus baseline Diffusers | Complete text-to-image pipeline, retained outputs, quality checks and end-to-end timing |
| [LFM2.5-Audio-1.5B-GGUF](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF) audio-to-audio | Tensor versus the checkpoint's audio runner | Input waveform through generated waveform, retained audio/text, quality and latency checks |
| Full llama.cpp GGUF weight-quantization lineup | Tensor versus pinned llama.cpp, using identical GGUF files | Exhaustive preset inventory, decoded-weight/operator checks, full-model quality, size/memory and prefill/decode measurements |
| Minimalist hybrid MoE block (**post-1.0 addon**) | Tensor CPU/GPU composition versus matched all-CPU and, where it fits, all-GPU execution | Correct routed-expert outputs under a fixed VRAM budget, residency/transfer accounting and completed-operation timing |

Demonstration GPU profiles, training model size/dataset, and resource allocation
are still decisions to make. The 1B requirement counts training tokens, not model
parameters. Do not silently extend the two LLM GPUs or five kernel GPUs to every
demonstration before its memory and execution requirements are established.

## Implementation prerequisites

The existing [LLM package](../../packages/tensor-llm/README.md) implements bounded
single-sequence LFM2 text inference. It does not establish Qwen, FP8 safetensors,
MoE, image generation, or a complete audio pipeline. The existing nanoGPT
[consumer](../../benchmarks/nanogpt/consumer.py) verifies ten synthetic updates;
its [benchmark](../../benchmarks/nanogpt/benchmark.py) resets state between timed
windows. Neither is evidence of a continuous 1B-token training run.

The engine implementation stages are [LLM-01–07](unified-llm-engine.md#staged-implementation-and-acceptance):
qualify the existing batch-1 path, separate shared model resources from request
state, add true LFM2 batching, then cache pooling/continuous scheduling and Qwen
adapters. Search and dispatch qualification run throughout. Local generation and
serving use the same engine; batch loops over independent model instances are
not the intended throughput implementation.

| Item | Development needed before measurement | Acceptance evidence |
|---|---|---|
| BENCH-01 — Harness and contracts | Versioned case/result manifests, baseline adapters, numerical gates, device profiling, retained search/replay reports | One complete small case per track; failures and unsupported cells reported explicitly |
| BENCH-02 — Attention families | Qualified forward/backward implementations and legal search spaces for FA, MLA, GDN; target-specific lowering/build validation | Outputs and required gradients pass references on each advertised hardware/precision profile |
| BENCH-03 — Qwen FP8 models | Exact checkpoint loading/scales, required precision representation, dense and MoE execution, hybrid attention/GDN state, tokenization and generation | Small-layer numerical checks, full-model logits/quality checks, reset/state isolation and memory preflight |
| BENCH-04 — LLM engine comparison | Extend `tensor-llm` with shared model/request execution, batch-1 and batched kernel dispatch, cache management, continuous scheduling and a serving adapter; matched offline/serving harness | Batch-1 regression gate and batch throughput evidence from the same engine; same request trace and policies; prefill/decode and client latency measured separately |
| BENCH-05 — Sustained nanoGPT | Dataset/tokenizer pipeline, persistent optimizer state, learning-rate schedule, checkpoint/resume, long-run monitoring | Resume verification and intermediate checks followed by both matched 1B-token runs |
| BENCH-06 — Image pipeline | Text conditioning, denoising model and scheduler, VAE decode, required precision and supported resolution profiles | Intermediate numerical checks and complete saved images under matched settings |
| BENCH-07 — Audio pipeline | Audio preprocessing/encoder, projectors, interleaved generation, audio token decoding/vocoder, complete checkpoint closure | Complete audio-to-audio outputs and streaming/reset checks where streaming is claimed |
| BENCH-08 — Release report | Physical hardware runs, raw records, reproduction commands, coverage table and published methodology | Exact candidate provenance; every planned cell has a result or documented disposition |
| BENCH-09 — llama.cpp quantization lineup | Complete preset/type inventory, GGUF readers and packed execution kernels, calibration/quantization pipeline, representative dense/MoE model profiles | Every canonical preset has explicit coverage; applicable Tensor paths pass decoded-weight, operator and complete inference checks against the same GGUF files |
| BENCH-10 — Minimalist hybrid MoE (post-1.0) | Model-correct top-k routing, packed CPU/GPU expert kernels, fixed residency plan, bounded device buffers and weighted result merge | One MoE block passes an independent reference across residency cases; complete latency and memory accounting establish when the split helps |

Workload-specific backward callbacks can fulfill the kernel and training tracks;
these requirements do not require a general autograd framework. Optional model
packages can retain their own stability labels while demonstrating core 1.0.
Any added runtime precision/ABI capability must follow the compatibility work in
[M1/M2](v1.md), with clear rejection by runtimes lacking that capability.

## Reference selection and capability audit

Pin a release or commit for every baseline and verify its actual entry point,
semantics, dtype, direction, head dimensions, and hardware requirements before
booking runs. Names such as "Triton" and "TileLang" identify implementation
systems; each row must identify the concrete kernel source and tuning method.

- **FlashAttention:** the [upstream documentation](https://github.com/Dao-AILab/flash-attention)
  distinguishes FA2 for Ampere/Ada/Hopper, FA3 for Hopper, and FA4 for
  Hopper/Blackwell. Use an appropriate qualified revision for each GPU and label
  it explicitly. FP8 forward support does not imply FP8 backward support.
- **FlashMLA:** the [current upstream documentation](https://github.com/deepseek-ai/FlashMLA)
  specifies SM100/SM103 for its NVIDIA path, whereas its history records Hopper
  implementations. Select and label a compatible older revision for Hopper if
  needed; do not assume A10G/L4/L40S support. It documents sparse MLA inference
  and a separate dense MHA forward/backward path. Specify latent attention,
  prefill/decode, dense/sparse, cache representation, and the gradients being
  compared. Dense MHA backward cannot stand in for MLA backward. If an equivalent
  upstream backward entry point is absent, keep that cell unsupported and retain
  the requested Tensor/TileLang/Triton gradient study with an independent reference.
- **GDN:** pin the original gated delta rule and its exact
  [FLA implementation](https://github.com/fla-org/flash-linear-attention), including
  chunk/recurrent mode, normalization and state conventions. FLA includes
  Triton-based implementations: a separate Triton baseline must identify distinct
  source or be explicitly marked as the same implementation, not a second
  independent competitor. Do not substitute GDN-2 or another recurrence.

The [35B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) describes
hybrid Gated DeltaNet/attention with MoE; the
[27B card](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) describes hybrid
Gated DeltaNet/attention with dense FFNs. Both specify block-scaled FP8 weights.
Implement their actual architecture and quantization contract. The MLA comparison
is an independent kernel track, not a claim that these Qwen models use MLA.

Recheck these upstream descriptions when freezing the benchmark versions. This
audit is a planning snapshot, not a permanent support matrix.

## Fair measurement protocol

Freeze the case manifest and acceptance tolerances before selecting winners.
Each case records shapes, strides/layout, dtype/accumulation, masks, variable
lengths, state/cache contents, outputs, gradients, seed, and input/reference hashes.
Include small correctness cases, boundary/tail cases, and representative large
workloads. Choose final shape sweeps from model profiles and published reference
contracts; retain every case, including ones unfavorable to Tensor.

For attention backward, define required dQ/dK/dV and any other trainable inputs,
the upstream gradient, and saved forward intermediates. Report backward-only
and forward-plus-backward timings separately, including recomputation in the
latter. For GDN, define gates and initial/final recurrent state, their gradients,
chunk boundaries, and whether state is detached. Decode inference is a separate
case from training backward. Check finite outputs, absolute/relative error and
relevant gradient tolerances against an independent reference before timing.

For Tensor search, retain the fixed baseline, space, seeds, legality rules,
candidate/time budgets, every rejected/failed trial, metric, winner, and profile
provenance. Report producer compilation/search cost separately from warm execution.
Compare to baseline-native autotuning where available, disclosing its budget and
settings. Rebuild and correctness-check the winner in a fresh process, then time
it in independent repetitions. State whether tuning is per shape/device or profiles
are reused. Do not select the winner and report its noisy selection timing as the
final result. Inspect matching profile/device identity on every replay.

Use the same timing boundary across competitors. Record completed GPU execution
and completed host-visible latency separately; disclose eager versus CUDA-graph
execution, warmup, repetitions and synchronization. Exclude build/search/loading
from warm kernel time but report them. Include required packing, metadata and
layout conversion in a separate complete-operation measurement so costs cannot
be hidden. For model demonstrations, also measure the complete user-visible path.

Run competitors independently on an otherwise idle device. Record medians and
dispersion, repeated-run counts, driver/toolkit/compiler versions, GPU SKU/VRAM,
SM, power/clock settings, CPU/RAM, and OS. Distinguish H100 PCIe/SXM or other SKU
variants and single GPU from tensor-parallel runs. Report memory measurement
method and both live allocations and allocator reservations where available.

Every planned implementation/device/direction/case cell gets one disposition:
`measured`, `unsupported`, `not-implemented`, `incorrect`, `oom`, `environment-failed`,
or `not-run` (for example, unavailable hardware). Keep logs/reasons and pinned
source evidence. An unsupported upstream cell cannot become a Tensor speedup,
zero-time result, or silent omission. Summarize performance only over comparable
correct cells, with the shared coverage denominator. Publish coverage separately.

## LLM inference protocol

Use the same model revision, weight/scales format, tokenizer/chat template,
activation/cache precision, prompt token IDs, generation length, sampling and
stop policy. Begin with a single GPU per run on each requested device. Preflight
actual weight, workspace, KV and recurrent-state memory; record OOM at the
declared case instead of changing precision, context or offloading one competitor.
Any later multi-GPU comparison is a separately specified configuration.

BENCH-04's baseline adapters must cover vLLM, SGLang and llama.cpp. Retain all
three across the same batch-1, batch-throughput and serving case matrix, with
pinned versions/settings and explicit capability/format dispositions. Measure
each supported path rather than assigning each baseline only one performance
regime. BENCH-09's GGUF quantization comparisons complement these engine sweeps.

Publish two measurements with their own boundaries:

1. **Offline execution:** matched prompt/output length and batch sweeps; completed
   prefill latency/throughput, decode latency/tokens per second, memory, cold load,
   and startup/build/search cost. Default proposed first scope is text-only; vision
   inputs need their own explicitly matched cases.
2. **Serving:** the same client, request arrival/concurrency trace and endpoint
   boundary; TTFT, inter-token latency, request latency percentiles, throughput,
   errors and goodput under a declared latency target. Match prefix-cache policy,
   batching, maximum context and generation settings, and disclose speculative
   decoding. Include warm/cold cache cases when caching is enabled.

Tensor needs a serving implementation before the second comparison is valid.
A single-sequence kernel loop cannot be presented as an equivalent serving
benchmark against vLLM/SGLang. Token/logit checks and an agreed quality task
accompany throughput; fast but incorrect or degraded outputs do not qualify.

Use the [unified engine's latency and throughput gates](unified-llm-engine.md#performance-and-comparison-gates)
to report batch-1 and peak-batch behavior independently, including equal and
heterogeneous lengths. Add 32K-input/16K-output as a long-context stress case
alongside short/mixed workloads, subject to checkpoint limits and memory preflight;
retain unsupported/OOM dispositions on the requested L40S/H100 profiles. H200
replication would be an explicitly additional case. Record client concurrency
and actual resident batch separately. Begin with matched autoregressive controls
and label speculative configurations separately; one mode's throughput cannot
hide another mode's latency regression.

The [first L40S throughput target](unified-llm-engine.md#first-l40s-throughput-target)
prioritizes beating the measured native-FP8 concurrency-4 stress baselines,
qualifying eight resident 48K contexts, and comparing all three baselines again
at concurrency 8. **600 aggregate output tok/s at concurrency 8** is a stretch
target over complete 32K-input/16K-output replays, subject to numerical,
batch-1 and memory gates. Explicit kernel beam search and separate request
scheduler tuning retain fixed/searched ablations; concurrency-8 performance
has not yet been measured.

## Demonstration protocols

**nanoGPT:** freeze architecture, initialization, tokenizer, dataset revision,
train/validation splits, sequence length, effective batch, optimizer, schedule,
precision and numerical policies. Include a matched PyTorch eager control and
a tuned PyTorch control (for example compiled execution/native attention), with
settings disclosed. Both runners train continuously over the same ordered data
for 1B tokens; synthetic replay remains a separate microbenchmark.

Count actual non-padding training target tokens. Exclude validation, discarded
warmup and replayed work; resumed runs do not count already accepted tokens twice.
Define the final partial batch/masking so the recorded budget is exactly 1B.
Checkpoint weights, optimizer moments, counters, RNG and data position, and verify
resume against an uninterrupted segment. Report loss/validation curves against
tokens, final quality, tokens/s, total wall time including data/checkpoints,
steady-state update time, memory, build/search time, interruptions and recovery.
Run shorter correctness/resume pilots before paying for both complete runs.

**Image:** use the exact
[Qwen-Image-2.1 checkpoint and pipeline](https://huggingface.co/Qwen/Qwen-Image-2.1).
Pin the Diffusers commit supporting `QwenImage21Pipeline`. Match prompts,
resolution, steps, scheduler, guidance, precision, batch, initial latent noise
and conditioning; matching a seed alone does not guarantee identical inputs.
Measure conditioning, denoising, VAE and complete generation separately. Save
images and intermediate checks plus a declared quality evaluation; approximate
floating-point agreement need not imply byte-identical final images. Text-to-image
is the requested first demonstration; editing/RGBA are separately scoped.
Disclose all host work and integration dependencies. Select a resident-memory
profile for the primary comparison; CPU/SSD offloading needs its own later track.
Model terms remain separate from Tensor's MIT license; do not bundle weights by
default with project distributions.

**Audio:** use the model card's
[audio-specific runner](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B-GGUF),
`llama-liquid-audio-cli` or its server, with a pinned source/build. The card links
its runner source to a llama.cpp change; verify the implementation rather than
assuming any generic llama.cpp release executes audio. Pin all GGUF components,
including projector, vocoder and audio tokenizer/speaker files, and match their
quantization. Match input WAVs, sample rate, preprocessing, prompt/task and decode
settings. The requested path is audio-to-audio, not text-only LFM2 generation.
Retain input/output audio and transcripts. Report preprocessing, model and vocoder
costs, end-to-end latency, first-audio latency if supported, real-time factor with
its duration denominator, memory, output duration and declared speech/content
quality checks. A common input corpus and matched output policy prevent shorter
or truncated speech from appearing faster without disclosure.

## Full llama.cpp quantization demonstration

The goal is to demonstrate every available GGUF weight-quantization preset in
a **pinned llama.cpp revision**, rather than only the currently supported Tensor
F16/Q4 profiles. BENCH-09 tracks this as new implementation and evidence work.
It does not promise every llama.cpp model architecture or an unbounded future
list of formats. Freeze the complete inventory when selecting the reference
revision, and refresh it deliberately for later releases.

The [quantizer source](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/quantize.cpp)
provides the preset registry and quantization options. The observed planning
snapshot includes these families; the pinned registry, not this summary, defines
the final exhaustive set:

| Family | Presets in the planning snapshot |
|---|---|
| Basic Q formats | Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0 |
| K presets | Q2_K, Q2_K_S, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K |
| Importance-aware IQ presets | IQ1_S/M, IQ2_XXS/XS/S/M, IQ3_XXS/XS/S/M, IQ4_NL/XS |
| Ternary presets | TQ1_0, TQ2_0 |
| MoE-specific preset | MXFP4_MOE |
| Precision controls | F16, BF16, F32 |

Expand slash notation into individual rows in the actual inventory. Preserve
Q3_K/Q4_K/Q5_K aliases but deduplicate their canonical targets for coverage totals.
Record COPY as a copy control, not a quantization format. Presets can produce
mixed tensor types; an `_M` recipe is not a single block decoder. Inventory the
actual tensor-type histogram and required storage layouts of every generated
GGUF, including higher-precision embeddings/output and other retained tensors.

Use one pinned original high-precision checkpoint per model, with the same
tokenizer and architecture across its quantized variants. Quantize each variant
directly from that source using the pinned tool; avoid chained requantization.
Retain command lines, flags, tensor overrides, calibration/importance-matrix
inputs and hashes, source/output hashes, quantizer build and file sizes. Keep
calibration data separate from held-out quality evaluation. Missing calibration,
incompatible shapes or an inapplicable architecture need explicit dispositions.
Use a representative dense model and an additional compatible MoE model for
architecture-specific presets; select exact checkpoints and device profiles in
BENCH-01 rather than expecting every preset to apply to LFM2 or the FP8 Qwen files.

Run Tensor and llama.cpp against the **identical produced GGUF bytes** for each
case. Verify block decoding against the pinned reference, then embeddings,
projections and applicable routed-expert operations, and finally complete
inference. Distinguish parsing, numerical correctness, full-model execution and
performance as separate coverage columns. A readable GGUF does not establish
usable model support. Packed kernels must also handle all tensor types in the
actual mixture. Disclose any full-weight dequantization or fallback path and its
startup/memory cost; do not label it native packed execution.

Publish two quality comparisons: Tensor versus llama.cpp for each identical
quantized checkpoint, and each quantized variant versus its high-precision
control. Use held-out perplexity/logit checks and a declared task evaluation.
Report quantization/calibration time, file size, effective bits per weight,
load time, peak/resident memory, prefill/decode throughput and latency, quality
delta, and the Tensor search/default ablation where applicable. Effective bits
must state whether metadata and non-weight data are included. Match prompt/context,
batch, token budget, sampler, cache dtype and GPU residency; validate memory and
correctness before timing. Keep weight quantization and KV-cache quantization as
separate axes so a cache change cannot masquerade as a weight-format improvement.

The primary demonstration covers available weight presets and required on-disk
tensor types. Backend-only repacking, deprecated/internal types and cache-only
formats get separate inventory notes rather than being counted as equivalent
weight presets. Native CPU/GPU differences are explicit reference configurations;
automatic CPU/GPU splitting and SSD offloading remain post-1.0 addons. Do not
silently use llama.cpp offloading to complete a resident-GPU comparison.

Aim for complete applicable preset execution in Tensor. A missing Tensor decoder
or kernel remains `not-implemented`, with exact missing tensor types recorded;
it is unfinished coverage, not an accepted demonstration. The release report
lists every preset and its applicable model/device cells, including failures,
OOMs and unsupported reference paths. Broad quantization coverage requires these
new kernels and checks; it is not already supplied by the current GGUF reader.

## Minimalist Strata-inspired hybrid MoE demonstration (post-1.0)

Borrow the architectural recipe from
[Strata](https://github.com/Niko1221/Strata) and its
[design explanation](https://github.com/Niko1221/Strata/blob/main/docs/HOW_IT_WORKS.md):
retain frequently used experts on the GPU, compute remaining selected experts
on the CPU, and merge their weighted outputs. Implement this as an experimental
inference workload using Tensor's providers, rather than porting Strata's engine
or adding a new raw Vulkan backend. Exercise CUDA and native wgpu/Vulkan as
separate device profiles; qualify each claimed profile on physical hardware.

The first deliverable is **one routed MoE block**, not a complete Strata model.
Select a small compatible MoE checkpoint or an explicitly labeled block fixture
in BENCH-01. Existing LFM2 inference supplies reusable packed-projection and
prepared-plan infrastructure, but its dense FFNs do not establish MoE support.
Pin expert shapes, activation function, shared-expert behavior where applicable,
router scoring, top-k selection/normalization and tie handling. Begin with an
independent high-precision reference and one packed weight format; use identical
packed bytes for all execution placements.

Keep the first implementation to one GPU, one sequence and one-token decode
steps, RAM-resident packed weights, a fixed VRAM budget and fixed expert
residency. Choose resident experts from a separate calibration routing trace;
freeze that selection before held-out evaluation. Reserve memory for persistent
activations, router metadata, result buffers and scratch before assigning expert
slots. Use bounded buffer segments and explicit offsets compatible with the
provider's binding limits. GPU hits and CPU misses may initially run sequentially;
merge every selected expert with the model's routing weight before continuing.
An efficient native packed CPU path is new work; a NumPy correctness reference
alone does not establish competitive CPU expert performance.

Validate router selections, per-expert outputs and the complete weighted result,
including shared experts where required. Cover no resident experts, partial
residency, all selected experts resident, selection ties and repeated calls with
changed inputs. Check resource lifetime and reset behavior. Freeze tolerances
before tuning, and disclose CPU/GPU arithmetic differences. Hold weights, inputs
and routing semantics constant across matched all-CPU, hybrid and all-GPU controls;
record an all-GPU OOM explicitly where it cannot fit.

Measure the complete block from host input through host-visible merged output,
including routing, metadata readback, uploads, CPU/GPU expert work, transfers,
merge and synchronization. Report stage costs, completed latency distributions,
RAM/VRAM use, expert hit rate, bytes transferred, startup/upload cost and CPU
thread settings across budget and routing-locality sweeps. Include cold setup
and steady-state runs separately. Fixed-routing kernel replay is a separate
microbenchmark, not evidence of complete hybrid execution. Retain cases where
the split loses; acceptance requires correct execution and evidence of the
conditions under which residency pays off, not a universal speedup.

After the block qualifies, evaluate CPU/GPU overlap and adaptive residency as
separate ablations. Chunked prefill, speculative decoding, KV offloading, SSD
streaming and multi-GPU execution remain later extensions. A complete model
demonstration needs its own checkpoint, architecture, tokenizer/state and quality
gates under the LLM protocol; the block does not establish Strata's GDN/QSA,
quantization or full-model support. Keep BENCH-10 evidence separately labeled
from the required 1.0 program and BENCH-09's resident-placement comparisons.

## Execution sequence and release evidence

1. **Specify:** BENCH-01 defines exact cases, quality gates, reference revisions,
   support dispositions, resource estimates, and the demonstration GPUs/dataset.
2. **Enable:** BENCH-02/03/05/06/07/09 implement the required kernels and complete
   pipelines. Qualify small pilots and memory before long or broad runs. GDN and
   FP8 work feed Qwen; image/audio closure and sustained training have separate
   dependencies. BENCH-09 adds the full quantization inventory and packed kernels;
   BENCH-04's shared engine and LFM2 batching can develop before Qwen is complete;
   its Qwen comparisons follow a correct Qwen runner in that same engine.
3. **Qualify:** validate the physical A10G/L4/L40S/H100/B200 kernel profiles and
   L40S/H100 model profiles using exact candidate artifacts. Reuse M4 packaging,
   installed-path and compiler-free consumer checks where applicable. Benchmark
   coverage does not automatically promote all providers/platforms to stable.
4. **Measure:** freeze search winners, repeat comparisons independently, run both
   1B-token trainings, and produce the matched image/audio and quantization demonstrations.
5. **Publish evidence:** BENCH-08 supplies the coverage table, raw reports,
   manifests/hashes, scripts/commands, outputs, methodological limits and results
   tied to the accepted candidate. Keep baselines reproducible in separate
   environments; include their versions and dependency locks.
6. **Follow-on composition:** BENCH-10 qualifies the minimal hybrid MoE block
   after 1.0, then uses its correctness and transfer/latency evidence to scope
   an end-to-end MoE workload and additional scheduling features.

For the requested 1.0 program, a successful Tensor demonstration requires a
complete correct run; an unimplemented Tensor path is unfinished work. Unavailable
hardware or unsupported upstream cells remain explicit gaps, not successes.
Retain these targets when planning the release and make any scope revision
explicit. Core contract acceptance in [M1–M5](v1.md) remains necessary in addition
to this program. No benchmark has been executed, GPU time booked, model downloaded,
or release version changed by this plan.
