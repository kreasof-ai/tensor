# Tensor readiness for LLT

Date: 2026-10-08. This is a requirements and acceptance checklist, not a claim
that the listed features have shipped. [Tensor](https://github.com/kreasof-ai/tensor) baseline:
`caf0118d870739b176a88f487ae9e54de1d5d214`; hardware: NVIDIA L40S (`sm_89`).

The development order is **Tensor readiness, then LLT completion**. First finish
the CUDA kernel, interoperability, and training capabilities that LLT needs.
Use small LLT fixtures to validate Tensor during that work; resume architecture,
quality, and scaling studies after the Tensor gate passes.

Tensor's broader [1.0 plan](v1.md) has separate release,
platform, and workload milestones. This checklist defines the LLT dependency
gate; it does not declare Tensor 1.0 complete or make unrelated image/audio
workloads prerequisites for LLT. PyTorch can continue to orchestrate execution,
autograd, and experiments. Performance-critical LLT kernels should execute
through Tensor, with any remaining PyTorch operations explicitly reported.

## Evidence and status

The implemented LLT dependency profile is qualified on NVIDIA L40S `sm_89`.
The [Tensor qualification report](../research/llt-readiness.md) links raw numerical,
training, memory, timing, packaging and provenance evidence. The explicit
`tensor_torch.llt` API supplies first-order FP16/BF16 attention, model backward,
full-vocabulary loss/update, RoPE, and persistent batched decode caches. PyTorch
continues to own storage, layout/control, gradient accumulation and checkpoint
scheduling; this boundary is explicit in the report.

The original [LLT L40S report](https://github.com/kreasof-ai/loop-latent-transformer/blob/main/benchmarks/L40S.md)
and [follow-ups](../research/llt-l40s-followups.md) retain the starting gaps.
The pinned TileLang reduced-fragment limitation remains, using the qualified
shared-memory extraction workaround. Other SMs, dropout, general masks,
higher-order gradients, paged serving and distributed training are not covered.
Shape-specific timing regressions are published under the policy below. Passing
this gate does not establish LLT's quality or architecture performance claims.

## A. Tensor requirements, in implementation order

| ID | Requirement | Current evidence | Completion criterion |
|---|---|---|---|
| T01 | BF16 end to end | Passed: ABI 1.3 ID 13, capability, storage and DLPack | BF16 survives runtime, compiler, artifacts, Torch binding, and numerical tests |
| T02 | Reusable attention forward | Passed: reusable FP16/BF16 MHA/shared-KV kernels | Supported BF16/FP16 naive and shared-latent prefill/decode profiles |
| T03 | Memory-efficient attention backward | Passed: recomputed tile gradients and deterministic shared-head sums | Correct dQ and accumulated shared-cache gradients without quadratic score storage |
| T04 | Decoupled positional attention | Passed: independent score/value dimensions, runtime RoPE and offsets | Separate positional score component, latent values, and backward validated |
| T05 | LLT training operator coverage | Passed: model operators and every-parameter gradient fixtures | Required BF16 operators, FP32 reductions, and all parameter gradients validated |
| T06 | Loss and optimizer path | Passed: 50,257-class loss, chunking, FP32 state and resume | Full-vocabulary training with measured memory, correct updates, and resumable state |
| T07 | Autograd and checkpoint integration | Passed: workload autograd, exact checkpoints and coverage records | Registered backward, exact recomputation, and observable kernel coverage |
| T08 | Efficient execution and cache lifecycle | Passed: native prepared calls, graphs and persistent variable-length caches | Prepared/native execution, safe cache updates, eager and graph benchmarks |
| T09 | Compiler, packaging, compatibility | Passed: pinned workaround, ABI compatibility and compiler-free wheels | Regressions, clear diagnostics, compatible artifacts, clean consumer validation |
| T10 | Sustained end-to-end qualification | Passed: 1,000-step matched fixtures, multi-token decode and resource sweep | Tensor-backed LLT and naive fixtures pass training, decode, numerical, and resource gates |

### T01 — BF16 storage and interoperability

- [x] Add BF16 consistently to ABI descriptors, dtype validation, manifests,
  DLPack import/export, Torch bindings, and compiler lowering.
- [x] Preserve existing numeric dtype identities and old FP16 artifacts. Declare
  any new runtime capability explicitly; older consumers must reject it clearly.
- [x] Test owned and borrowed CUDA buffers, round trips, prepared bindings, and
  unsupported layouts. Test rounding, tails, finite extremes, NaNs, and infinities.
- [x] Specify mixed precision: BF16 compute/storage where appropriate; FP32
  accumulation, normalization/softmax reductions, master weights, and optimizer
  state. Validate actual kernel behavior rather than relying on input dtype.

BF16 is implemented and qualified in Tensor. IDs 1–12 and descriptor layouts
remain unchanged; older consumers reject new BF16 capability requirements.

### T02 — Attention forward for both LLT and fair baselines

- [x] Promote the local online-softmax prefill and partitioned decode kernels
  into reusable Tensor exports/profiles, with BF16 and FP16 coverage.
- [x] Support ordinary MHA baselines and shared-KV/absorbed-latent attention;
  include the layerwise-cache control. Do not expand shared KV across heads or
  materialize a global sequence-by-sequence score matrix.
- [x] Validate batches 1/4/8, ranks 32/64/96/128, odd lengths, short and long
  contexts, partition merges, and K/V aliasing. Define supported layouts and
  alignment explicitly; reject or report unsupported cases.
- [x] Pass an explicit attention scale: folded LLT uses the original head
  dimension, not automatically the latent rank. Specify causal alignment when
  query and KV lengths differ, including offset queries during chunked prefill.
- [x] Compare complete calls with the matching Torch Flash baseline, including
  cache writes and required layout conversions. Publish regressions by shape.

### T03 — Attention backward and shared-state gradients

- [x] Implement memory-efficient dQ/dK/dV with FP32 reductions and bounded saved
  state, such as softmax statistics plus recomputation.
- [x] For an aliased latent C used as both K and V, accumulate
  `dC = sum_heads(dK + dV)`. Also accumulate every layer and loop that reads C;
  avoid both missing contributions and double counting.
- [x] Validate masking, tails, shared heads, large logits, and gradients against
  an unfolded high-precision reference and Torch. Check all parameter gradients,
  not just attention outputs.
- [x] Measure saved tensors, temporary workspace, full backward latency, and
  allocator peaks. No quadratic attention matrix may be hidden in backward.

### T04 — Positional attention compatible with absorption

- [x] Add the decoupled RoPE score path needed for an MLA-style positional
  variant. A query/key score dimension can differ from the value dimension:
  the score includes latent and positional terms, while values remain latent.
- [x] Support positional-key storage, offsets, chunked prefill, cached decoding,
  rotary forward/backward, and gradient accumulation into positional projections.
- [x] Verify the unfolded/folded equivalence. Rotating the compressed latent
  blindly is not a justified replacement for decoupled RoPE.

Learned absolute positions can serve the first training fixture. The positional
kernel path must pass before declaring the intended RoPE LLT variant supported;
model quality and length generalization remain LLT experiments.

### T05 — Remaining model kernels and differentiation

- [x] Extend and qualify GEMMs for projection, MLP, classifier, dX, and dW,
  including transpose/batched forms, tails, and BF16/FP32 accumulation.
- [x] Provide RMSNorm forward/backward with stable FP32 reductions. Existing NN
  normalization and LLM inference RMS kernels are useful starting points, not
  evidence of complete BF16 RMSNorm training coverage.
- [x] Qualify residual add, GELU forward/backward, embedding lookup and scatter
  gradients, head packing, casts, and reductions for LLT shapes. SwiGLU is an
  additional requirement only if that architectural variant is selected.
- [x] Differentiate folded projection construction correctly. Accumulate tied
  weights across loops; update them once per optimizer step. Refresh prepared
  inference folds after weight updates or checkpoint loading.

### T06 — Loss, optimization, and the training memory floor

- [x] Validate numerically stable full-vocabulary cross-entropy and its backward,
  including padding/ignored labels if used. Profile a roughly 50K vocabulary.
- [x] Evaluate chunked or fused classifier/loss execution if logits dominate
  memory; measure its recomputation cost. Count classifier and embedding weights,
  their gradients, master weights, and optimizer state in every comparison.
- [x] Qualify gradient zeroing/accumulation, clipping, finite checks, AdamW,
  schedule application, and mixed-precision state against Torch updates.
- [x] Save and restore model, optimizer, scheduler, random state, and data position.

Existing Tensor NN kernels cover parts of this path. Extend those bounded
templates where useful. A standalone general-purpose autograd framework is not
required: registered workload-specific backward formulas and PyTorch control
code are sufficient if the execution coverage is explicit.

### T07 — Training integration and exact checkpointing

- [x] Register backward formulas for required custom operators, with correct
  saved-tensor ownership, mutation/alias contracts, and output metadata.
- [x] Support exact loop checkpoint recomputation; compare gradients and updates
  with the uncheckpointed version. Handle randomness if stochastic layers are used.
- [x] Publish forward and backward operator coverage, graph breaks, and semantic
  fallbacks. Calling a Tensor compiler backend is not proof that every op ran in Tensor.
- [x] Preserve current-stream ordering and allocation lifetimes through backward
  and checkpoint replay; validate auxiliary streams when used.

### T08 — Submission overhead and persistent caches

- [x] Use prepared calls/native execution for repeated attention and model
  regions. Measure ordinary eager calls separately from CUDA graph replay.
- [x] Support preallocated append slots, capacity checks, reset/reuse, prefix
  lengths, and chunked prefill. Keep allocations and host synchronizations out
  of repeated decode where the supported execution mode permits it.
- [x] Validate graph warmup, capture, replay, pointer lifetimes, cache mutation,
  and bounded shape buckets. Test complete multi-token generation.
- [x] Add variable-length batched cache handling before claiming that serving
  mode. Paged KV, speculative decoding, and a general serving scheduler are later
  extensions, not prerequisites for the initial single-GPU LLT study.

The measured eager adapter tax is a concrete optimization target. Graph-only
improvements cannot be advertised as ordinary eager performance improvements.

### T09 — Compiler and distribution reliability

- [x] Retain the reduced-fragment lowering reproducer and test the shared-memory
  extraction workaround. Improve diagnostics or qualify an upstream frontend
  fix; current evidence identifies a compiler dependency limitation.
- [x] Pin the compiler/NVRTC stack and select schedules only after numerical and
  resource checks. Store exact shape, dtype, scale, mask, and target requirements.
- [x] Build and validate `sm_89` artifacts, source/binary hashes, cold/cache-hit
  behavior, old FP16 compatibility, and new BF16 capability rejection.
- [x] Validate an installed consumer outside the source tree without compiler
  dependencies. Document the tested Torch/native-executor pairing and fallbacks.

### T10 — Tensor exit gate before LLT research resumes

- [x] Required FP16/BF16 forward and backward profiles pass output, gradient,
  aliasing, stream, and checkpoint tests on the L40S.
- [x] A small LLT and a matched naive model run at least 1,000 training steps
  through the declared Tensor kernel coverage. Compare updates and loss behavior
  with Torch using the same initialization, batches, and optimizer policy.
- [x] Predeclare precision-specific numerical tolerances. Require finite losses,
  no unexplained drift or accumulating leak, and successful checkpoint/resume.
- [x] Real-prefix prefill followed by repeated cached decode matches full causal
  inference, including cache growth, tails, and supported positional modes.
- [x] Record full-step latency and peak CUDA allocations across ranks, contexts,
  and loop counts. Separate eager/graph, cold/warm, and kernel/full-call timing;
  disclose non-allocator device memory and every fallback.
- [x] Publish reproducible commands, exact Tensor revision/environment,
  operator coverage, raw results, accepted regressions, and capability limits.

Numerical tolerances and acceptable performance regressions must be chosen before
qualification runs. Tensor readiness requires reliable, measured implementations;
it does not require LLT to win every shape or prove its architecture claims.

## Working sequence

1. T01: BF16 and compatibility tests.
2. T02–T04: attention forward, backward, and positional paths.
3. T05–T07: model kernels, loss/update, and checkpoint integration.
4. T08–T09: execution overhead, caches, compiler, and packaging evidence.
5. Pass T10 on the L40S and pin the accepted Tensor revision.
6. Resume the [LLT research roadmap](https://github.com/kreasof-ai/loop-latent-transformer/blob/main/research/ROADMAP.md).

This dependency gate is complete for the declared profile. It does not establish LLT's open
architecture, quality, total training-memory, or communication claims.

## Qualification performance policy (fixed before final runs)

The dependency gate accepts shape-specific latency regressions if numerical
checks, sustained training, bounded caches, and linear attention saved/workspace
storage pass. It requires publication of every measured eager and graph
regression against Torch SDPA and full-model controls, with at least nine samples;
it does not require a speedup on every shape. Eager host overhead remains part
of the measurement. The final sweep runs one GPU qualification process at a time.
Model architecture targets (50% total-memory reduction and 20% latency reduction)
remain LLT research targets and are not relaxed by this Tensor gate.
