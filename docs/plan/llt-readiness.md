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

The [L40S report](https://github.com/kreasof-ai/loop-latent-transformer/blob/main/benchmarks/L40S.md) demonstrates FP16 Tensor forward
attention, prefill, and cached decode. Training uses PyTorch BF16 Flash attention
and backward. Global latent cache size is independent of loop count, but total
training memory is not. None of the 20 GPU training configurations met the
combined memory/latency target. There is no completed trained-quality study.

The [existing follow-ups](../research/llt-l40s-followups.md) contain BF16 and
compiler reproducers. Tensor already has CUDA execution, DLPack, artifacts,
stream handling, prepared calls, and bounded manually differentiated NN kernels.
These are foundations to extend and qualify, not missing systems to rebuild.

## A. Tensor requirements, in implementation order

| ID | Requirement | Current evidence | Completion criterion |
|---|---|---|---|
| T01 | BF16 end to end | Missing ABI/DLPack dtype; Torch BF16 import fails | BF16 survives runtime, compiler, artifacts, Torch binding, and numerical tests |
| T02 | Reusable attention forward | Custom LLT FP16 kernels work; generic adapter coverage is narrower | Supported BF16/FP16 naive and shared-latent prefill/decode profiles |
| T03 | Memory-efficient attention backward | Optimized shared-latent backward missing | Correct dQ and accumulated shared-cache gradients without quadratic score storage |
| T04 | Decoupled positional attention | Current LLT uses learned absolute positions; RoPE untested | Separate positional score component, latent values, and backward validated |
| T05 | LLT training operator coverage | GEMM, embedding, GELU, norm and other bounded NN kernels exist | Required BF16 operators, FP32 reductions, and all parameter gradients validated |
| T06 | Loss and optimizer path | Bounded CE, clipping, AdamW kernels exist | Full-vocabulary training with measured memory, correct updates, and resumable state |
| T07 | Autograd and checkpoint integration | Torch training coverage experimental; custom operators lack automatic backward | Registered backward, exact recomputation, and observable kernel coverage |
| T08 | Efficient execution and cache lifecycle | Graph decode works; ordinary calls have significant host overhead | Prepared/native execution, safe cache updates, eager and graph benchmarks |
| T09 | Compiler, packaging, compatibility | Local L40S builds work; reduced-fragment lowering limitation has a workaround | Regressions, clear diagnostics, compatible artifacts, clean consumer validation |
| T10 | Sustained end-to-end qualification | Synthetic forward and PyTorch training studies completed | Tensor-backed LLT and naive fixtures pass training, decode, numerical, and resource gates |

### T01 — BF16 storage and interoperability

- [ ] Add BF16 consistently to ABI descriptors, dtype validation, manifests,
  DLPack import/export, Torch bindings, and compiler lowering.
- [ ] Preserve existing numeric dtype identities and old FP16 artifacts. Declare
  any new runtime capability explicitly; older consumers must reject it clearly.
- [ ] Test owned and borrowed CUDA buffers, round trips, prepared bindings, and
  unsupported layouts. Test rounding, tails, finite extremes, NaNs, and infinities.
- [ ] Specify mixed precision: BF16 compute/storage where appropriate; FP32
  accumulation, normalization/softmax reductions, master weights, and optimizer
  state. Validate actual kernel behavior rather than relying on input dtype.

This is the first implementation task. BF16 already works on the L40S through
PyTorch; the missing feature is Tensor's interoperability and execution path.

### T02 — Attention forward for both LLT and fair baselines

- [ ] Promote the local online-softmax prefill and partitioned decode kernels
  into reusable Tensor exports/profiles, with BF16 and FP16 coverage.
- [ ] Support ordinary MHA baselines and shared-KV/absorbed-latent attention;
  include the layerwise-cache control. Do not expand shared KV across heads or
  materialize a global sequence-by-sequence score matrix.
- [ ] Validate batches 1/4/8, ranks 32/64/96/128, odd lengths, short and long
  contexts, partition merges, and K/V aliasing. Define supported layouts and
  alignment explicitly; reject or report unsupported cases.
- [ ] Pass an explicit attention scale: folded LLT uses the original head
  dimension, not automatically the latent rank. Specify causal alignment when
  query and KV lengths differ, including offset queries during chunked prefill.
- [ ] Compare complete calls with the matching Torch Flash baseline, including
  cache writes and required layout conversions. Publish regressions by shape.

### T03 — Attention backward and shared-state gradients

- [ ] Implement memory-efficient dQ/dK/dV with FP32 reductions and bounded saved
  state, such as softmax statistics plus recomputation.
- [ ] For an aliased latent C used as both K and V, accumulate
  `dC = sum_heads(dK + dV)`. Also accumulate every layer and loop that reads C;
  avoid both missing contributions and double counting.
- [ ] Validate masking, tails, shared heads, large logits, and gradients against
  an unfolded high-precision reference and Torch. Check all parameter gradients,
  not just attention outputs.
- [ ] Measure saved tensors, temporary workspace, full backward latency, and
  allocator peaks. No quadratic attention matrix may be hidden in backward.

### T04 — Positional attention compatible with absorption

- [ ] Add the decoupled RoPE score path needed for an MLA-style positional
  variant. A query/key score dimension can differ from the value dimension:
  the score includes latent and positional terms, while values remain latent.
- [ ] Support positional-key storage, offsets, chunked prefill, cached decoding,
  rotary forward/backward, and gradient accumulation into positional projections.
- [ ] Verify the unfolded/folded equivalence. Rotating the compressed latent
  blindly is not a justified replacement for decoupled RoPE.

Learned absolute positions can serve the first training fixture. The positional
kernel path must pass before declaring the intended RoPE LLT variant supported;
model quality and length generalization remain LLT experiments.

### T05 — Remaining model kernels and differentiation

- [ ] Extend and qualify GEMMs for projection, MLP, classifier, dX, and dW,
  including transpose/batched forms, tails, and BF16/FP32 accumulation.
- [ ] Provide RMSNorm forward/backward with stable FP32 reductions. Existing NN
  normalization and LLM inference RMS kernels are useful starting points, not
  evidence of complete BF16 RMSNorm training coverage.
- [ ] Qualify residual add, GELU forward/backward, embedding lookup and scatter
  gradients, head packing, casts, and reductions for LLT shapes. SwiGLU is an
  additional requirement only if that architectural variant is selected.
- [ ] Differentiate folded projection construction correctly. Accumulate tied
  weights across loops; update them once per optimizer step. Refresh prepared
  inference folds after weight updates or checkpoint loading.

### T06 — Loss, optimization, and the training memory floor

- [ ] Validate numerically stable full-vocabulary cross-entropy and its backward,
  including padding/ignored labels if used. Profile a roughly 50K vocabulary.
- [ ] Evaluate chunked or fused classifier/loss execution if logits dominate
  memory; measure its recomputation cost. Count classifier and embedding weights,
  their gradients, master weights, and optimizer state in every comparison.
- [ ] Qualify gradient zeroing/accumulation, clipping, finite checks, AdamW,
  schedule application, and mixed-precision state against Torch updates.
- [ ] Save and restore model, optimizer, scheduler, random state, and data position.

Existing Tensor NN kernels cover parts of this path. Extend those bounded
templates where useful. A standalone general-purpose autograd framework is not
required: registered workload-specific backward formulas and PyTorch control
code are sufficient if the execution coverage is explicit.

### T07 — Training integration and exact checkpointing

- [ ] Register backward formulas for required custom operators, with correct
  saved-tensor ownership, mutation/alias contracts, and output metadata.
- [ ] Support exact loop checkpoint recomputation; compare gradients and updates
  with the uncheckpointed version. Handle randomness if stochastic layers are used.
- [ ] Publish forward and backward operator coverage, graph breaks, and semantic
  fallbacks. Calling a Tensor compiler backend is not proof that every op ran in Tensor.
- [ ] Preserve current-stream ordering and allocation lifetimes through backward
  and checkpoint replay; validate auxiliary streams when used.

### T08 — Submission overhead and persistent caches

- [ ] Use prepared calls/native execution for repeated attention and model
  regions. Measure ordinary eager calls separately from CUDA graph replay.
- [ ] Support preallocated append slots, capacity checks, reset/reuse, prefix
  lengths, and chunked prefill. Keep allocations and host synchronizations out
  of repeated decode where the supported execution mode permits it.
- [ ] Validate graph warmup, capture, replay, pointer lifetimes, cache mutation,
  and bounded shape buckets. Test complete multi-token generation.
- [ ] Add variable-length batched cache handling before claiming that serving
  mode. Paged KV, speculative decoding, and a general serving scheduler are later
  extensions, not prerequisites for the initial single-GPU LLT study.

The measured eager adapter tax is a concrete optimization target. Graph-only
improvements cannot be advertised as ordinary eager performance improvements.

### T09 — Compiler and distribution reliability

- [ ] Retain the reduced-fragment lowering reproducer and test the shared-memory
  extraction workaround. Improve diagnostics or qualify an upstream frontend
  fix; current evidence identifies a compiler dependency limitation.
- [ ] Pin the compiler/NVRTC stack and select schedules only after numerical and
  resource checks. Store exact shape, dtype, scale, mask, and target requirements.
- [ ] Build and validate `sm_89` artifacts, source/binary hashes, cold/cache-hit
  behavior, old FP16 compatibility, and new BF16 capability rejection.
- [ ] Validate an installed consumer outside the source tree without compiler
  dependencies. Document the tested Torch/native-executor pairing and fallbacks.

### T10 — Tensor exit gate before LLT research resumes

- [ ] Required FP16/BF16 forward and backward profiles pass output, gradient,
  aliasing, stream, and checkpoint tests on the L40S.
- [ ] A small LLT and a matched naive model run at least 1,000 training steps
  through the declared Tensor kernel coverage. Compare updates and loss behavior
  with Torch using the same initialization, batches, and optimizer policy.
- [ ] Predeclare precision-specific numerical tolerances. Require finite losses,
  no unexplained drift or accumulating leak, and successful checkpoint/resume.
- [ ] Real-prefix prefill followed by repeated cached decode matches full causal
  inference, including cache growth, tails, and supported positional modes.
- [ ] Record full-step latency and peak CUDA allocations across ranks, contexts,
  and loop counts. Separate eager/graph, cold/warm, and kernel/full-call timing;
  disclose non-allocator device memory and every fallback.
- [ ] Publish reproducible commands, exact Tensor revision/environment,
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

This is planned work. The Tensor dependency gate does not establish LLT's open
architecture, quality, total training-memory, or communication claims.
