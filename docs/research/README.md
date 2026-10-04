# Research and benchmarks

[Documentation](../README.md) · [Roadmap](../plan/roadmap.md)

Reports retain the measured environment, workload, correctness checks, timing
protocol, and reproduction commands. Raw JSON, CSV, and figures live in
[data/](data/). Results apply to the recorded profiles and revisions.

For the RX 6700 XT Vulkan campaign, start with the
[kernel and inference improvement history](rx6700xt-vulkan-history.md):
the first hardware attempt through 1K 2.6B prefill, with 23 inference milestones,
kernel and throughput plots, source hashes and an archive commit timeline.

## Inference and training workloads

| Report | What to read it for |
|---|---|
| [F16, Q4_0 and Q4_K_M CUDA optimization](lfm2-cuda-formats.md) | Tuned prefill tiles and packed loaders, Q4_K decode, F16 FFN fusion and shared long-context K/V; matched frozen-before and llama.cpp controls |
| [Vulkan inference optimizations transferred to CUDA](lfm2-cuda-vulkan-transfer.md) | Packed FP32 decode, fusion, split-KV attention and guarded suffix prefill on A10G; matched QAD checkpoint and b11310 llama.cpp baseline |
| [2.6B QAD Q4_0 1K prefill experiment](lfm2-2.6b-prefill-1k.md) | 1,058/1,047 tok/s prefill, twice the packed control; mixed arithmetic, shared layout search, guarded suffix liveness and unchanged decoding |
| [2.6B QAD Q4_0 parity search](lfm2-2.6b-q4_0-parity.md) | 164 tok/s decode versus native 170, a further 51% packed prefill gain, corrected replay controls and the remaining 3–5% decode gap |
| [2.6B QAD Q4_0 revisit](lfm2-2.6b-q4_0-revisit.md) | 146 tok/s decode versus native 169–170, a controlled 5% runtime gain, 48–49% larger-chunk prefill gains and the remaining quantized projection gap |
| [230M adaptive prefill and the CLBlast transfer](lfm2-prefill-chase.md) | 16% short-prompt and 2.19× long-prompt prefill gains, mixed-precision outer products, chunk ablations and tuned llama.cpp controls; decoding unchanged |
| [230M runtime overhead and wider discovery](lfm2-230m-runtime-search.md) | 10.5–12.0% decode gain from ordered readback and fewer submissions; 601 more projection candidates, experimental attention fusion, and a 1.09–1.13× remaining native gap |
| [230M F16 decode search](lfm2-230m-decode-search.md) | 1,081 streamed-weight projection candidates, paired FFN and parallel attention; 39–63% decode gains and a 1.23× remaining llama.cpp gap |
| [Searched prefill and decoding throughput](lfm2-prefill-search-throughput.md) | Full-model F16 rates for Tensor, tinygrad and llama.cpp; 45–47% Tensor prefill gains, 19–28% tinygrad gains, unchanged decoding and explicit accuracy metrics |
| [Tensor search versus tinygrad and llama.cpp](lfm2-tensor-search-comparison.md) | Tensor's own 30-minute Vulkan schedule search, 2.85×/5.93× projection GPU gains, and fresh precision-checked three-framework kernel/call measurements |
| [Wider tinygrad projection search](lfm2-tinygrad-long-search.md) | A combined 30-minute cap, per-candidate oracle checks and independent GPU/completed-call remeasurement of wider beams |
| [LFM2 and tinygrad schedule search](lfm2-tinygrad-comparison.md) | Matched 230M F16 inference, OpenCL/Vulkan precision checks, bounded BEAM projection search and a producer-side search design |
| [Further LFM2 WebGPU decode optimization](lfm2-webgpu-decode-push.md) | Packed floating dots, measured row widths and residual fusion: +9.3% 2.6B decode, 1.21x gap to llama.cpp, smaller-model regression checks |
| [LFM2.5-2.6B Q4_0 matched run](lfm2-2.6b-q4_0-matched-run.md) | The 230M submission protocol re-run on a 2.6B QAD checkpoint, plus two schedule corrections: decode reduction width and prefill staging cost, worth +8.1% decode and +19.0% prefill |
| [LFM2.5-230M native submission](lfm2-230m-native-submission.md) | Optional native prepared-plan encoder, four-wide prefill dots, Python fallback and further Vulkan gains |
| [LFM2.5-230M WebGPU compiler optimization](lfm2-230m-webgpu-compiler-optimization.md) | Register microtiles, parallel reductions, subgroup decode/attention, tuned prefill and matched Vulkan benchmarks |
| [LFM2.5-230M Vulkan optimization](lfm2-230m-vulkan-optimization.md) | Timestamp profiling, packed decode, register-tiled prefill, fusion and GPU greedy generation with fresh before/after measurements |
| [LFM2.5-230M on Radeon Vulkan](lfm2-230m-vulkan.md) | Portable wgpu F16/Q4_0 generation, independent numerical validation and matched native llama.cpp Vulkan timings on RX 6700 XT |
| [LFM2 decode optimization](lfm2-decode-optimization.md) | Latest experimental split-KV attention and packed-weight loading, ablations, and matched llama.cpp CUDA results through 8K context |
| [LFM2 packed FP16 decode](lfm2-fp16-decode.md) | Half2 arithmetic, numerical validation, and alternatives to unpacking all weights |
| [Standalone LFM2 GGUF inference](lfm2-inference.md) | Default engine, model/tokenizer contract, F16/Q4_0/Q4_K_M correctness, and original llama.cpp comparison |
| [Standalone nanoGPT training](phase6-nanogpt.md) | Ten updates of the 124M model, manual backward, bounded autotuning, and the Torch training baseline |

The earlier experimental CUDA LFM2 variants live in the benchmark harness;
the Vulkan transfer is an opt-in `optimized` profile in the ordinary runner.
The WebGPU reports describe changes to the ordinary `tensor_llm.LFM2` runner.
Each report records its supported profile, implementation and precision contract.

## Kernel performance and framework overhead

| Report | Comparison |
|---|---|
| [Staged outer-product WebGPU GEMM](webgpu-outer-product-gemm.md) | Fresh 3.37× large-FP32 gain, a roughly 2× remaining CLBlast gap, opt-in unroll lowering, coupled discovery and separate skinny-shape validation |
| [Whole-loop WebGPU GEMM accumulators](webgpu-gemm-accumulation.md) | About 1.6× gains on large GEMMs, unchanged tile sizes and bitwise outputs; conservative depth selection avoids shorter-K regressions |
| [CLBlast on RX 6700 XT](clblast-rx6700xt.md) | Matched FP32 OpenCL GEMM versus generated WebGPU kernels, full linear epilogues, bounded tuning and precision-gated FP16 observations |
| [Latency scaling](latency-scaling.md) | Matching pointwise, GEMM, and attention shapes across Tensor, Torch, native TileLang, Triton, and WebGPU; includes A10G, RX 6700 XT and a fresh compiler-optimization repeat |
| [Direct backend comparison](direct-backend-comparison.md) | Native TileLang/Triton versus Tensor, separating GPU execution, prepared submission, allocating calls, and `torch.compile` |
| [C++ executor](native-executor.md) | Native PyTorch allocation/submission and its effect on wrapper overhead |
| [FlashAttention demonstration](flash-attention-demo.md) | NVRTC-produced forward attention versus forced Torch FlashAttention SDPA |

## Provider validation

- [WebGPU implementation](phase5-webgpu.md): portable lowering, correctness,
  and the initial validation boundary.
- [Windows RX 6700 XT](webgpu-rx6700xt.md): physical Vulkan execution and
  backend-specific feature limitations.
- [Linux-to-Windows AMD transfer](webgpu-rx6700xt-transfer.md): compiler-free
  cross-host acceptance with retained suite and consumer evidence.

## Accepted milestones

These reports record each phase's accepted scope. Counts and environments are
historical snapshots; current instructions are in the [guides](../README.md#guides).

| Milestone | Evidence |
|---|---|
| Phase 0: architecture validation | [Final exit](e16-phase0-exit.md), [remaining validation](e15-phase0-validation.md) |
| Phase 1: CUDA CLI | [Exit report](phase1-exit.md), [initial CLI baseline](phase1-cli-baseline.md) |
| Phase 2: runtime ABI and NVRTC | [Exit report](phase2-exit.md), [implementation validation](phase2-validation.md) |
| Phase 3: modules and registry transport | [Offline module exit](phase3-exit.md), [PyPI transport](phase3-pypi.md) |
| Phase 4: PyTorch inference integration | [Exit report](phase4-exit.md) |
| Phase 5: portable WebGPU inference | [Implementation](phase5-webgpu.md), [physical cross-host acceptance](webgpu-rx6700xt-transfer.md) |
| Phase 6: standalone training | [nanoGPT acceptance](phase6-nanogpt.md) |
| Repository cleanup | [Package/source reorganization](reorganization.md), [canonical imports](canonical-layout.md) |

## Foundational experiments

These documents explain the original design constraints. Several observations
were superseded by later implementation and validation; use their recorded
revision when reproducing them.

- [Ground truth](phase0-ground-truth.md): initial compiler, import, and packaging probes.
- [Ecosystem survey](ecosystem-and-precedents.md): compiler/runtime and packaging precedents.
- [Artifact shape](e4-artifact-shape.md): frontend IR serialization and retargeting.
- [Artifact versioning](e4a-artifact-versioning.md): compatibility checks and metadata.
- [Symbolic shapes](e4b-symbolic-shapes.md): runtime dimensions versus rebuilds.
- [Opaque artifact prototype](e13-opaque-artifact-prototype.md): the executable boundary.
- [CUDA execution](e14-cuda-execution.md): first compiler-free GPU consumer and startup measurements.
