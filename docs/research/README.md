# Research and benchmarks

[Documentation](../README.md) · [Roadmap](../plan/roadmap.md)

Reports retain the measured environment, workload, correctness checks, timing
protocol, and reproduction commands. Raw JSON, CSV, and figures live in
[data/](data/). Results apply to the recorded profiles and revisions.

## Inference and training workloads

| Report | What to read it for |
|---|---|
| [LFM2 decode optimization](lfm2-decode-optimization.md) | Latest experimental split-KV attention and packed-weight loading, ablations, and matched llama.cpp CUDA results through 8K context |
| [LFM2 packed FP16 decode](lfm2-fp16-decode.md) | Half2 arithmetic, numerical validation, and alternatives to unpacking all weights |
| [Standalone LFM2 GGUF inference](lfm2-inference.md) | Default engine, model/tokenizer contract, F16/Q4_0/Q4_K_M correctness, and original llama.cpp comparison |
| [Standalone nanoGPT training](phase6-nanogpt.md) | Ten updates of the 124M model, manual backward, bounded autotuning, and the Torch training baseline |

The optimized LFM2 variants live in the benchmark harness. Their results do not
describe the default `tensor_llm.LFM2` runner. The original inference report
establishes that runner's behavior and precision contract.

## Kernel performance and framework overhead

| Report | Comparison |
|---|---|
| [Latency scaling](latency-scaling.md) | Matching pointwise, GEMM, and attention shapes across Tensor, Torch, native TileLang, Triton, and WebGPU; includes A10G and RX 6700 XT measurements |
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
