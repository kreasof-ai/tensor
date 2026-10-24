# Roadmap and validation plans

[Documentation](../README.md) · [Research evidence](../research/README.md)

The [path to 1.0](v1.md) proposes the next development milestones, stability
boundary, and release acceptance criteria. Start there for future work.

The [dynamic shapes and batch invariance plan](dynamic-shapes-and-batch-invariance.md)
extends existing symbolic dimensions into performance-oriented AOT batch/token
execution for AKAS, with compiler-visible shape ranges, tuned region portfolios,
packed prefill and an optional qualified numerical policy. It separates Tensor
contracts from engine changes and includes performance gates from dynamic decode.

The [compilation latency and benchmark turnaround plan](compilation-latency.md)
treats cold builds, warm/incremental reuse, validation memory and time to the first
benchmark sample as product acceptance criteria, with profiling before attribution.

The [compiler design literature review](../studies/compiler-design/reports/literature-review.md)
maps foundational and dynamic-compiler research to these plans, with experiments
for shared search, shape assumptions, ragged metadata and incremental builds.

The [LLT readiness plan](llt-readiness.md) lists the CUDA kernel, BF16,
interoperability, and training gates required before the LLT study resumes.

The [1.0 benchmark and demonstration program](v1-benchmarks.md) specifies the
requested models, operators, GPUs, baselines, implementation prerequisites and
measurement rules. It is planned work, separate from retained experimental results.

The [unified Tensor LLM engine plan](unified-llm-engine.md) stages an extension
of `tensor-llm` for both batch-1 latency and batch throughput, sharing model
resources and request execution across local generation and serving.

The proposed [tuning workbench](tuning-workbench.md) covers shared profiling
reports, visual/text interfaces, AI kernel refinement and recipe-only reconstruction.

The [milestone history](roadmap.md) records the agreed Phase 0–6 scopes and their
accepted evidence. Those phases are complete within measured profiles;
standalone LFM2 inference and its experiments extend them as concrete workloads.
Completion of those experiments does not by itself establish 1.0 readiness.

## Workload contract

- [Standalone nanoGPT training](phase6-nanogpt.md): the ten-update workload,
  manual backward interface, autotuning scope, and correctness criteria.

## Validation runbooks

These plans record how milestone acceptance was performed. Use the current
[quickstart](../guides/quickstart.md) for setup and the corresponding research
report for a measured revision and its results.

| Plan | Scope |
|---|---|
| [Phase 3 validation](phase3-validation.md) | Module closures, frozen/offline installation, remote producers, and PyPI transport |
| [Phase 2 validation](phase2-validation.md) | Bundled NVRTC, runtime ABI, native hosts, compiler comparison, and GPU regressions |
| [Opaque artifact validation](opaque-artifact-validation.md) | Initial executable transfer and independent consumers |
| [Phase 0 experiment design](phase0-experiment-design.md) | Original architecture probes and two-machine protocol |

Current WebGPU transfer instructions are in the
[WebGPU guide](../guides/webgpu.md). Training and LFM2 benchmark reproduction
commands are in the [research index](../research/README.md).
