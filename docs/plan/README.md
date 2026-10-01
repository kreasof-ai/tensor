# Roadmap and validation plans

[Documentation](../README.md) · [Research evidence](../research/README.md)

The [roadmap](roadmap.md) defines the agreed phase scopes and deferred work.
Phases 0–6 are complete within their measured profiles. Standalone LFM2 inference
and its decode experiments extend those profiles as a concrete workload.

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
