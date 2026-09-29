# ADR 0004 — Phase 0 research runs on a no-GPU host; execution runs remotely

**Status:** Accepted · 2026-09-29

## Context

The proposal's Phase 0 assumes a working target environment. The reality is a development
machine with an RDNA2 GPU, no CUDA, and — per ground truth §7 — no supported way to run
TileLang kernels on it.

The first instinct is to treat the local machine as unable to contribute and to wait for
access to the NVIDIA box.

## Decision

Split Phase 0 explicitly in two, and run the no-GPU half now:

- **Local, no device:** environment/packaging, source emission, pass tracing, IR
  inspection, artifact shape, cache behaviour, provider registration, ABI surface. 26/26
  measurements pass today.
- **Remote, NVIDIA only:** full compile latency with `nvcc`, numerics validation, kernel
  performance, warm-start latency.

Both halves run the same harness from the same git revision so their numbers are
comparable.

## Consequences

- The experiments that decide Tensor's architecture start immediately, rather than waiting
  on hardware access.
- The split must be maintained honestly. Any local number must be labelled as excluding the
  vendor toolchain step, or it overstates what users will experience. E9 exists precisely to
  correct the local figures.
- The proposal's "do not require users to assemble a TVM/TileLang build environment"
  success criterion (§21) can be tested locally, since the local environment was created
  with three commands and no manual dependency assembly.
