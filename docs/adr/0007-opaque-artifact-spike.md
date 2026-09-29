# ADR 0007 — Validate opaque executable transfer before expanding the module API

**Status:** Accepted · 2026-09-29

## Context

E4 demonstrates frontend IR serialization and re-targeting within CUDA. It
does not demonstrate module composition or fusion. The product's separate
promise of fast prebuilt execution without compiler imports also needs a real
load-and-launch test. The local Windows/AMD host cannot run CUDA and has no
`nvcc`; NVIDIA hardware will be prepared later.

## Decision

Keep the prototype under `experiments/p0/`, consistent with ADR 0001. Use one
fixed float32 elementwise kernel and a temporary versioned manifest around
source or cubin payloads. Separate TileLang source emission, `nvcc` compilation
and direct CUDA Driver API execution. Require an exact SM match for this
experiment rather than invent a broader binary-compatibility policy.

The consumer imports only stdlib and NumPy. An import guard rejects TileLang,
TVM, TVM FFI and PyTorch. It validates format, argument/launch contract and
payload hashes before loading native code. Reports identify the producer's
source, lockfile and toolchain, and the consumer's driver and device.

Use local format/isolation/driver-contract tests and real lowering now. Keep
GPU numerics and latency explicitly unverified until the opt-in NVIDIA tests
run. A simulated driver checks host-side packing and resource lifetime; it
cannot validate the generated GPU program.

## Consequences

There is a concrete build/transfer/load path to evaluate without freezing a
public module format or provider ABI. Source bundles include CUTLASS/CuTe as
well as TileLang headers; executable bundles retain redistribution notices
but do not carry compiler headers. Fusion and dynamic-shape execution remain
independent questions. The runtime prototype is compiler-free; the pinned
TileLang producer still imports PyTorch.

Runbook: [opaque artifact validation](../plan/opaque-artifact-validation.md).
