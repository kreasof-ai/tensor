# ADR 0009 — Complete Phase 0 with scoped provider and composition contracts

**Status:** Accepted · 2026-09-29
**Evidence:** [E16](../research/e16-phase0-exit.md), exact experiment commit
`157ae365605eda3c4c9ca4aa739fb7f033b406e8`.

## Context

ADR 0008 kept Phase 0 open because the existing CPU execution path did not
prove independent provider registration, frontend artifacts had not been
composed, Rust hosting was untested, foreign stream ownership was undefined,
and static-versus-symbolic throughput had not been compared. Cross-GPU
benchmarking was subsequently removed from the exit scope by explicit user
decision.

## Decision

Phase 0 is complete for the MVP architecture and Phase 1 product work may
begin.

Tensor will build its provider surface over public TileLang backend contexts.
The independent CPU proof uses an unclaimed CPU target and provider-owned
native execution while reusing upstream lowering/codegen components. Tensor
will own the adapter that connects this context to its runtime; it will not
take over a built-in target or modify a private registry.

Versioned frontend TIRx remains the source artifact tier. It supports the
measured pointwise composition and schedule variants. General fusion stays a
Phase 3 capability and must expand the verifier before accepting broader
programs. Post-lowering IR is not a source artifact.

The runtime boundary uses borrowed tensor and stream resources with explicit
ownership and event ordering. The first implementation may expose a
CUDA-specific contract. A provider-neutral event/signal ABI waits until a
second asynchronous provider supplies evidence for the common semantics.

Native hosting supports C++ and Rust executable calls through TVM FFI. Native
compiler packaging is not required for Phase 1: compilation may remain a
pinned Python producer while shipped opaque executables retain a compiler-free
consumer.

Performance claims are limited to the measured A10G. Cross-GPU benchmarking
must be completed before advertising performance portability across GPU
architectures.

## Consequences

ADR 0001's gate is satisfied, so product packages may now be created. Phase 1
starts with the doctor/build/run path and an explicit artifact envelope. The
experiment code remains evidence and should not be promoted wholesale into a
public ABI.

The following are planned product work rather than unresolved Phase 0 gates:
Tensor-owned execution adapters, general composition, neutral asynchronous
primitives, native compiler distribution, and cross-GPU performance evidence.
