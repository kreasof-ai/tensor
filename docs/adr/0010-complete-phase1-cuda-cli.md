# ADR 0010 — Complete Phase 1 with the measured single-device CUDA profile

**Status:** Accepted · 2026-09-29

## Context

The proposal's Phase 1 requires `doctor`, `build`, `inspect`, `run` and `bench`
on NVIDIA, with installation, latency, diagnostics, runtime and portability
measurements. ADR 0006 adds the kernel-author workbench. The initial static
CLI was validated, but scalar/symbolic exports and GPU DLPack/stream borrowing
were still open. The user requested completion of that remaining work.

## Decision

Mark Phase 1 complete for the implemented CUDA profile. New artifacts carry
typed frontend arguments, lowered CUDA parameter order, integer shape/launch
expressions and alignment in format v2. The compiler-free consumer infers
runtime dimensions, checks scalar widths and shapes, allocates outputs and
launches the same binary across tested shapes. Existing v1 static artifacts
remain readable.

The workbench imports CPU and GPU DLPack producers without importing their
frameworks. GPU imports borrow contiguous writable allocations and hold their
managed-tensor lifetime. Sessions retain the CUDA primary context and own or
borrow a stream. CUDA events provide `wait_for` and `handoff`; cleanup preserves
foreign handles and restores context state.

These are scoped CUDA contracts. They do not freeze a provider-neutral runtime
ABI. Phase 2 owns that separation, versioning and stability work.

## Evidence and accepted limits

[The Phase 1 exit report](../research/phase1-exit.md) records:

- 72 GPU-enabled regression tests passing with zero skips;
- five product artifacts built on a GPU-free Actions host and executed from
  the downloaded wheel in a NumPy-only A10G consumer;
- 22 numerical/compatibility cases, scalar CLI execution and eight expected
  runtime failures passing, with compiler import guards;
- real PyTorch-client producer/consumer stream ordering, unchanged borrowed
  addresses, retained primary contexts and surviving borrowed streams;
- clean-wheel installation and inspection passing on Ubuntu and Windows;
- cold/warm compilation, fresh startup, launch overhead and defined
  installation-step/dependency counts.

Execution is measured on A10G, with exact SM matching. General strided/zero-size
buffers, other GPU execution, cross-device copying, symbolic GEMM tile sizes,
DLPack output export and stable provider-neutral descriptors are not accepted
claims. These limits do not leave the requested Phase 1 work unfinished.

## Consequences

The roadmap advances to Phase 2. The CLI, artifact checks, numerical harness
and CUDA interoperability are concrete consumers against which the stable
runtime ABI can be designed. Host launch overhead includes per-call binding
and validation; its measured cost is part of the baseline, not an optimization
claim. Broader layouts and faster binding can be developed against this
working contract as subsequent runtime work.
