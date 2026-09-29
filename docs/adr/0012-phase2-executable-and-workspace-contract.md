# ADR 0012: Complete Phase 2 executable, event and workspace contracts

Status: accepted, 2026-09-29.

## Context

ADR 0011 delivered the shared call ABI and NVRTC producer. The remaining
Phase 2 contracts were executable identity/lifetime and explicit workspace
requirements. Remote Linux and Windows producers also needed GPU acceptance.

## Decision

Publish runtime ABI 1.1 while preserving all ABI 1.0 call, buffer, scalar,
stream and error layouts. Add executable (64 bytes), workspace (24 bytes) and
event (40 bytes) descriptors to the packaged C header and native host examples.
Use opaque, non-reused, process-local handles qualified by a device session.
Resolve through the originating host registry and validate descriptor metadata.
Release waits for outstanding work, unloads the image and invalidates the token;
session close invalidates every owned resource. Hosts serialize session mutation.

The current kernel profile declares zero external workspace with alignment one.
Reject nonzero or unknown requirements before loading instead of silently
allocating scratch space. Explicit buffer parameters and CUDA dynamic shared
memory retain their existing meanings. Older CUDA artifacts remain readable;
ABI 1.0 artifacts have implicit zero external workspace.

Keep provider dispatch in the host. The descriptor contract is not a C plugin
vtable or a native artifact parser. CPU remains a Linux x86-64 validation
provider; optimized CPU and additional GPU providers remain later work.
Direct PTX remains experimental after Tensor v1.

Cache immutable buffer metadata and activate the CUDA context once at actual
submission. Preserve shape, scalar, alignment, launch and ownership validation.
Measure host enqueue separately from launch plus synchronization.

## Acceptance

Exercise release/reload, failed loads, foreign/stale identities and modified
workspace metadata on CPU; release queued work and order sessions through an
event descriptor on CUDA. C++ and Rust hosts validate the same layouts and
reject invalid handles/workspace. Download successful Linux and Windows CI
archives, verify GitHub digests and source hashes against the exact producer
revision (LF or CRLF), then run their kernels on a separate GPU host with only
Tensor and NumPy. Record limitations and measurements in the Phase 2 exit report.
