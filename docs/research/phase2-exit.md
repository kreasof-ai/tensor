# Phase 2 exit — stable runtime ABI and NVRTC

Phase 2 is complete for the contiguous, positive-extent, single-device profile.
Acceptance on 2026-09-29 used implementation commit
`1a5a904112dcb439fe90ec854cc12aa6748894b3`, Linux x86-64 and NVIDIA A10G
(`sm_86`). The consumer checkout was clean at acceptance. The
[contract](../runtime-abi.md), [ADR 0011](../adr/0011-runtime-call-abi-and-nvrtc.md)
and [ADR 0012](../adr/0012-phase2-executable-and-workspace-contract.md) define
the compiler/runtime/provider boundary.

## Contract gates

| Proposal Phase 2 requirement | Implemented and validated |
|---|---|
| Tensor descriptor | Fixed buffer/scalar layouts, contiguous byte strides, capacity, dtype, device and immutable metadata |
| Executable descriptor | Session-qualified opaque identity, argument count, asynchronous flag, workspace and release/reload lifetime |
| Stream abstraction | Provider-tagged stream descriptor; owned and borrowed CUDA streams with explicit handoff |
| Event abstraction | Session-qualified completion identity, CPU known-complete flag, CUDA ordering across sessions |
| Workspace contract | Declared zero external bytes and alignment one; unsupported forms fail before loading; dynamic shared memory stays separate |
| Provider capability model | Requirements negotiated before initialization/image loading; independent CUDA and CPU implementations |

Runtime ABI 1.1 adds executable (64 bytes), workspace (24 bytes) and event
(40 bytes) descriptors while preserving ABI 1.0 call layouts. Failed loads
publish no handle. Explicit executable release waits for queued and handed-off
work, unloads the image and invalidates snapshots. Session close/reopen cannot
revive old identities. C++ and Rust hosts use these descriptors with private
dispatch registries. Host operations that mutate a session must be serialized.

The artifact envelope remains `tensor.module` v3 and is versioned independently
of the runtime and frontend compiler. Existing CUDA v1/v2 artifacts and v3
ABI 1.0 artifacts still execute. Compiler versions are provenance rather than
runtime equality requirements. NVRTC remains the default; nvcc is explicit.

## Acceptance evidence

| Check | Result |
|---|---|
| Full GPU-enabled suite at implementation commit | **89 passed, zero skips**, 53.82 s |
| Linux Actions producer | **73 passed, 16 skipped**, 8.77 s; all five profiles built and inspected |
| Windows Actions producer | **66 passed, 23 skipped**, 9.86 s; all five profiles built and inspected |
| Linux producer → separate A10G consumer | **21 numerical/interop cases + eight diagnostics**, CLI run/bench and five executable/workspace checks |
| Windows producer → separate A10G consumer | Same acceptance matrix, using the Windows-produced wheel and cubins |
| Event descriptor acceptance | Both transferred consumers resolved events, waited and rejected released identities |
| Compatibility / compiler-free CPU | CUDA v1, v2, v3 ABI 1.0 and a native CPU ABI 1.1 image executed |
| Native C++ CPU / Rust CPU | Each: 129 numerical results and five rejection checks |
| Native C++ CUDA | 129 numerical results and three executable rejection checks |

The [successful CI run](https://github.com/kreasof-ai/tensor/actions/runs/36643849222)
compiled elementwise, GEMM + ReLU, dynamic affine, dynamic GEMM and int64 offset,
checked cold/warm caches and corrupt-image recovery, and installed the wheel
in a fresh Tensor/NumPy environment with compiler imports prohibited. GPU tests
skip on the Actions runners; the scoped CPU native-image tests also skip on
Windows. GPU execution of Windows-produced artifacts took place on Linux.

Each downloaded archive matched the SHA-256 digest from GitHub's Actions API.
Artifact hashes matched the producer report and source hashes matched its
exact revision, with Linux LF and Windows CRLF checkout bytes verified.
Each GPU consumer contained exactly Tensor 0.1.0 and NumPy 2.5.3; its installed
package files matched the wheel in that archive. Producer and GPU consumer
hostnames differ, and the consumer used the same clean implementation commit.
Compiler imports remained prohibited during numerical and CLI acceptance.

The prior driver/GPU/toolchain-free container and include audit remain recorded
in the [initial report](phase2-validation.md): all five kernels compiled, with
2,885 successful include/source reads confined to explicit bundle roots.
The executable compiler and CUDA source-emission implementation did not change
during contract completion. Fresh CI additionally validates ABI 1.1 manifests.

Raw evidence: [exit](data/phase2-exit.json),
[CI jobs and archives](data/phase2-completion-ci.json),
[Linux transfer](data/phase2-transfer-linux.json),
[Windows transfer](data/phase2-transfer-windows.json) and
[compiler/runtime measurements](data/phase2-exit-metrics.json).
The [runbook](../plan/phase2-validation.md) reproduces these checks.

## Launch overhead

Buffer shape/stride backing is now cached with immutable buffer metadata.
Binding retains validation, and CUDA activates its context once immediately
before submission. Final consumer timings below use ten warmups, 100
iterations and preallocated outputs on A10G.

| Workload | NVRTC / nvcc host enqueue (µs) | NVRTC / nvcc launch + synchronization (µs) |
|---|---:|---:|
| Elementwise | 58.2 / 56.7 | 61.8 / 61.3 |
| GEMM + ReLU | 64.7 / 65.7 | 72.2 / 72.1 |
| Dynamic affine | 85.1 / 85.9 | 91.0 / 91.1 |
| Dynamic GEMM | 76.4 / 78.2 | 81.5 / 83.7 |
| int64 offset | 56.3 / 57.0 | 61.0 / 61.7 |

The initial Phase 2 NVRTC enqueue measurements were 64.4 µs static and
97.0 µs dynamic affine; the final observations improve those by about 10% and
12%. They remain above the historical Phase 1 49.4/69.6 µs measurements.
These are host binding/submission measurements, not GPU kernel duration or
cross-GPU throughput claims. Run-to-run timing varies; there is no latency
threshold in the Phase 2 contract.

The compiler comparison used 20 fresh cold/warm build processes with independent
caches during completion. Outputs matched for all five NVRTC/nvcc pairs.
Those exact artifacts were then remeasured with the final Linux CI consumer
wheel; the raw report distinguishes build provenance and runtime revision.
Whole-process build costs still include frontend imports, lowering and compiler
library/header hashing. OS and compiler caches were not reset.

## Scope after completion

CUDA still requires one exact-SM cubin and a compatible driver; the measured
device is A10G. There is no cross-SM image selection or PTX fallback. External
workspace is supported only as an explicit zero requirement. CPU remains a
Linux x86-64 validation provider, rather than an optimized CPU implementation.

Provider dispatch remains host-owned. A C provider-plugin lifecycle table and
native `.tbin` parser are future interfaces. The producer still uses Python
compiler packages that import PyTorch; the NVRTC/header bundle alone occupies
245,936,741 installed bytes, so compact compiler packaging remains future work.
**Direct PTX, including a possible tinygrad bridge, stays experimental after
Tensor v1.** None of these later extensions blocks the scoped Phase 2 contracts.
