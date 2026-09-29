# ADR 0011 — Separate runtime calls from compiler IR; adopt bundled NVRTC

**Status:** Accepted · 2026-09-29

## Context

Phase 1 provides a compiler-free CUDA consumer, typed launch metadata and
tested primary-context/foreign-stream ownership. Its implementation combines
argument binding with CUDA execution. Compiler producers still invoke nvcc
and need a CUDA toolkit and a host C++ compiler.

The user authorized Phase 2 with NVRTC and explicitly deferred direct PTX
generation, including a possible tinygrad bridge, until after Tensor v1.
Runtime ABI version 1 is distinct from that future product release.

## Decision

1. Introduce Tensor runtime **call ABI 1.0**, specified in
   [the runtime contract](../runtime-abi.md) and the packaged
   [`tensor/abi.h`](../../src/tensor/include/tensor/abi.h). It describes buffers,
   scalar bits, resolved calls, provider-tagged streams and errors, with fixed
   widths, sizes, borrowing rules and explicit version rejection. Compiler IR
   is not part of the runtime contract.
2. Independently version the artifact envelope. New `tensor.module` v3
   artifacts declare their runtime ABI/capabilities, runtime provider and
   compiler provenance. Existing `tensor.cuda` v1/v2 cubins remain readable.
   The compiler's lowered parameter order remains explicit.
3. Share binding, output allocation, Buffer/Executable lifetime checks and
   benchmarking across CUDA and an independent synchronous CPU provider.
   Providers own allocation, image loading, execution and events. CUDA retains
   its tested context, DLPack and external-stream mechanics. CPU artifacts use
   native exported ABI functions and are built with TileLang's C backend.
4. Default CUDA production to NVRTC 12.9. An explicit `--compiler nvcc`
   (or existing explicit `--nvcc`) retains the offline compiler. Load NVRTC
   and builtins from the configured bundle; resolve headers explicitly. Cache
   identity includes the compiler, options, libraries and header contents.
5. Keep one exact-SM cubin per CUDA artifact in this phase. Neither changing
   compiler nor changing envelope format establishes cross-GPU portability.
   Multiple images/PTX fallback require a later artifact selection contract.

## TVM FFI evaluation

The Phase 0 C++/Rust hosts establish that TVM FFI can host modules without
Python, and its compiler-side use remains valuable. It supplies an object
system, reflection and typed-function dispatch. The supported product runtime
needs resolved buffer/scalar calls, stream tokens and errors; its existing
CUDA consumer already avoids that dependency.

The native Phase 2 hosts execute the real TileLang CPU image through Tensor's
call ABI without TVM FFI, Python or PyTorch. The C++ host also executes the
NVRTC cubin through CUDA driver calls with the same buffer/call layouts.
Choose the smaller Tensor descriptor contract for this scope. Preserve TVM
FFI inside the pinned compiler instead of making its object model or version
the public runtime ABI. Keep DLPack as the external tensor interop protocol.
This is a scoped decision, not a finding that TVM FFI cannot implement a runtime.

## Consequences

CUDA producers can compile on a driver/GPU/toolchain-free host using bundled
NVIDIA compiler libraries and headers. Those components have a substantial
footprint; this is an installation improvement, not a compact native compiler
claim. Python/TileLang/PyTorch still belong to the current producer.

Native callers can consume ABI 1 descriptors without compiler packages. The
validation hosts receive verified/extracted images; they are not standalone
`.tbin` parsers or a native Tensor CLI. Provider allocation/session lifecycles
remain implemented through the Python workbench or owned by native hosts.
The call ABI does not promise a C provider-plugin vtable or full tensor library.

Direct PTX remains a post-Tensor-v1 experimental option. Its acceptance would
require a supported lowering profile, numerical/performance evidence and
measured packaging benefit; tinygrad renderers are not interchangeable TIRx
backends by default.
