# Tensor runtime contract — call ABI 1.0

This contract separates an executable call from compiler representations and
provider mechanisms. It applies to the current contiguous, positive-extent,
single-device profile on 64-bit little-endian hosts. The
[C header](../src/tensor/include/tensor/abi.h) ships in the consumer wheel.

## Independent versions

| Version | Meaning |
|---|---|
| Tensor package `0.1.0` | Current development distribution |
| Artifact envelope `tensor.module` v3 | Container, image selection and metadata schema |
| Runtime call ABI 1.0 | Buffer, argument, call, stream and error layouts |
| TileLang / TVM FFI versions | Producer/frontend IR provenance; not consumer dependencies |

V3 declares `runtime_abi.major`, `minor`, and `required_capabilities`. Unknown
major versions, newer required minors and missing capabilities fail before
image loading. Compatible additions must preserve existing field meanings,
numeric IDs and offsets. Native calls carry `abi_version` and `struct_size`;
larger trailing structures can be ignored, but undersized ones fail.

Consumers also read the old CUDA envelope versions 1 and 2. Those artifacts
are adapted to the current call layout at launch. They do not gain CPU or
cross-SM execution support. Older consumers reject v3 through format checks.

## Descriptors

| C structure | Bytes | Meaning |
|---|---:|---|
| `TensorBufferV1` | 48 | Address, capacity, shape, byte strides, dtype and device |
| `TensorArgumentV1` | 64 | Buffer or scalar, in the image's declared parameter order |
| `TensorStreamV1` | 16 | Device type, ordinal and opaque provider-specific handle |
| `TensorCallV1` | 72 | Version, size, argument array, resolved launch geometry and stream |
| `TensorErrorV1` | 512 | Status and bounded NUL-terminated UTF-8 diagnostic |

Device types 1 (CPU) and 2 (CUDA) follow DLPack's device numbers. Dtype IDs are
listed in the header and never renumbered. Scalar values occupy the first
`sizeof(dtype)` little-endian bytes of the eight-byte scalar field; unused
bytes are zero. Buffer arguments carry rank, dtype, address, capacity and
contiguous **byte** strides. Descriptors do not own data or contain framework
objects. GPU addresses remain device addresses, never dereferenced by the host.

The shared runtime infers direct symbolic dimensions, verifies all input
shapes/scalars/alignment, resolves bounded integer expressions, allocates
declared outputs and creates the call in lowered argument order. Native callers
are responsible for equivalent manifest binding/validation before CUDA launch.
CPU image wrappers validate their argument count, kinds, dtype, extents,
strides, capacity, stream and ABI before touching data.

## Ownership and completion

- Calls borrow argument arrays, shapes and strides through return. CPU calls
  complete synchronously; CUDA submission consumes metadata before return.
- Buffer storage must survive until the execution stream finishes. Borrowed
  DLPack storage retains its producer-managed owner until synchronized release.
- Sessions own allocations, loaded images and events they create. A borrowed
  external stream remains owned by the caller and must outlive the session.
- `record_event()` returns a session-owned completion point; `wait(event)`
  establishes a dependency on the same provider/device. CPU events are already
  complete. `release()` and session exit release events exactly once.
- CUDA `wait_for(handle)` / `handoff(handle)` remain explicit external-stream
  adapters. Session exit also waits for handed-off consumer work before
  releasing borrowed buffers and restores the previous CUDA context.
- Closing a session invalidates its buffers/executables/events. Reopening the
  same session object never makes old executable handles valid again.

## Runtime provider boundary

`tensor.runtime` owns Buffer/Executable binding, outputs, events and timing.
`tensor.providers.Device(provider=...)` selects an implementation without
importing its compiler. Providers implement allocation, upload/download,
image loading, resolved call submission, synchronization and event operations.
`require_capabilities` rejects unsupported requests before initializing a
provider. This is a runtime workbench interface; ABI 1 freezes the native call
descriptors, not a C provider-plugin lifecycle table.

| Capability | CUDA | CPU validation provider |
|---|---|---|
| Contiguous buffers, typed scalars, direct symbolic dimensions | Yes | Yes |
| Events | CUDA completion events | Synchronous completion |
| Asynchronous launch | Yes | No |
| Borrowed external streams / GPU DLPack | Yes | No |
| Executable image | One exact-SM cubin | Linux x86-64 native shared library |

CUDA uses the primary context, checks runtime driver entrypoints and enforces
the existing exact-SM/alignment/launch limits. CPU is a contract-validation
provider using TileLang's C lowering, not an optimized CPU library. The current
CPU producer accepts scalar dtypes supported by the profile and constant/direct
symbolic shapes; float16 buffer code generation is outside its profile. Native
shared libraries also depend on their producer's OS/libc ABI.

## Compiler boundary

Source → TileLang/TIRx lowering → target source → executable compiler → image.
The CUDA executable compiler is NVRTC by default, with explicit nvcc support.
The CPU executable compiler is the host C++ compiler. Neither compiler belongs
to the consumer/provider path. Producer compiler versions and hashes identify
cache entries; they are not runtime version-equality requirements.

See [ADR 0011](adr/0011-runtime-call-abi-and-nvrtc.md) for the TVM FFI evaluation
and the post-v1 direct-PTX decision.
