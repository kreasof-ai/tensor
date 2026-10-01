# Tensor runtime contract — ABI 1.2

[Documentation](../README.md) · [Python runtime guide](../guides/runtime.md)

This contract separates an executable call from compiler representations and
provider mechanisms. It applies to the current contiguous, positive-extent,
single-device profile on 64-bit little-endian hosts. The
[C header](../../src/tensor/include/tensor/abi.h) ships in the consumer wheel.

## Independent versions

| Version | Meaning |
|---|---|
| Tensor package `0.1.0` | Current development distribution |
| Artifact envelope `tensor.module` v3 | Container, image selection and metadata schema |
| Runtime ABI 1.1 | Existing call layouts plus executable, event and workspace descriptors |
| Runtime ABI 1.2 | Same layouts; explicit session-qualified opaque buffer arguments |
| TileLang / TVM FFI versions | Producer/frontend IR provenance; not consumer dependencies |

V3 declares `runtime_abi.major`, `minor`, and `required_capabilities`. Unknown
major versions, newer required minors and missing capabilities fail before
image loading. Compatible additions must preserve existing field meanings,
numeric IDs and offsets. Native calls carry `abi_version` and `struct_size`;
larger trailing structures can be ignored, but undersized ones fail.

Consumers also read the old CUDA envelope versions 1 and 2. Those artifacts
are adapted to the current call layout at launch. They do not gain CPU or
cross-SM execution support. Older consumers reject v3 through format checks.
ABI 1.1 producers declare executable descriptors and zero external workspace.
Existing ABI 1.0 v3 artifacts remain accepted with implicit zero workspace.
CUDA/CPU producers still require minor 1. WebGPU requires minor 2 and the
`opaque_buffer_handles` capability; old consumers and pointer-only providers
reject those artifacts before loading them.

ABI 1.2 reserves device type 256 for WebGPU, independently of DLPack. Argument
kind 3 (`TENSOR_ARG_OPAQUE_BUFFER`) keeps the same 64-byte structure: `buffer`
carries shape, strides, capacity and device metadata with `address == 0`, while
`scalar` holds a nonzero buffer handle. The call's `stream.handle` identifies the
owning live session. Providers resolve the handle and validate metadata and the
artifact's logical shapes before submission. Session close or buffer release
invalidates every copy of the handle. Kind 1 remains a raw-address buffer and
kind 2 remains a scalar. No field offsets, structure sizes or existing IDs change.

WebGPU exposes an owned queue, asynchronous submission, completion events and
NumPy upload/download. It rejects external streams and raw-pointer DLPack imports.
Its workgroup allocations live in the shader and are checked against adapter
limits, rather than passing CUDA dynamic shared memory at launch. External global
workspace remains zero. WebGPU event waits currently wait for the originating
queue's completion, including subsequent work; they guarantee correctness but
do not provide CUDA's fine-grained stream ordering.

## Descriptors

| C structure | Bytes | Meaning |
|---|---:|---|
| `TensorBufferV1` | 48 | Address, capacity, shape, byte strides, dtype and device |
| `TensorArgumentV1` | 64 | Buffer or scalar, in the image's declared parameter order |
| `TensorStreamV1` | 16 | Device type, ordinal and opaque provider-specific handle |
| `TensorCallV1` | 72 | Version, size, argument array, resolved launch geometry and stream |
| `TensorErrorV1` | 512 | Status and bounded NUL-terminated UTF-8 diagnostic |
| `TensorWorkspaceRequirementsV1` | 24 | External scratch size, alignment and device placement |
| `TensorExecutableV1` | 64 | Session-qualified image identity and workspace requirements |
| `TensorEventV1` | 40 | Session-qualified completion-point identity |

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

## Executable and workspace contract

The manifest specifies the image, entrypoint, target, arguments, guards,
required ABI and workspace. Payload hashes and requirements are checked before
loading. A failed load publishes no executable handle.

`kernel.descriptor` returns a snapshot with nonzero, process-local session and
handle tokens. These tokens are opaque registry keys, never function addresses
or serialized artifact data. `device.get_executable(snapshot)` resolves only
that live session's registered executable and verifies its argument count,
flags and workspace metadata. Tokens are not reused. A snapshot borrows its
resource's lifetime; closing or reopening a session cannot revive it.

`kernel.release()` synchronizes submitted and handed-off work, unloads the
image and invalidates its handle. Repeated release is harmless. Loaded kernels
can also be context managers. Session exit releases remaining executables.
C++ and Rust hosts use the same descriptors with their own private dispatch
registries; this does not define a C provider-plugin lifecycle table.

`kernel.workspace_requirements()` returns zero bytes, alignment one and the
provider's device type, with zero flags and reserved fields. ABI 1.1 manifests
declare `workspace: {"bytes": 0, "alignment": 1}`. Nonzero requirements and
unknown workspace forms fail before image loading. Providers allocate no
hidden external workspace. Explicit frontend buffer parameters remain ordinary
arguments; unbound lowered parameters are rejected by the producer. CUDA
dynamic shared memory remains a separate per-call launch field.

Buffer address, shape, strides, dtype and capacity are immutable. Their cached
descriptor backing lives with the buffer; calls borrow it through submission.
Storage must still survive asynchronous execution.

`event.descriptor` similarly carries a session-qualified token.
`device.get_event(snapshot)` resolves it in its originating session. The
resolved event may order another session on the same provider and device.
CPU sets the known-complete flag. CUDA leaves it unset and uses provider event
ordering; an unset flag does not assert that work is unfinished. Release and
session exit invalidate event tokens.

Hosts must serialize operations that mutate a session, including launch,
release and close. Asynchronous device execution does not permit concurrent
mutation of its host registry.

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
provider. ABI 1.1 freezes descriptor layouts and their lifetime rules.
Provider dispatch remains owned by the host; a C plugin table is future work.

| Capability | CUDA | CPU validation provider |
|---|---|---|
| Contiguous buffers, typed scalars, direct symbolic dimensions | Yes | Yes |
| Events | CUDA completion events | Synchronous completion |
| Executable descriptors / zero external workspace | Yes | Yes |
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

See [ADR 0011](../adr/0011-runtime-call-abi-and-nvrtc.md) for the TVM FFI evaluation
and the post-v1 direct-PTX decision, and [ADR 0012](../adr/0012-phase2-executable-and-workspace-contract.md)
for the ABI 1.1 executable, event and workspace contract.
