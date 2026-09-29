# Phase 2 — runtime call ABI and bundled NVRTC

This is the historical ABI 1.0 implementation report. Phase 2 subsequently
completed with ABI 1.1 executable/event/workspace contracts and GPU acceptance
of both remote producer bundles. See the [exit report](phase2-exit.md) for the
final gates, updated timings and evidence.

Local validation on 2026-09-29, Linux x86-64 and NVIDIA A10G (`sm_86`).
The initial local run used an uncommitted working tree; its retained reports
identify the base revision, lock hash, artifact hashes, compiler libraries and
consumer wheel. Subsequent remote CI results are recorded below.
[ADR 0011](../adr/0011-runtime-call-abi-and-nvrtc.md) records the decision and
[the runtime contract](../runtime-abi.md) specifies the supported profile.

## Implemented boundary

- Call ABI 1.0: fixed C layouts for buffer capacity/shape/byte strides,
  typed scalar bits, launch geometry, provider-tagged streams and errors.
- Independently versioned `tensor.module` v3 envelope, with runtime
  capabilities and compiler provenance. CUDA v1/v2 remain readable.
- Shared Buffer/Executable binding, output allocation, lifecycle checks,
  events and timing. CUDA preserves its primary-context, DLPack and external
  stream adapters; a synchronous CPU provider executes native ABI functions.
- NVRTC 12.9 by default, with explicit nvcc support. Compiler options,
  library contents and headers enter cache identity; corrupt images rebuild.
- Standalone C++ CPU/CUDA and Rust CPU validation hosts. The CPU image is
  actual TileLang C lowering, rather than a separately handwritten kernel.

This freezes the native **call** contract. It does not freeze a C provider
plugin lifecycle table. Native hosts receive verified/extracted images;
they are not standalone `.tbin` readers. CPU code generation is currently
Linux x86-64, contiguous buffers and constant/direct symbolic extents;
float16 buffers are outside that validation profile.

## Results

| Check | Observed result |
|---|---|
| Full GPU-enabled pytest suite | 82 passed, zero skips, 52.83 s |
| Isolated NVRTC producer | All five profiles compiled, cold/warm cache and corruption recovery passed |
| Producer isolation | No CUDA driver, GPU access, nvcc, ptxas or host C/C++ compiler |
| Compiler include audit | 2,885 successful distinct include/source reads, all inside explicit bundle/TileLang/CUTLASS roots |
| Clean Tensor/NumPy CUDA consumer | 22 numerical/interop/legacy cases and eight diagnostic cases passed, compiler imports prohibited |
| NVRTC versus nvcc | Five workloads × two compilers; identical output snapshots |
| C++ CPU and Rust CPU hosts | Each: 129 numerical results, two descriptor rejection checks |
| C++ CUDA host | 129 numerical results through CUDA driver calls and ABI descriptors |
| Clean consumer CPU / CUDA v2 | CPU dynamic extents and existing CUDA v2 execution passed separately |

The producer ran in a GPU-free Docker container on the **same physical host**
as the consumer. It used the pinned installed compiler packages copied into
the image, not a fresh install of those packages. The consumer environment
contained exactly `tensor-workspace==0.1.0` and `numpy==2.5.3`.
This is environment isolation and artifact transfer, not a new two-host result.
The previous Phase 1 two-host evidence remains in its historical report.
The subsequent [Linux/Windows CI run](https://github.com/kreasof-ai/tensor/actions/runs/36640410723)
passed on commit `b90c70b`: Linux had 69 tests passed and 15 skipped;
Windows had 65 passed and 19 skipped. GPU tests skip on these GPU-free runners,
and the scoped native CPU profile also skips on Windows. Both jobs compiled
all five NVRTC profiles, verified cold/warm cache and corruption recovery,
built the wheel, and inspected the artifacts in fresh Tensor/NumPy consumers
with compiler imports prohibited. This is cross-platform producer and consumer
inspection evidence; GPU numerical execution remains the local A10G result.

The first Windows run exposed a case-sensitive test assertion (`nvcc.EXE`
versus `nvcc.exe`). After correcting it, the producer exposed an incompatible
audit that replaced `subprocess.Popen` with a function before Windows asyncio
subclassed it. The audit now observes process creation with a scoped Python
audit hook, preserving the Popen class and blocking compiler calls. Regression
tests cover class inheritance and quoted Windows executable names. Matrix
jobs finish independently when another platform fails.

Raw evidence: [producer](data/phase2-nvrtc-producer.json),
[consumer](data/phase2-consumer.json), [compiler comparison](data/phase2-metrics.json),
and [local validation](data/phase2-validation.json), plus
[remote CI evidence](data/phase2-ci.json). The producer report records
the file-trace digest and accepted header roots; the full trace is retained
under `build/phase2-isolated/` rather than committed.

## Cost and performance

The bootstrap installs **245,936,741 bytes** of dynamic NVRTC/builtins libraries,
CUDA/CRT/CCCL headers and licenses. Downloads total **197,844,784 bytes**;
the nvcc redistribution archive supplies CRT headers but its executables,
ptxas and NVVM are discarded. Archives are stored outside the installed bundle.
TileLang/CUTLASS headers and Python compiler dependencies are additional.
The installed Linux compiler environment was approximately 5.6 GB, including
PyTorch and its dependencies. This removes a system-toolkit installation
requirement; producer size remains substantial.

Fresh producer processes and independent caches were used for the comparison.
Warm builds still lower source and hash the compiler libraries/headers.
These runs were uninstrumented; the isolated producer's strace timings are
separate evidence and should not be compared directly.

| Workload | NVRTC / nvcc compile (s) | NVRTC / nvcc process cold (s) | NVRTC / nvcc process warm (s) | NVRTC / nvcc host enqueue (µs) |
|---|---:|---:|---:|---:|
| Elementwise | 1.106 / 1.373 | 4.652 / 4.330 | 3.417 / 2.956 | 64.4 / 64.9 |
| GEMM + ReLU | 1.168 / 1.725 | 5.264 / 5.260 | 4.079 / 3.583 | 77.2 / 75.8 |
| Dynamic affine | 1.099 / 1.361 | 4.827 / 4.626 | 3.672 / 3.237 | 97.0 / 96.1 |
| Dynamic GEMM | 1.176 / 1.691 | 5.276 / 5.270 | 4.083 / 3.633 | 89.4 / 88.1 |
| int64 offset | 1.119 / 1.355 | 4.584 / 4.300 | 3.472 / 3.007 | 65.5 / 66.6 |

Consumer timing uses ten warmups and 100 iterations with preallocated buffers.
It includes binding/validation/call packing; it is not GPU kernel duration.
Static launch plus stream synchronization measured 68.3 / 69.1 µs for
NVRTC/nvcc; dynamic affine measured 101.3 / 101.7 µs. Compiler selection had
little observed runtime effect. Relative to the historical Phase 1 static
49.4 µs and dynamic 69.6 µs enqueue measurements, the shared descriptor path
adds visible Python overhead. No throughput or cross-GPU claim follows.

NVRTC `.tbin` files in this comparison were 24–36 KB versus nvcc's 6–16 KB.
Compiler stages were faster with NVRTC here, while library hashing increased
whole-command and warm build costs. Neither compiler nor OS caches were reset.

## Remaining scope

CUDA still loads one exact-SM cubin. NVRTC does not make artifacts portable
across GPUs or eliminate driver/image compatibility requirements. No broader
GPU matrix, PTX fallback, multi-image selection or optimized CPU provider is
claimed. A C provider plugin table and native artifact parser remain future
work. Direct PTX, including any tinygrad bridge, remains an experimental
option **after Tensor v1**, distinct from runtime call ABI version 1.

See [the reproduction runbook](../plan/phase2-validation.md) for producer,
consumer, native-host and measurement commands.
