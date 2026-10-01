# Tensor: A Simple, Extensible Runtime and Compiler Environment for Tensor Programs

This is the original design proposal. Current implementation scope is in the
[roadmap](../plan/roadmap.md); current usage is in the
[documentation index](../README.md).

**Status:** Draft proposal  
**Working project name:** Tensor  
**Primary executable:** `tensor`  
**Compiler subsystem:** `tensorc`

## 1. Summary

Tensor is a proposed execution environment for high-performance tensor programs built around one principle:

> Make tensor software as easy to build, distribute, compile, and run as modern application software, while containing rather than eliminating the complexity of heterogeneous accelerator hardware.

The project initially targets **single-device tensor computation**. It does not attempt to become a complete ML framework, distributed training framework, cluster scheduler, or universal accelerator abstraction.

The first implementation would build upon existing compiler work rather than create another compiler stack from scratch:

- **TileLang** provides the initial kernel programming and compatibility interface.
- **TIRx** provides the primary internal compiler substrate and architectural inspiration.
- **Tensor** provides the missing product layer around them: CLI, modules, artifact caching, provider loading, framework integration, runtime ABI, packaging, diagnostics, and eventually an ecosystem of reusable tensor modules.

TileLang currently uses TVM's TIRx representation internally, following a May 20, 2026 migration. TIRx itself is explicitly positioned as a hardware-native compiler foundation beneath higher-level systems such as TileLang. Its design keeps pipeline structure, synchronization, hardware roles, memory placement, and backend intrinsics explicit while providing reusable execution scopes, layouts, and tile primitives.

The long-term goal is not merely a compiler executable. It is an environment in which:

```text
tensor run model.py
tensor build attention.py
tensor bench attention.py
tensor add flash-attention
```

feels ordinary regardless of whether the target hardware is NVIDIA, AMD, Apple, or a future accelerator.

---

# 2. Motivation

Tensor compiler infrastructure today is technically sophisticated but operationally fragmented.

Users frequently interact with several independent layers:

```text
framework
    ↓
graph capture
    ↓
graph compiler
    ↓
kernel compiler
    ↓
vendor compiler
    ↓
runtime
    ↓
driver
    ↓
hardware
```

Each layer may introduce its own installation requirements, caches, runtime assumptions, binary formats, device abstractions, version constraints, and diagnostics.

This complexity is partly unavoidable. NVIDIA, AMD, Apple, CPUs, NPUs, and future AI accelerators are genuinely different systems.

Tensor therefore does **not** assume that accelerator infrastructure can be made intrinsically uniform.

Instead it proposes a strong complexity boundary:

```text
                   clean side
─────────────────────────────────────────

frontends
modules
Tensor runtime ABI
compiler interface
CLI/package tooling

══════════════ provider boundary ═════════

NVIDIA-specific complexity
AMD-specific complexity
Metal-specific complexity
future accelerator complexity

─────────────────────────────────────────
                 hardware
```

The core is expected to remain conceptually small.

Provider implementations are allowed to be complicated.

---

# 3. Design Principles

## 3.1 Single-device first

The initial compiler should be exceptionally good at:

> Turning a tensor/kernel program into efficient executable code for one accelerator.

It should not initially own:

- distributed scheduling,
- cluster membership,
- fault tolerance,
- elasticity,
- checkpoint orchestration,
- data-parallel policy,
- tensor-parallel policy,
- collective algorithm selection,
- Kubernetes integration,
- Slurm integration,
- multi-node topology optimization.

These may be implemented later by libraries and infrastructure systems above Tensor.

---

## 3.2 Compatibility before language invention

Tensor should not initially require developers to learn another kernel language.

TileLang provides an existing productive interface with constructs such as:

```python
T.alloc_shared(...)
T.alloc_fragment(...)
T.copy(...)
T.gemm(...)
T.Pipelined(...)
T.Parallel(...)
```

Tensor's first frontend should therefore target a useful **TileLang compatibility profile**.

The objective is not necessarily arbitrary Python compatibility.

The objective is compatibility with tensor/kernel programs written using the supported TileLang programming model.

---

## 3.3 Compiler core and frontend must be independent

TileLang is an initial frontend, not Tensor's permanent language identity.

Conceptually:

```text
TileLang ─────────────┐
                     │
PyTorch FX ──────────┤
                     │
JAX / StableHLO ─────┤
                     ↓
              compiler interface
                     ↓
                 Tensor core
                     ↓
                  provider
```

Different frontends may enter at different abstraction levels.

A TileLang frontend may already contain scheduling and memory placement information.

A PyTorch FX frontend may contain high-level tensor operations requiring substantially more lowering.

---

## 3.4 TIRx is an implementation substrate, not the public module ABI

TIRx is especially attractive because its current philosophy matches Tensor's desired hardware-extension model:

```text
new hardware feature
        ↓
backend intrinsic
        ↓
repeated usage pattern
        ↓
tile primitive
        ↓
higher-level automation, if useful
```

TIRx explicitly argues that future hardware should grow backend libraries rather than continuously enlarging the core language.

Tensor should adopt this philosophy.

However, Tensor packages should not permanently promise:

```text
Tensor Module ABI = serialized TIRx
```

TIRx should initially be an internal representation and compiler dependency.

The module contract must remain independently versioned so that Tensor could eventually replace or augment TIRx without breaking the ecosystem.

---

## 3.5 Intent above mechanism

Generic interfaces should express:

```text
allocate memory with these properties
execute this operation asynchronously
wait for this event
copy between these address spaces
launch this executable on this stream
```

rather than:

```text
cudaMallocAsync
hipMemcpyAsync
MetalCommandBuffer
NVSHMEM put
```

Mechanism belongs in providers.

Semantics belong in the core.

---

## 3.6 Escape hatches are intentional

Hardware-specific behavior must be allowed without immediately polluting the portable abstraction.

Evolution should follow:

```text
raw intrinsic
     ↓
target capability
     ↓
reusable provider operation
     ↓
portable primitive
```

Only concepts demonstrated across meaningful workloads or architectures should enter the common compiler model.

---

# 4. Proposed Architecture

```text
┌──────────────────────────────────────────────────┐
│                    FRONTENDS                     │
│                                                  │
│  TileLang     PyTorch FX     future native DSL   │
│      │             │                 │           │
└──────┼─────────────┼─────────────────┼───────────┘
       │             │                 │
       ▼             ▼                 ▼

┌──────────────────────────────────────────────────┐
│                 COMPILER LAYER                   │
│                                                  │
│        semantic graph / frontend lowering        │
│                       │                          │
│                       ▼                          │
│               TileLang / TIRx                    │
│                       │                          │
│                 specialization                   │
│                 optimization                     │
│                 tile dispatch                    │
│                       │                          │
└───────────────────────┼──────────────────────────┘
                        ▼

┌──────────────────────────────────────────────────┐
│                TENSOR RUNTIME ABI                │
│                                                  │
│ executable   tensors   streams   events          │
│ workspace    capabilities    artifacts           │
└───────────────────────┬──────────────────────────┘
                        ▼

┌──────────────────────────────────────────────────┐
│                    PROVIDERS                     │
│                                                  │
│ NVIDIA      AMD      Metal      CPU      future  │
└───────────────────────┬──────────────────────────┘
                        ▼

                    vendor driver
                        ↓
                     hardware
```

---

# 5. The `tensor` Executable

The primary product surface is one executable:

```text
tensor
```

Initial commands:

```text
tensor run
tensor build
tensor bench
tensor inspect
tensor doctor
```

Later:

```text
tensor add
tensor remove
tensor install
tensor publish
tensor tune
tensor test
tensor fmt
```

Examples:

```bash
tensor run attention.py

tensor bench attention.py

tensor build attention.py --target native

tensor inspect attention.py --stage tirx
```

The primary UX objective is:

> Users should install Tensor, not assemble a compiler toolchain.

This does not require every compiler component to be internally reimplemented by Tensor.

An early AMD provider, for example, may internally use existing ROCm compiler infrastructure.

A Metal provider may emit MSL and invoke platform tooling.

An NVIDIA implementation may emit PTX.

Those implementation differences are acceptable as long as the Tensor contract remains coherent.

---

# 6. Compiler Interface

The compiler should accept **fragments**, not only complete applications.

Conceptually:

```text
tc_compile(program, target, options) -> executable
```

A program contains:

```text
inputs
outputs
tensor operations
shape constraints
layout information where known
required capabilities
```

The output contains:

```text
executable code
entrypoint
guards
workspace requirements
metadata
target requirements
```

This fragment-oriented contract is essential for framework interoperability.

---

# 7. TileLang Compatibility Frontend

The first frontend should support a deliberately defined TileLang subset/profile.

Candidate initial constructs include:

```text
T.Tensor

T.alloc_shared
T.alloc_fragment

T.copy
T.gemm

T.Parallel
T.Pipelined
T.serial

basic arithmetic
indexing
control flow
reductions
barriers
```

The initial implementation should avoid promising arbitrary Python behavior.

For example:

```python
@T.prim_func
def kernel(...):
    ...
```

is considered Tensor-compilable source.

But arbitrary execution such as:

```python
import random_external_python_package
result = dynamic_python_metaprogram(...)
```

does not automatically become part of the compatibility promise.

The compatibility target is:

> TileLang kernel semantics.

Not:

> CPython.

This constraint is necessary to keep Tensor's compiler surface finite.

---

# 8. TIRx Integration

TIRx should initially provide the low-level compiler substrate.

Its model is particularly relevant because TIRx exposes three authoring layers within the same representation:

```text
core TIRx operations
tile primitives
backend-specific operations
```

Tile primitives are dispatched according to execution scope, layout, operands, and backend, while backend-specific operations may bypass portable tile dispatch entirely.

This provides the desired extension path:

```text
portable kernel
      ↓
tile primitives
      ↓
provider implementation
```

while still permitting:

```text
kernel
  ↓
hardware-specific intrinsic
```

for frontier hardware.

Tensor should initially reuse this architecture rather than attempt to generalize it further.

---

# 9. Provider Model

Hardware support should eventually be independently extensible.

Conceptual provider interface:

```text
provider.init()

provider.devices()

provider.capabilities(device)

provider.allocate(device, descriptor)

provider.compile(program, target)

provider.load(artifact)

provider.launch(
    executable,
    arguments,
    stream
)

provider.copy(...)

provider.create_event()
provider.record_event(...)
provider.wait_event(...)
```

The provider owns vendor-specific mechanisms.

For example:

```text
Tensor                       NVIDIA provider

allocate(size, usage)   →    cudaMallocAsync / VMM / ...
launch(...)             →    CUDA driver launch
event(...)              →    CUDA event
compile(...)            →    PTX / cubin path
```

Tensor does not need to expose those NVIDIA-specific details in the common runtime ABI.

---

# 10. Provider Capabilities

Provider selection should be based primarily on capabilities rather than vendor names.

Example:

```text
matrix_multiply
bf16
fp8
async_copy
remote_memory
remote_atomic
peer_access
hardware_collective
gpu_initiated_transfer
```

A future architecture should be able to implement Tensor without pretending to be CUDA-compatible.

Target-specific code may still inspect finer-grained hardware properties where required.

---

# 11. Tensor Modules

Compiled tensor programs should be first-class reusable modules.

Example package:

```text
flash-attention/
├── tensor.json
├── src/
│   └── attention.py
└── artifacts/
    ├── portable/
    │   └── attention.ir
    ├── sm100/
    │   └── attention.tbin
    └── gfx950/
        └── attention.tbin
```

Possible manifest:

```json
{
  "name": "flash-attention",
  "version": "0.1.0",
  "tensorAbi": 1,
  "exports": {
    "attention": "./src/attention.py"
  },
  "capabilities": [
    "matrix-multiply",
    "async-copy"
  ]
}
```

A module may contain:

1. source,
2. portable compiler representation,
3. scheduled/tiled representation,
4. hardware-specific executables,
5. autotuning metadata.

The runtime chooses the best available artifact.

Conceptually:

```text
exact target binary?
       │
      yes ─────────→ load
       │
      no
       ↓
compatible compiled artifact?
       │
      yes ─────────→ load
       │
      no
       ↓
portable representation?
       │
      yes
       ↓
compile locally
       ↓
cache
       ↓
execute
```

---

# 12. Fusible vs Opaque Modules

Not every module export should necessarily be an opaque binary.

Tensor should eventually distinguish between:

## Fusible exports

Contain compiler-visible tensor/tile representation.

```text
module A ─┐
          ├→ combined compiler graph → fused executable
module B ─┘
```

These can be specialized, fused, or rescheduled.

## Opaque exports

Contain a target-native executable.

```text
module
   ↓
precompiled entrypoint
```

These load quickly but form an optimization boundary.

This distinction allows Tensor packages to behave more like language modules than conventional shared libraries.

---

# 13. Framework Integration

## 13.1 PyTorch

PyTorch should be treated as a frontend/runtime client rather than something Tensor attempts to replace.

`torch.compile` currently supports custom backends with a contract roughly equivalent to:

```python
backend(
    gm: torch.fx.GraphModule,
    example_inputs
) -> Callable
```

TorchDynamo performs graph capture and invokes the backend on captured FX graphs. AOTAutograd can additionally provide forward and backward graphs over a reduced core Aten operator set.

A Tensor backend could therefore conceptually be:

```python
def tensor_backend(gm, example_inputs):
    return tensor.compile_fx(gm, example_inputs)
```

PyTorch continues owning:

```text
Python execution
graph capture
guards
graph breaks
autograd orchestration
nn.Module semantics
```

Tensor owns:

```text
captured tensor fragment
        ↓
compile
        ↓
device executable
```

---

# 14. Graph Breaks

Tensor must not require the entire application to be representable.

Example:

```text
Python
  ↓
captured graph A
  ↓
Tensor executable A
  ↓
Python graph break
  ↓
captured graph B
  ↓
Tensor executable B
```

The compiler only sees A and B.

The host framework owns the gap.

This means the fundamental compiled unit is:

> A callable tensor fragment.

Not necessarily:

> A complete model.

Standalone whole-program compilation may exist separately for deployment scenarios.

---

# 15. Runtime Streams and Events

Compiled fragments must support externally owned execution streams.

A minimal execution interface should resemble:

```text
run(
    executable,
    tensors,
    stream
) -> event
```

Tensor must avoid implicit global synchronization whenever possible.

This is important for:

- framework interoperability,
- asynchronous execution,
- input/output overlap,
- future communication overlap,
- multi-stream scheduling,
- RDMA integration.

---

# 16. Future Communication Support

Distributed communication is explicitly **not part of the MVP**.

However, the runtime ABI should avoid decisions that make future integration impossible.

Tensor should eventually be able to represent a small set of communication-relevant semantics:

```text
address spaces
asynchronous operations
events
signals
waits
memory ordering
```

For example:

```text
event = async_copy(remote, local)

compute(...)

wait(event)
```

The compiler's concern is dependency and latency overlap.

It does not need to know whether the transfer uses:

```text
NVLink
GPUDirect RDMA
InfiniBand
RoCE
NVSHMEM
UCX
future coherent fabric
```

Those mechanisms should belong to runtime or infrastructure providers.

High-level operations such as:

```text
all_reduce
all_gather
reduce_scatter
MoE dispatch
```

should initially remain library/runtime functionality rather than core compiler operations.

---

# 17. Caching

Fast compilation alone is insufficient.

Tensor should treat compilation artifacts as persistent reusable assets.

Example:

```text
~/.tensor/cache/
    modules/
    kernels/
    tuning/
    providers/
```

Cache keys may include:

```text
program hash
compiler version
provider ABI
hardware architecture
shape specialization
dtype
compiler options
```

Cold-start compile latency and warm-start module loading should both be explicit performance metrics.

---

# 18. Autotuning

Tensor should not initially attempt to build a globally intelligent scheduling system.

Instead:

```text
reasonable compiler defaults
        +
explicit schedule controls
        +
bounded autotuning
```

A tunable kernel may expose parameters such as:

```text
tile M
tile N
tile K
pipeline stages
execution group size
```

Tensor can:

```text
generate candidates
compile candidates
benchmark candidates
cache winner
```

This keeps scheduling policy outside the essential compiler architecture.

---

# 19. Diagnostics and Inspection

Compiler observability is a first-class feature.

Commands should eventually include:

```bash
tensor inspect kernel.py

tensor inspect kernel.py --stage frontend

tensor inspect kernel.py --stage tirx

tensor inspect kernel.py --stage target

tensor bench kernel.py --verbose
```

The user should be able to understand:

```text
what was compiled
what was fused
what was specialized
which provider was selected
which artifact was loaded
why recompilation occurred
which capabilities were required
```

TileLang already carries source locations into TIRx diagnostics and provides pass/lowering inspection tooling, giving Tensor an existing foundation to build upon rather than starting from nothing.

---

# 20. Non-Goals for Version 1

Version 1 should explicitly not attempt to provide:

- a replacement for PyTorch,
- arbitrary Python compilation,
- a new general-purpose programming language,
- distributed training,
- cluster scheduling,
- RDMA implementation,
- fault-tolerant execution,
- universal graph optimization,
- automatic globally optimal schedules,
- replacement GPU drivers,
- replacement CUDA/ROCm/Metal driver stacks,
- support for every accelerator,
- a new universal compiler IR.

These exclusions are features of the design, not missing ambitions.

---

# 21. Proposed MVP

The smallest meaningful MVP should prove the architecture rather than the ecosystem.

## MVP target

**One executable, one frontend, one hardware family.**

For example:

```text
tensor executable
      +
TileLang-compatible source
      +
TIRx compilation
      +
NVIDIA provider
```

Required commands:

```text
tensor run kernel.py
tensor build kernel.py
tensor bench kernel.py
tensor inspect kernel.py
tensor doctor
```

Required workloads:

```text
elementwise fusion
GEMM
reduction
FlashAttention-like kernel
irregular gather/scatter workload
```

The MVP succeeds if these workloads can be compiled and executed without users manually assembling the underlying TVM/TileLang build environment.

---

# 22. Phase Plan

## Phase 0 — Architecture validation

Goal:

Determine whether Tensor needs a new compiler core at all.

Tasks:

- build several representative TileLang kernels,
- inspect their TIRx representations,
- trace lowering to target code,
- measure compile latency,
- identify required runtime dependencies,
- prototype loading and calling compiled artifacts,
- document which functionality belongs to TileLang, TIRx, TVM runtime, and vendor tooling.

Deliverable:

```text
architecture decision record
+
benchmark suite
+
minimal compilation prototype
```

---

## Phase 1 — Single-device CLI

Implement:

```text
tensor run
tensor build
tensor bench
tensor inspect
tensor doctor
```

Initial target:

```text
NVIDIA
```

Focus metrics:

```text
installation complexity
cold compile latency
warm startup latency
runtime performance
diagnostic quality
artifact portability
```

---

## Phase 2 — Stable runtime ABI

Separate:

```text
compiler
runtime
provider
```

Define:

```text
tensor descriptor
executable descriptor
stream abstraction
event abstraction
workspace contract
provider capability model
```

The runtime ABI should be language-neutral enough for Python, Rust, C++, and other hosts.

---

## Phase 3 — Module system

Introduce:

```text
tensor.json
module resolver
artifact cache
exports
versioned Tensor ABI
```

Possible commands:

```text
tensor add
tensor install
tensor publish
```

This phase turns Tensor from a compiler tool into an ecosystem substrate.

---

## Phase 4 — PyTorch backend

Implement:

```python
torch.compile(..., backend="tensor")
```

Start with inference-oriented FX graphs.

Then evaluate AOTAutograd for training compilation.

Key requirement:

Graph breaks must continue to work because PyTorch remains responsible for orchestration between compiled fragments.

---

## Phase 5 — Second provider

Add either:

```text
AMD
```

or:

```text
Metal
```

This is the first real test of whether the provider ABI and portable compiler assumptions are valid.

A provider design that appears elegant with one hardware vendor should be considered provisional.

---

## Phase 6 — Ecosystem experimentation

Only after the core works should Tensor explore:

```text
distributed communication providers
framework adapters
remote memory
package registry
native Tensor frontend
JAX integration
higher-level graph compiler
autotuning services
```

---

# 23. Success Criteria

Tensor should optimize for more than generated-kernel throughput.

Primary engineering metrics should include:

## Developer experience

```text
steps from download to first kernel
number of manually installed dependencies
quality of diagnostics
predictability of caching
```

## Compiler performance

```text
cold compilation latency
incremental compilation latency
peak compiler memory
artifact size
```

## Runtime performance

```text
kernel performance vs baseline
launch overhead
warm module startup
memory overhead
```

## Architecture quality

```text
LOC outside providers
effort to add provider
effort to expose new hardware intrinsic
effort to add frontend
number of vendor-specific concepts leaking into core ABI
```

A particularly useful objective would be:

> Achieve near-specialized kernel performance without requiring specialized toolchain complexity from the user.

---

# 24. Key Architectural Risks

## Risk 1: TileLang compatibility becomes Python compatibility

Mitigation:

Define a precise kernel compatibility profile.

Do not execute arbitrary Python as part of the compiler contract.

---

## Risk 2: TIRx/TVM dependency makes the binary distribution difficult

Mitigation:

Treat initial bundling work as an experiment.

The product contract should not depend on how compilation is internally packaged.

If required, Tensor can progressively replace or isolate expensive dependencies.

---

## Risk 3: Provider ABI becomes CUDA-shaped

Mitigation:

Do not stabilize provider ABI until at least two substantially different hardware implementations exist.

Prefer semantic properties and capability queries over vendor terminology.

---

## Risk 4: Portable module IR becomes permanently coupled to TIRx

Mitigation:

Version Tensor Module ABI independently.

Treat embedded compiler representation as an implementation-specific artifact.

---

## Risk 5: Compiler scope expands into infrastructure

Mitigation:

Maintain the rule:

> Tensor compiles and executes tensor fragments. Infrastructure manages machines.

Distributed systems should initially integrate through runtime/library interfaces.

---

## Risk 6: Performance requires increasingly complicated mandatory compiler passes

Mitigation:

Keep expert scheduling and target intrinsics available.

Make sophisticated automation optional layers.

Follow the TIRx approach of allowing frontier features before universal abstractions exist.

---

# 25. Open Questions

The following should be answered experimentally rather than architecturally guessed.

### Compiler packaging

Can TileLang/TIRx compilation realistically be distributed behind a low-friction `tensor` executable?

### Artifact representation

What is the minimum representation required for a module to remain fusible and re-specializable?

### Provider boundary

Which responsibilities belong to Tensor runtime versus hardware provider?

### Shape polymorphism

How much symbolic-shape compilation should Tensor support before specializing?

### Compilation latency

Can the selected TIRx/TVM path meet the desired interactive compile-time budget?

### Autotuning

What amount of tuning provides meaningful gains without turning first-run compilation into a long optimization process?

### PyTorch lowering

Should Tensor initially consume FX directly, core Aten graphs through AOTAutograd, or another normalized representation?

### Package ABI

How should packages communicate:

```text
tensor signatures
shape constraints
required capabilities
side effects
workspace
mutability
aliasing
```

without exposing compiler internals?

---

# 26. Initial Research Benchmark

Before implementation, the same workload set should be studied across:

```text
TileLang
TIRx
Triton
CuTe DSL
tinygrad
Pallas where applicable
```

Workloads:

```text
1. fused elementwise
2. tiled GEMM
3. FlashAttention
4. gather/reduction
5. dynamic-shape operation
6. synthetic new-hardware primitive
```

Measure:

```text
source LOC
compile latency
runtime performance
IR complexity
number of lowering stages
dependency footprint
quality of diagnostics
ease of adding a hardware operation
ease of adding a memory space
ease of adding a backend
```

The purpose is not to select a winner solely on kernel performance.

It is to identify the smallest architecture that provides sufficient control and extensibility.

---

# 27. Long-Term Vision

If the architecture succeeds, Tensor could eventually look like this:

```text
                     Tensor ecosystem

                          registry
                             │
               ┌─────────────┴─────────────┐
               │                           │
          tensor modules               providers
               │                           │
     ┌─────────┼──────────┐                │
     │         │          │                │
 PyTorch    TileLang    native             │
     │         │          │                │
     └─────────┴────┬─────┘                │
                    ↓                      │
              Tensor compiler             │
                    │                      │
              Tensor runtime ABI ─────────┘
                    │
                    ↓
                 hardware
```

Users interact primarily with:

```text
tensor
```

Frameworks interact with a compiler/runtime ABI.

Kernel authors may use TileLang or future frontends.

Library authors publish reusable tensor modules.

Hardware vendors or the community provide provider implementations.

Infrastructure systems remain free to build distributed execution, serving, training, RDMA integration, and cluster orchestration around the same compiled tensor fragments.

The intended outcome is not that heterogeneous accelerator computing becomes simple internally.

The intended outcome is that **its complexity becomes localized behind stable boundaries**.

---

# 28. Proposal

Proceed with **Phase 0: Architecture Validation** before designing a new compiler or native language.

The initial hypothesis to test is:

> Tensor can provide a Bun-like developer experience for tensor computation by combining a TileLang-compatible programming interface, TIRx-based compilation, a small runtime/provider boundary, and first-class compiled tensor modules.

The most important early question is therefore not:

> How should we design `tensorc`?

It is:

> How much of `tensorc` already exists in TileLang and TIRx, and what minimal layer is actually missing between those systems and the developer experience we want?

If the answer is that TileLang/TIRx already provides the necessary compiler substrate, Tensor should concentrate its engineering effort on the runtime, module system, packaging, provider interfaces, caching, framework adapters, and tooling.

If those systems expose fundamental architectural limitations, Tensor will then have concrete evidence for which compiler components need replacement.

This keeps the project experimental, incremental, and grounded in existing compiler research while preserving the larger goal:

> **One coherent environment for building and running high-performance tensor software across heterogeneous hardware.**
