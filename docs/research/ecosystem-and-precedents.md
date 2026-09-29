# Ecosystem Survey and Packaging Precedents

**Verified:** 2026-09-29 · Companion to [`phase0-ground-truth.md`](phase0-ground-truth.md)

Covers proposal §26 (the cross-toolchain benchmark) and §25's packaging questions. Numbers
come from live PyPI JSON and upstream sources. Claims that could not be verified are marked
**UNVERIFIED**.

---

## Part A — The §26 comparison, at a glance

| | Triton | CuTe DSL | tinygrad | JAX / Pallas | TileLang / TIRx |
|---|---|---|---|---|---|
| Latest | 3.8.0 | 4.8.0 | 0.14.0 | jax/jaxlib 0.11.2 | tilelang 0.1.14, tvm 0.27.0 |
| Runtime deps | **none** | `nvidia-cutlass-dsl-libs-*` pinned | **none** | jaxlib, numpy, scipy | `apache-tvm-ffi`, `torch-c-dlpack-ext`, **torch** |
| x86_64 wheel | 248 MB | 92 MB (libs) | 2.8 MB (pure Python) | jaxlib 90 MB | tilelang 45 MB |
| **Windows wheel** | **none in 3.8.0** | yes (76 MB) | yes (trivially) | yes | **yes** (27 MB) |
| Compile stages | **5** (TTIR→TTGIR→GLIR→LLVM IR→PTX→CUBIN) | UNVERIFIED | **2** | Jaxpr→Triton/MGPU IR→PTX | **~63 passes** (measured) |
| Artifact | AOT C source with embedded cubin | AOT object/header export, C/C++ and TVM FFI integration | **none** | Python only | source emission; P0 adds an experimental cubin envelope |
| New backend | `BaseBackend` + `add_stages` | UNVERIFIED | drop `ops_<x>.py` | lowering-rule namespace | `register_backend(BackendModule)` |
| New memory space | dialect-level, needs C++ | UNVERIFIED | tied to device | explicit `plgpu.SMEM` scratch | `target.attrs["keys"]` / `register_tag` |

**Two findings that change Tensor's positioning:**

1. **TileLang is one of very few in this set with a first-class Windows wheel.** Triton
   3.8.0 ships no `win_amd64` wheel at all. For a project whose first user is on Windows,
   that is a differentiator worth stating plainly — and a reason the local no-GPU dev loop
   was even possible.

2. **TileLang can already target CuTe DSL.** The `cutedsl` backend is registered in the
   TileLang backend manifest with its own pipeline and codegen. §26 asks to compare against
   CuTe DSL; that comparison can be run *from inside TileLang* rather than as a separate
   integration. Cheap, and it directly informs the "ease of adding a backend" metric.

### Who has actually solved "wrap a kernel compiler behind a friendly binary"?

Several systems solve parts of this workflow. The initial survey missed Triton
and CuTe DSL's AOT paths; compiled artifact export is not an unoccupied gap.

- **CuTe DSL** exports object files and C headers, supports C/C++ loading, and
  offers a TVM FFI ABI. See NVIDIA's [AOT guide](https://docs.nvidia.com/cutlass/4.5.2/media/docs/pythonDSL/cute_dsl_general/dsl_ahead_of_time_compilation.html).
- **Triton** has an AOT tool emitting C source with embedded cubin data and
  load/unload/launch utilities. See the upstream [compiler tool](https://github.com/triton-lang/triton/blob/main/python/triton/tools/compile.py).

- **`uv` / `ruff`** solved *distribution*. `ruff` ships a 10.6 MB `py3-none-any` wheel with
  `requires_dist: null` — one static binary, zero Python. But it works only because there is
  no Python in the loop. Tensor does have one.
- **AOTInductor** solved the *artifact container*. `torch._inductor.aot_compile` with
  `package_aoti(...)` produces a `.pt2` package; `load_package(...)` and
  `AOTIModelPackageLoader` load it. But it is a container of serialized binaries, drags in
  all of LibTorch, and is target-specific rather than portable.
- **Triton** solved the *plugin contract*. `BaseBackend.add_stages` + a `Language` enum is a
  real versioned compiler-plugin interface, and `get_cache_key(src, backend, options, env)`
  gives correct invalidation for free. It is a wheel, not an executable.

**Tensor's hypothesis is an integrated developer workflow:** consistent setup,
diagnostics, build/run commands, cache behavior and independently loadable
artifacts. It needs comparison against these existing AOT workflows. A binary
export path or one CLI alone does not establish a differentiated product.

---

## Part B — Precedents worth copying

### B1. Plugin registration: use an entry-point group, not an import path

Dynamo discovers backends through the setuptools entry-point group
**`torch_dynamo_backends`** (`torch/_dynamo/backends/registry.py`). A downstream product
ships a tiny wheel declaring the entry point and plugs in with **zero patching of torch**.
That is the mechanism Tensor should mirror — a `tensor_providers` entry-point group.

TVM's `tvm.target.register_tag(name, config)` + `list_tags()` is the cleaner, simpler
precedent for *describing hardware* specifically, and it is already available.

Contrast: tinygrad discovers backends by `importlib.import_module(f"{base}.runtime.ops_{x}")`
plus an `inspect.getmembers` name match. Simple, but it is convention, not a contract, and
it gives you no way to describe capabilities.

**Do not model the backend contract on Inductor's.** `torch._inductor.compile_fx` is a
private module and its docstring warns *"This function TAKES OWNERSHIP of the input
`model_` and can potentially mutate it!"* An ownership-transfer footgun in a public contract
is a design smell to avoid copying.

### B2. Version the format, and keep it separate from the artifact

The clearest precedent available is already a dependency: `apache-tvm-ffi` is a separate
package with its own semver line, and `include/tvm/ffi/c_api.h` exports
`TVMFFIGetVersion()` returning `{major, minor, patch}` — three parts, queryable at runtime,
decoupled from the Python wheel.

DLPack supplies the semantics to copy: **major = breaking layout change, minor = additive**,
with consumers required to check major. MLIR bytecode and GGUF both additionally keep the
format *append-only* so unknown keys are skipped by old readers.

The failure mode to avoid is AOTInductor's: a `.pt2` container with no version field at all.

**Concrete for Tensor:** `tensor.json` should carry a format version distinct from the
package version, and the format must be append-only. §25 asks exactly this ("How should
packages communicate ... without exposing compiler internals?") and the answer is a versioned
manifest beside a versioned IR — not a version of the IR's own name.

### B3. The smallest honest runtime vocabulary

Every device runtime can express these seven things, and no eighth is portable:

1. device / context
2. buffer + buffer view (strides, not byte counts)
3. queue / stream
4. event / fence — the only cross-stream primitive
5. host-visible vs device-only memory, plus one async copy between them
6. kernel launch (function handle, grid, packed args, shared-memory size)
7. graph / batch submit — an optimization, not a primitive

**Where they genuinely diverge** — this is the part that decides whether §15's "avoid
implicit global synchronization" is achievable:

| Seam | CUDA | Metal | Vulkan | WebGPU |
|---|---|---|---|---|
| Workgroup sync | `__syncthreads()` (one op) | `threadgroup_barrier(mem_flags)` (one op) | `subgroupBarrier()` **+** `memoryBarrierBuffer` (two ops) | `workgroupBarrier()` (one op) |
| Command encoding | stream, implicit ordering | explicit `CommandBuffer`/`Encoder` objects | command buffer + secondary/bundled | single encoder + submit |
| Dispatch shape | grid × block × thread | `dispatchThreads` / `dispatchThreadgroups` | `dispatch(x,y,z)` | `dispatchWorkgroups(x,y,z)` |
| Dynamic smem sizing | kernel attribute | `threadgroupMemoryLength` | dynamic offset | `workgroup_size` + max storage |

Two seams are irreducible: **barrier vs memory-barrier as separate operations** (Vulkan,
WebGPU) versus one combined op (CUDA, Metal), and **Metal's explicit encoder objects**
versus stream-as-ordering-primitive.

The proposal's §16 list — address spaces, async operations, events, signals, waits, memory
ordering — maps cleanly onto items 3–6 above. The finding to carry into Phase 2: the
vocabulary is small, but the *sync semantics* are where a vendor-neutral ABI will leak.
Design that seam explicitly rather than discovering it late.

### B4. Bound recompilation instead of minimizing it

The most transferable number in the survey: Dynamo treats recompilation as a **budget to be
bounded**, not a cost to be eliminated — `recompile_limit = 8`,
`accumulated_recompile_limit = 256`, `fail_on_recompile_limit_hit = False`
(`torch/_dynamo/config.py`). It degrades observably instead of stalling silently.

Triton's cache key is `(src, backend, options, env_vars)`, with an optional
`knobs.runtime.add_stages_inspection_hook` folded in. Any Tensor cache must treat
*target + provider + options + backend version* as key components from v1.

**UNVERIFIED:** no primary source was found publishing per-stage milliseconds for Dynamo,
AOTAutograd, Triton codegen, JAX `jit`, TVM, or TileLang. Rather than cite numbers nobody
stands behind, Phase 0 should *publish Tensor's own* per-stage timings as the §26 benchmark
output — which the harness already does.

---

## Top 5 lessons for Tensor's design

1. **Provider registration via a `tensor_providers` entry-point group**, mirroring
   `torch_dynamo_backends` — never via import-path convention. It is the only mechanism
   observed that lets a downstream product plug in with zero patching.
2. **Separate format version from artifact version**, `{major, minor, patch}`, queryable at
   runtime, append-only — copying `TVMFFIVersion` + DLPack's major/minor split exactly.
3. **Ship a portable *fusible* tier above the fast tier.** Every precedent either has a
   portable IR (TIRx, ExportedProgram, MLIR) or a fast artifact (`.so`, `.cubin`, `.pt2`) —
   nobody ships only the fast one and stays extensible. This is the argument for E4, and it
   cuts both ways: if serialized TIRx cannot be re-lowered, Tensor must own the portable IR.
4. **Bound recompilation explicitly** — a per-graph budget with observable, non-fatal
   degradation, not an unbounded retry loop.
5. **Assume the CLI will be Rust/Go with a Python sidecar resolved at runtime**, and that the
   sidecar has its own version line (the `apache-tvm-ffi` / `nvidia-cutlass-dsl-libs-*`
   split). Pre-empt two failure modes: transitive version pinning that breaks out-of-tree
   use, and a Windows story where "wheels exist but the toolchain doesn't".

---

## One counter-finding worth flagging

TIRx's own intrinsic extension is `TVM_DECLARE_INTRIN_UNARY/BINARY` macros over
`Op::Get("tirx.<name>")` — **C++ macros, not Python-callable**. So while TileLang's
`register_dispatch` gives a comfortable Python path for *tile primitives*, TIRx's raw
intrinsic registration is C++-only.

That sharpens rather than contradicts the proposal: §3.6's ladder (raw intrinsic → target
capability → reusable operation → portable primitive) is real, but the bottom rung is
currently C++ only. Anyone planning to add a genuinely new hardware feature needs a C++
build, which is a much heavier ask than "extend a Python provider" and belongs in the
provider-model cost estimate.

---

## Sources

- PyPI JSON: `triton`, `tinygrad`, `nvidia-cutlass-dsl(-libs-cu12)`, `jax`, `jaxlib`,
  `apache-tvm`, `tilelang`, `apache-tvm-ffi`, `torch`, `torch-c-dlpack-ext`, `ruff`, `uv`
- `triton/python/triton/compiler/compiler.py`, `python/triton/backends/compiler.py`,
  `third_party/nvidia/backend/compiler.py`
- `tinygrad/{uop/ops.py, renderer/__init__.py, device.py}`
- `pytorch/torch/_dynamo/{backends/registry.py, config.py}`,
  `pytorch/torch/_inductor/compile_fx.py`
- `jax/jax/_src/pallas/{triton,mosaic_gpu}/lowering.py`
- `apache/tvm/{include/tvm/tirx/op.h, include/tvm/ffi/c_api.h, python/tvm/target/target.py,
  python/tvm/target/tag_registry/registry.py, python/tvm/runtime/module.py}`
- `dmlc/dlpack/include/dlpack/dlpack.h` · `mlir/docs/BytecodeFormat.md` · `ggml/docs/gguf.md`
