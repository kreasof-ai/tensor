# Tensor — Roadmap

Derived from proposal §22, reordered by what the Phase 0 evidence actually supports.
The proposal's phase list is sound; the changes below are about *sequencing* and about
being explicit about which work is blocked on hardware.

Current status: **Phases 0–3 complete within their measured profiles.**
[ADR 0011](../adr/0011-runtime-call-abi-and-nvrtc.md) selects NVRTC as the default
CUDA compiler and the independent v3 module envelope.
[ADR 0012](../adr/0012-phase2-executable-and-workspace-contract.md) completes
runtime ABI 1.1 with executable/event identities, lifetime rules and zero
external workspace. CUDA and a CPU validation provider share the workbench;
native C++/Rust hosts exercise the descriptors without Python or TVM FFI.
A driver/toolchain-free container builds all five NVRTC profiles. Fresh remote
Linux and Windows CI artifacts and their wheels each pass GPU execution on a
separate A10G with only Tensor and NumPy. The full GPU suite passes 89 tests
with zero skips. Details and limits are in
[the Phase 2 exit report](../research/phase2-exit.md).

Phase 3 adds the offline module system: manifests, pinned closures, deterministic
packages, verified caches and named exports. Linux and Windows CI packages each
execute through installed exports on the separate A10G without compiler imports.
The full GPU suite passes 104 tests with zero skips. See
[ADR 0013](../adr/0013-phase3-offline-module-system.md), the
[module guide](../modules.md) and [Phase 3 exit report](../research/phase3-exit.md).
The [PyPI transport adapter](../adr/0014-pypi-module-transport.md) extends that
profile with publishing and exact-version registry retrieval, retaining the
offline package format and existing runtime compatibility checks.
Its [acceptance report](../research/phase3-pypi.md) records 116 GPU-enabled tests,
Linux/Windows CI and execution of both producers' registry packages on A10G.

Phase 1 baseline: `doctor`, `build`,
`inspect`, `run`, `bench`, cache inspection and the CUDA workbench are validated
on the A10G. Product v2 artifacts support typed scalar arguments and symbolic
dimensions. CPU/GPU DLPack imports and foreign CUDA stream ordering work, with
primary-context and resource ownership preserved. A NumPy-only consumer passes
22 numerical/compatibility cases with transferred product artifacts from a
GPU-free GitHub Actions host. Installation and inspection pass on Ubuntu and
Windows runners. The full GPU-enabled suite passes 72 tests with zero skips.
Phase 1 scope, installation counts, latency, diagnostics and transfer evidence
are recorded in [the exit report](../research/phase1-exit.md) and accepted by
[ADR 0010](../adr/0010-complete-phase1-cuda-cli.md). GPU execution beyond A10G
remains outside the measured single-device scope. Two-host Phase 0 opaque
executable transfer passes from GitHub Actions to an A10G. Cache behavior,
independent CPU provider execution, C++/Rust hosting, bounded composition,
symbolic dimensions, frontend contracts, foreign CUDA stream ordering, full
compilation and A10G baselines are measured in
[E15](../research/e15-phase0-validation.md) and
[E16](../research/e16-phase0-exit.md). Negative results and deferred hardware
scope remain explicit.

---

## What Phase 0 has already changed about the plan

Four findings move work earlier or change its shape:

1. **Codegen research is not blocked on the GPU.** Source emission, IR inspection, pass
   counting, diagnostics, and packaging all run on a no-GPU machine. The proposal assumed
   Phase 0 needed a target; it does not. *Effect: start the module-system experiments now,
   in parallel, not after the GPU box is set up.*

2. **Inspection, backend registration, and several §26 comparison targets already exist.**
   `lower_trace` gives IR at every pass; `register_backend` gives a provider manifest; the
   `cutedsl` backend means CuTe DSL is reachable from TileLang. *Effect: do not build a
   competing IR dumper or a competing backend registry. Build the product surface over them.*

3. **E4 demonstrated re-lowerable frontend IR** within a pinned toolchain and
   across CUDA architectures. Versioning and cache identity can now be designed
   against a real artifact. Fusion and cross-provider portability remain open.

4. **A torch-free compiler is a candidate.** `tvm_compiler.dll` and `tvm_runtime.dll` have *no*
   torch in their PE import tables — the dependency is entirely Python-layer, and traces to
   one `import torch  # preload torch to avoid dlopen errors` line. PyTorch support therefore
   ships as a client-side adapter (ADR 0005). The current producer still imports
   PyTorch; the opaque executable consumer has now run without it on NVIDIA.

---

## Phase 0 — Architecture validation *(complete)*

**Goal:** answer §28's question — how much of `tensorc` already exists, and what layer is missing?

| # | Experiment | Where | Status |
|---|---|---|---|
| E1 | environment / packaging footprint | local | ✅ |
| E2 | codegen — source emission per kernel × target | local | ✅ 15/15 |
| E3 | pass trace — IR at every lowering stage | local | ✅ 5/5 |
| E4 | artifact shape — serialization and CUDA re-targeting | local / NVIDIA | ✅ frontend reuse and bounded pointwise composition; general fusion deferred |
| E4a | artifact versioning across TIRx versions | local | ✅ **answered** |
| E4b | symbolic-shape source signature / serialization | NVIDIA | runtime dimensions pass; symbolic GEMM tile rejected |
| E5 | cache behaviour — what TileLang already does | local / NVIDIA | measured memory/disk hits, invalidation, corruption recovery, workload differences |
| E6 | provider contract — register a second backend | local | ✅ independent `p0_cpu` manifest compiles and executes on CPU |
| E7 | ABI surface — non-Python host feasibility | local / NVIDIA | ✅ C++/Rust hosting and CUDA foreign-stream ordering; native compiler packaging deferred |
| E8 | framework contract — FX vs AOTAutograd | either | FX-to-CPU kernel passes; AOT forward/backward captured and reference-executed |
| E9 | full compile latency (with `nvcc`) | **NVIDIA** | five workloads × six targets measured; compiler rejections recorded |
| E10 | numerics validation | **NVIDIA** | ✅ 12 NumPy-reference cases on A10G |
| E11 | kernel performance | **NVIDIA** | ✅ five baselines and matched static/symbolic comparisons on A10G; cross-GPU matrix deferred |
| E12 | warm-start / artifact load latency | **NVIDIA** | five sizes × three processes; Actions-to-A10G executable transfer passes |

**E4 came back positive and it changes the shape of the project.** A serialized TIRx module
round-trips byte-identically, re-targets across architectures, and reloads in a **fresh
interpreter** — 50 KB of JSON reproduces the same CUDA as the in-memory module.
Tensor need not own a new IR for the measured source tier. This does not establish
fusion. See [`../research/e4-artifact-shape.md`](../research/e4-artifact-shape.md)
and the post-lowering restriction in [E15](../research/e15-phase0-validation.md).

Three consequences:

1. **Frontend IR storage can reuse TIRx serialization.** E16 demonstrates
   bounded pointwise composition and consumer re-scheduling. General fusion
   remains separate work in Phase 3.
2. **Phase 1's `tensor build` is implementable now**, not speculative.
3. **The new sharpest risk is versioning, not representation** — and E4a has now measured
   it. Artifacts do not cross a TileLang version boundary in either direction, and carry no
   usable version metadata of their own. But the compatibility surface is four statically
   extractable op names, so the gate is cheap and precise. See
   [`../research/e4a-artifact-versioning.md`](../research/e4a-artifact-versioning.md).

Static shapes are baked at the frontend. E4b observes a runtime dimension
parameter for a symbolic extent; E15 executes one compiled elementwise artifact
at five extents and one static-tile GEMM at five row extents.
Cache keys must distinguish static specialization from symbolic dimensions.

**Exit criteria:** E4 answered, E6 passed with a non-CUDA backend, E9 measured, and an
architecture decision record written for each of the four load-bearing choices
([`../adr/`](../adr/)).

E9 and the architectural decisions are recorded in
[ADR 0008](../adr/0008-phase0-evidence-boundaries.md). E16 then registers an
independent CPU manifest, composes serialized frontend stages, exercises Rust
and foreign CUDA streams, and measures static-versus-symbolic throughput.
[ADR 0009](../adr/0009-complete-phase0-with-scoped-provider-and-composition.md)
accepts those scoped contracts and closes the phase.

**Scope update — 2026-09-29:** at the user's request, cross-GPU benchmarking
is deferred and does not block Phase 0 completion. Performance validation uses
the available A10G, including the completed static-versus-symbolic comparison.
Source emission and full compilation across targets remain measured evidence;
they do not establish execution or performance on those other GPUs. Revisit
cross-GPU benchmarking before making performance claims across architectures.

---

## Phase 1 — Single-device CLI *(complete)*

**Goal:** the §5 executable, on the hardware that exists.

Sequence:

1. `tensor doctor` **implemented**. It is the cheapest honest probe of the packaging story, and every
   later command depends on the same environment/registry/target-detection logic it needs.
   It is also the command most likely to be run by a confused user, so it is the right
   thing to get right early.
2. `tensor build` **typed CUDA profile implemented** — the artifact layer that
   does not exist yet in TileLang (§3 of ground truth: no `export_library` equivalent).
   It produces a cubin and versioned TIRx envelope from one explicit export,
   with typed scalar arguments, symbolic dimensions, launch expressions,
   pointer alignment and a content-addressed cubin cache. Product artifact
   transfer across hosts passes for five workload profiles.
3. **The prototyping surface (ADR 0006), CUDA implementation** — device buffers, launch, `assert_close`, `bench`,
   NumPy/DLPack interop. Pulled forward out of Phase 6 because it needs only *opaque*
   executables, so it is **not gated on E4**, and because it is the harness E9–E12 run on.
   Sessions retain the primary context and own or borrow their stream. CPU
   DLPack upload, GPU DLPack borrowing, managed-tensor lifetime and foreign
   stream waits/handoffs are implemented and tested.
4. `tensor inspect` **implemented** — frontend TIRx, target CUDA source and
   per-pass traces use existing TileLang hooks; artifact manifests are checked
   without compiler imports.
5. `tensor run` and `tensor bench` **implemented for typed CUDA artifacts** —
   named `.npy` inputs, `--scalar` values, runtime dimension bindings, declared
   outputs, and a compiler-free consumer. Existing v1 static artifacts pass.

**Measured exit criteria (§23):**

| Metric | Evidence |
|---|---|
| Downloaded payloads to first kernel | Four commands, including input generation; five with artifact download |
| Application installation | Tensor wheel selected once; NumPy automatic; two installed distributions |
| Cold/warm compilation | 1.659 / 0.308 s inside command; 4.295 / 2.975 s whole process |
| Fresh consumer first result | 0.433 / 0.188 / 0.189 s after command entry |
| Diagnostics | Eight expected typed-runtime failures identify their cause; five initial CLI probes also recorded |
| Product portability | Five v2 artifacts transfer between clean matching hosts; NumPy-only consumer passes 22 cases plus CLI execution |
| Host launch overhead | Static 49.4 µs enqueue / 54.7 µs launch plus sync; dynamic 69.6 / 74.8 µs |
| Installation beyond the execution host | Clean-wheel installation and inspection pass on Ubuntu and Windows |

Timing boundaries, prerequisites and accepted profile limits are explicit in
[the exit report](../research/phase1-exit.md). Cross-GPU benchmarking is not
a gate for this single-device phase.

**Design constraints discovered so far:**

- Do not import the compiler eagerly in the CLI. The Phase 0 compiler-import
  floor was about 4.2 s; the measured product consumer avoids it.
- Handle "no device present" gracefully. `determine_target()` currently raises.
- Source bundles need both `tl_templates` and CUTLASS/CuTe headers. Cubin bundles
  contain the executable and notices, without compiler headers.
- Pin TileLang exactly. API drift inside a single 0.1.x release is already observed.

---

## Phase 2 — Stable runtime ABI *(complete)*

**Goal:** separate compiler, runtime, and provider.

Implemented: the [runtime contract](../runtime-abi.md) and packaged C header
freeze buffer, scalar, resolved-call, stream and error layouts, plus executable,
event and workspace descriptors, as runtime ABI 1.1. Executable/event tokens
are session-qualified, never reused and invalidated by release or session close.
Release waits for outstanding work. External workspace is explicitly zero;
unsupported requirements fail before loading.
Shared binding, outputs and lifetime checks serve CUDA and a synchronous CPU
provider. The module envelope is v3; legacy CUDA v1/v2 envelopes remain readable.
C++/Rust hosts execute native CPU images, and C++ executes an NVRTC cubin.

[ADR 0011](../adr/0011-runtime-call-abi-and-nvrtc.md) evaluates TVM FFI and
retains it inside the compiler, choosing Tensor descriptors for the runtime
boundary. NVRTC 12.9 is the default executable compiler; nvcc remains explicit.
The producer needs pinned Python compiler packages plus bundled libraries and
headers, but no system CUDA toolkit, driver, GPU or host C++ compiler.

All six Phase 2 contract areas from proposal §22 are defined for this profile:
tensor and executable descriptors, streams, events, workspace and capabilities.
[ADR 0012](../adr/0012-phase2-executable-and-workspace-contract.md) and the
[exit report](../research/phase2-exit.md) record acceptance.

Scope: this freezes descriptor layouts and lifetime rules; provider dispatch
remains host-owned. A C provider lifecycle/plugin table and native `.tbin` CLI
remain future interfaces. CPU is a validation implementation on Linux x86-64.
CUDA remains one exact-SM cubin. Cross-SM images/PTX fallback and optimized
non-NVIDIA providers need further work. **Direct PTX, including a possible
tinygrad lowering path, stays experimental until after Tensor v1.**

---

## Phase 3 — Module system *(offline profile complete; PyPI transport added)*

`tensor.json` schema 1 defines named exports, exact module versions, Tensor ABI
major and capabilities. `tensor.lock` pins every dependency's content identity.
The resolver accepts local directories, deterministic `.tpack` archives and
Tensor transport wheels from local files or a Python Simple Index;
packages include the complete transitive closure. Cycles, conflicting identities,
modified snapshots and stale frozen locks fail explicitly.

`add`, `install`, `pack`, `publish`, `resolve` and `module-name::export_name` references
integrate with build, inspect, run and bench. A packaged exact-provider/target
image is preferred, then a verified generated image. Missing images compile
only when explicitly requested, through bundled frontend TIRx or Python source.
Portable reuse checks frontend versions, IR/operator integrity and registered
operators before deserialization. Actual CPU recompilation and NVRTC sm_80 to
sm_86 affine/GEMM retargeting pass. Executable images instead negotiate runtime
ABI/capabilities and provider target; frontend versions remain provenance.

All five exports, dependency closure, frozen installation, relocation and
compiler-free execution pass from Linux and Windows CI producers to A10G.
[ADR 0013](../adr/0013-phase3-offline-module-system.md) defines the profile and
[the exit report](../research/phase3-exit.md) records its gates and evidence.
`publish` uses optional Twine for PyPI/TestPyPI/custom uploads. Registry `add`
pins wheel and module hashes; frozen installation can restore an empty cache
from the pinned index, while `--offline` prohibits network retrieval. See
[ADR 0014](../adr/0014-pypi-module-transport.md). Version ranges, authenticated
download indexes, fusion and autotuning remain later extensions.
NVRTC remains the default; direct PTX stays experimental after Tensor v1.

---

## Phase 4 — PyTorch backend *(complete: inference-first scope)*

`torch.compile(..., backend="tensor")`, inference-oriented FX graphs first. E8 informs the
representation choice. Keep graph breaks working — PyTorch owns the gap.

Include FP16 forward self-attention in the inference benchmark set: contiguous
BHSD, causal/non-causal, tail sequences, batching and head dimensions 64/128.
Compare with forced PyTorch FlashAttention SDPA and measure GPU execution and
host submission separately. The current
[attention demonstration](../research/flash-attention-demo.md) uses Tensor's
existing NVRTC artifacts/runtime. The separate [tensor-torch adapter](../pytorch.md)
now lowers inference FX regions for pointwise operations, rank-two FP16 GEMM
with epilogues, and this SDPA profile. Installed exports also have functional
and mutable-output custom-op registrations with symbolic FakeTensor shapes.
Unsupported regions and graph breaks remain in PyTorch, with observable coverage.
AOTAutograd evaluation executes compiled forward and backward regions, with
unsupported backward operators reported as PyTorch fallback. Full training and
attention backward remain outside the inference-first scope.

Validation includes current-stream launches, allocator lifetime, CUDA graph
capture, compiler-free cached execution, numerical and cache recovery checks,
and clean adapter wheels on Linux/Windows. Performance acceptance uses 20 A10G
inference cases: end-to-end geometric mean at most 1.10× Inductor, no case above
1.25×, selected fused graphs at least 1.25× eager, prepared submission at most
15 µs, cached preparation at most 100 ms, and cold pointwise/GEMM preparation
at most 10/30 s. All gates pass in the final 20-case run: 0.756× Inductor geometric mean,
1.116× worst case, 8.75 µs prepared submission, and 57.30 ms maximum cached
preparation. GPU execution and host submission are measured separately.
The full GPU suite passes 141 tests with zero skips, both Linux and Windows
CI pass, and their actual FX-produced profiles execute on the A10G in a clean
compiler-free consumer. Exact scope, limitations and evidence are recorded in
[the Phase 4 exit report](../research/phase4-exit.md). The compact submission shim needs Python
headers to build its wheel and no CUDA toolkit; installed consumers need no
host compiler. NVRTC remains the only automatic kernel compiler.

---

## Phase 5 — Second provider

AMD or Metal. **The first real test of the provider ABI**, and the reason E6 exists
locally with the `cpu` backend: a design that only ever met NVIDIA should be considered
provisional, and validating against a second backend is much cheaper locally than by
renting hardware.

RDNA2 specifically is not in scope: no supported TileLang path exists.

---

## Phase 6 — Ecosystem

Only after the core works. As §22 says.

The **tensor algebra and `nn` layers** discussed as a Bun-style "batteries included" surface
were explicitly *deferred*, not rejected — see ADR 0006. They are gated on E4, because
without fusible modules a from-scratch tensor library would be slower than PyTorch eager.
The prototyping surface is the version of that idea that survives a negative E4.

---

## Cross-cutting

- **Product code began after E4 and the Phase 0 exit.** `src/tensor` owns the CLI,
  artifact envelope and CUDA workbench; Phase 0 experiments remain evidence.
- **Every claim carries a number or a source.** Anything else is a guess, and §25's open
  questions were guesses for long enough.
- **Re-verify ground truth at each phase boundary.** The proposal already had one stale
  claim. Knowledge of this stack has a shelf life measured in weeks.
