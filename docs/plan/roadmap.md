# Tensor — Roadmap

Derived from proposal §22, reordered by what the Phase 0 evidence actually supports.
The proposal's phase list is sound; the changes below are about *sequencing* and about
being explicit about which work is blocked on hardware.

Current status: **Phase 0 in progress.** The active next milestone is
[opaque executable transfer and compiler-free loading](opaque-artifact-validation.md).
Local checks pass; CUDA compilation, GPU numerics and two-host execution are unverified.

---

## What Phase 0 has already changed about the plan

Three findings move work earlier or change its shape:

1. **Codegen research is not blocked on the GPU.** Source emission, IR inspection, pass
   counting, diagnostics, and packaging all run on a no-GPU machine. The proposal assumed
   Phase 0 needed a target; it does not. *Effect: start the module-system experiments now,
   in parallel, not after the GPU box is set up.*

2. **Inspection, backend registration, and several §26 comparison targets already exist.**
   `lower_trace` gives IR at every pass; `register_backend` gives a provider manifest; the
   `cutedsl` backend means CuTe DSL is reachable from TileLang. *Effect: do not build a
   competing IR dumper or a competing backend registry. Build the product surface over them.*

3. **E4 answered positively** — the portable tier exists. The open questions shifted from
   "does a portable artifact exist?" to "how is it versioned, and what does a cache key
   contain?"

4. **A torch-free compiler is feasible.** `tvm_compiler.dll` and `tvm_runtime.dll` have *no*
   torch in their PE import tables — the dependency is entirely Python-layer, and traces to
   one `import torch  # preload torch to avoid dlopen errors` line. PyTorch support therefore
   ships as a client-side adapter (ADR 0005), not as a compiler dependency.

---

## Phase 0 — Architecture validation *(in progress)*

**Goal:** answer §28's question — how much of `tensorc` already exists, and what layer is missing?

| # | Experiment | Where | Status |
|---|---|---|---|
| E1 | environment / packaging footprint | local | ✅ |
| E2 | codegen — source emission per kernel × target | local | ✅ 15/15 |
| E3 | pass trace — IR at every lowering stage | local | ✅ 5/5 |
| E4 | artifact shape — serialization and CUDA re-targeting | local | ✅ **positive; fusion untested** |
| E4a | artifact versioning across TIRx versions | local | ✅ **answered** |
| E4b | symbolic-shape source signature / serialization | local | ✅ **source-level; execution unverified** |
| E5 | cache behaviour — what TileLang already does | local | ⬜ |
| E6 | provider contract — register a second backend | local | ⬜ |
| E7 | ABI surface — non-Python host feasibility | local | ⬜ |
| E8 | framework contract — FX vs AOTAutograd | either | ⬜ |
| E9 | full compile latency (with `nvcc`) | **NVIDIA** | ⬜ |
| E10 | numerics validation | **NVIDIA** | ⬜ |
| E11 | kernel performance | **NVIDIA** | ⬜ |
| E12 | warm-start / artifact load latency | **NVIDIA** | ⬜ |

**E4 came back positive and it changes the shape of the project.** A serialized TIRx module
round-trips byte-identically, re-targets across architectures, and reloads in a **fresh
interpreter** — 50 KB of JSON reproduces the same CUDA as the in-memory module. Tensor does
**not** need to own a portable IR. See [`../research/e4-artifact-shape.md`](../research/e4-artifact-shape.md).

Three consequences:

1. **Frontend IR storage can reuse TIRx serialization.** Composition, fusion and
   re-scheduling were not tested and remain separate work in Phase 3.
2. **Phase 1's `tensor build` is implementable now**, not speculative.
3. **The new sharpest risk is versioning, not representation** — and E4a has now measured
   it. Artifacts do not cross a TileLang version boundary in either direction, and carry no
   usable version metadata of their own. But the compatibility surface is four statically
   extractable op names, so the gate is cheap and precise. See
   [`../research/e4a-artifact-versioning.md`](../research/e4a-artifact-versioning.md).

Static shapes are baked at the frontend. E4b observes a runtime dimension
parameter for a symbolic extent; runtime shape reuse has not been executed.
Cache keys must distinguish static specialization from symbolic dimensions.

**Exit criteria:** E4 answered, E6 passed with a non-CUDA backend, E9 measured, and an
architecture decision record written for each of the four load-bearing choices
([`../adr/`](../adr/)).

---

## Phase 1 — Single-device CLI

**Goal:** the §5 executable, on the hardware that exists.

Sequence:

1. `tensor doctor` first. It is the cheapest honest probe of the packaging story, and every
   later command depends on the same environment/registry/target-detection logic it needs.
   It is also the command most likely to be run by a confused user, so it is the right
   thing to get right early.
2. `tensor build` — the artifact layer that does not exist yet in TileLang (§3 of ground
   truth: no `export_library` equivalent). This is the first genuinely Tensor-owned
   component and the first real engineering risk after Phase 0.
3. **The prototyping surface (ADR 0006)** — device buffers, launch, `assert_close`, `bench`,
   NumPy/DLPack interop. Pulled forward out of Phase 6 because it needs only *opaque*
   executables, so it is **not gated on E4**, and because it is the harness E9–E12 run on.
   It also forces the §15 runtime ABI into existence against real use rather than on paper.
4. `tensor inspect` — a thin wrapper over `lower_trace`, which already works.
5. `tensor run`, then `tensor bench`.

**Metrics (§23), all of which need a number, not a vibe:**

- steps from download to first kernel
- manually installed dependencies
- cold compile latency, warm compile latency
- warm-start latency — **note the 4.2 s import floor and the 23 s first-run cost**
- diagnostic quality (scored against a set of deliberately broken kernels)
- artifact portability across hosts
- **launch overhead** — newly load-bearing, because the prototyping surface is the thing
  that exposes it

**Design constraints discovered so far:**

- Do not import the compiler eagerly in the CLI. 4.2 s per invocation is not acceptable for
  a tool whose main verb is `run`.
- Handle "no device present" gracefully. `determine_target()` currently raises.
- Ship the `tl_templates` header tree with any artifact. Generated code is not self-contained.
- Pin TileLang exactly. API drift inside a single 0.1.x release is already observed.

---

## Phase 2 — Stable runtime ABI

**Goal:** separate compiler, runtime, and provider.

Deferred deliberately. §3.4 argues the module contract must be independently versioned so
TIRx can be replaced later, and E4 will tell us how much of the portable representation is
TIRx. Designing the ABI before E4 risks freezing TIRx's shape into Tensor's public contract —
which is precisely the coupling §3.4 and Risk 4 warn against.

Start from `apache-tvm-ffi` (3.4 MB) rather than inventing an FFI. Spike the C++/Rust claim
before designing around it.

---

## Phase 3 — Module system

`tensor.json`, module resolver, artifact cache, exports, versioned ABI — the §11 design.
**Now a packaging-and-versioning problem rather than a representation problem**, because E4
established that the portable tier is a serialized TIRx module.

The work that actually matters here is done in design and waiting on E4b. E4a produced a
concrete manifest spec: `format_version`, exact `tilelang_version`, exact `tvm_ffi_version`,
the extracted `op_set`, target specialization, and a content hash. The gate is three cheap
checks before the load — TileLang version, FFI version, op set.

Also still open from E4: a source-based artifact needs the 896.7 KB `tl_templates` header
tree to compile, so "portable across hosts" is a layered answer — either ship headers, ship
pre-compiled code, or accept that codegen happens where the compiler lives.

---

## Phase 4 — PyTorch backend

`torch.compile(..., backend="tensor")`, inference-oriented FX graphs first. E8 informs the
representation choice. Keep graph breaks working — PyTorch owns the gap.

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

- **No product code before E4.** Creating `src/tensor` now would freeze structural decisions
  the experiments are meant to inform. The repo has no `src/` yet, on purpose.
- **Every claim carries a number or a source.** Anything else is a guess, and §25's open
  questions were guesses for long enough.
- **Re-verify ground truth at each phase boundary.** The proposal already had one stale
  claim. Knowledge of this stack has a shelf life measured in weeks.
