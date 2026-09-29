# E4 — Artifact Shape and Re-targetability

**Experiment:** `python -m experiments.p0.harness --only artifact_shape`
**Run:** 2026-09-29 · 9/9 measurements ok · no GPU, no CUDA toolkit
**Artifact produced:** `experiments/p0/out/artifact_frontend_mod.json` (53.4 KB on disk)

This was the experiment the whole roadmap was waiting on. It asks whether Tensor can ship
a **portable, re-specializable compiled module** — or needs its own portable IR.

---

## Verdict

> **The portable tier exists. A serialized TIRx module survives serialization, re-targets,
> and reloads in a fresh interpreter. Tensor does not need to own a portable IR for MVP.**

With one important qualifier, stated plainly below: the tier is **re-targetable, not
re-shapeable**.

---

## The structural finding that made the question tractable

TileLang's CUDA pipeline runs **`BindTarget(target)` as its very first pass**
(`tilelang/cuda/pipeline.py`). The target is baked in immediately, so there is **no
target-agnostic capture point inside lowering**. Any portable artifact must be captured
*before* `lower()` is called.

The only such thing is the frontend TIRx `PrimFunc` produced by `@T.prim_func` — which is
already a `tvm.tirx.function.PrimFunc`, i.e. plain TIRx, not a TileLang-private type. That
is §3.4's "TIRx as an implementation substrate," confirmed empirically.

## Measurements

### Artifact size — what a `.tbin` must carry

| Form | Size | Note |
|---|---|---|
| FFI JSON (`tvm.ir.save_json`) | **50.3 KB** | the portable artifact candidate |
| TVMScript text (`mod.script()`) | 2.3 KB / 32 lines | human-readable, diffable, **not** lossless-looking |
| Generated CUDA source | 17.0 KB (sm_80) / 7.5 KB (sm_90) | **not portable** — target-specific, and `#include`s headers |
| `tl_templates` header tree | **896.7 KB / 46 files** | required to compile the source at all |

**The headers are 18× the size of the IR.** Generated code `#include`s `tl_templates/...`,
so a source-only artifact does not compile without them.

### Re-targetability and round-trip fidelity

| Measurement | Result |
|---|---|
| `original` → sm_80 | 207 lines, digest `08912d4fef6d1aa0` |
| `original` → sm_90 | 133 lines, digest `5358ab907bc2133f` |
| `reloaded` → sm_80 | 207 lines, digest `08912d4fef6d1aa0` — **identical** |
| `reloaded` → sm_90 | 133 lines, digest `5358ab907bc2133f` — **identical** |
| **retargetable** | **YES** — re-targeting changes output (207 vs 133 lines) |

The serialized artifact round-trips **byte-identically** and still produces genuinely
different, target-appropriate code per arch. That is the definition of a portable tier.

### Cross-process reload — the test that actually matters

```
cross_process_reload → RELOAD_OK sm_80=08912d4fef6d1aa0/207L sm_90=5358ab907bc2133f/133L
```

A **fresh interpreter** loaded the 53 KB JSON and reproduced both digests exactly. An
artifact that only loads in the process that created it would be worthless; this one does
not have that problem.

### Shape specialization — the qualifier

| Measurement | Result |
|---|---|
| gemm 512³ @ sm_80 | digest `08912d4fef6d1aa0` |
| gemm 256³ @ sm_80 | digest `605d521b12be9d6b` — **differs** |

**Shapes are baked in at the frontend.** A captured artifact is re-targetable but **not
re-shapeable**: producing the same kernel at a new shape requires re-running the TileLang
frontend, not re-lowering stored IR.

> **Corrected by [E4b](e4b-symbolic-shapes.md).** This holds for the *default*,
> statically-shaped kernel. TileLang's `T.dynamic` turns an extent into a runtime kernel
> parameter instead, and one artifact then serves every shape of that dimension
> (`dyn_add_kernel(..., int M)`, with predication). The statement above is accurate about
> static shapes and is not accurate in general.

Two consequences that must land in the design:

1. **The cache key must record the symbolic/static choice per dimension**, not just the
   concrete shapes. A key of `(source, target, shape)` is right for static kernels and badly
   wrong for symbolic ones — it would miss every reuse.
2. **Shape polymorphism is a frontend concern, and it already works.** TileLang exposes
   `T.symbolic` (a deprecated alias) and `T.dynamic`. E4b confirms a symbolic artifact
   re-lowers to any shape of that dimension without recompiling.

---

## What this settles, and what it does not

**Settles:**

- A versioned envelope can carry **re-lowerable frontend TIRx JSON**. A new
  portable IR is not required for this serialization experiment. Module
  composition, fusion and re-scheduling were not tested, so their cost is unknown.
- The artifact is small — 50 KB for a full GEMM — which makes §17's cache and §11's module
  resolver tractable.
- Compiled artifacts are portable across processes today, before Tensor writes a line of code.

**Does not settle:**

- **Numerics.** Nothing was executed. Correctness is still E10.
- **Version compatibility across TileLang/TIRx versions.** The JSON is a snapshot of an IR
  that is ~3 months old with breaking changes landing weekly. A `.tbin` needs a format
  version and probably a substrate-version floor. This is now the sharpest risk in §11/§12.
- **Symbolic/dynamic shape re-lowering** (above).
- **Fusion.** Re-lowering one function is not evidence that separately authored
  modules can be composed or fused. This still needs its own experiment.
- **Opaque executables.** E4 covered frontend IR serialization. Whether a compiled binary
  (`.so`/`.cubin`) can be shipped and loaded without a compiler is still open — and the
  896.7 KB header tree says a source-based artifact cannot be, without also shipping headers.
- **The `tl_templates` dependency.** A design that wants artifacts to build on a machine
  without TileLang must either ship the header tree, ship pre-compiled code, or accept that
  codegen happens where the compiler lives. §25's "artifact portability across hosts" is a
  *layered* answer, not a single one.

## Consequences for the roadmap

- **IR storage can reuse an existing representation.** This reduces one part of
  Phase 3; composition and fusion remain unmeasured engineering work.
- **Phase 1's `tensor build` is now implementable** rather than speculative.
- **ADR 0006's Tier 2 (fused user ops) remains gated on E4** — and E4 answered the artifact
  half of that question *positively*. The remaining half is whether a composition layer can
  be built on the stored IR, which E4 did not test.
- **New highest-priority experiment:** symbol version + symbolic-shape re-lowering, because
  the cache key and the module format both depend on the answer.
