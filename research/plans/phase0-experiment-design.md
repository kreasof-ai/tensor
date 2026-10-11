# Phase 0 — Experiment Design

How the open questions in proposal §25 get answered, on two machines, producing data that
can be compared rather than anecdotes that cannot.

Current state: the original measurements are in
[`../research/phase0-ground-truth.md`](../research/phase0-ground-truth.md).
The workload corrections and active compiler-free execution milestone are
tracked in [opaque artifact validation](opaque-artifact-validation.md).
GPU numerics and startup latency pass on an A10G. The broad validation and
Actions-to-A10G executable transfer are reported in
[E15](../research/e15-phase0-validation.md); the final exit gates and scoped
completion decision are reported in [E16](../research/e16-phase0-exit.md).

---

## 1. The two-machine protocol

The same repository, the same kernels, the same harness, two hosts. Results are only
comparable if the *program* is identical, so kernels live in `experiments/p0/kernels.py`
and are never duplicated per machine.

```text
   no-GPU host (Windows, RX 6700 XT)          NVIDIA host (Linux + CUDA)
   ┌──────────────────────────────────┐       ┌──────────────────────────────┐
   │ environment   import, footprint │       │ numerics    correctness      │
   │ codegen       source emission    │       │ execute     run + benchmark │
   │ pass_trace    IR at every stage  │       │ nvcc_time   full compile     │
   │ artifact      shape of outputs   │       │ warm_start  load latency     │
   │ diagnostics   error quality      │       │ launch      per-launch cost │
   └──────────────────────────────────┘       └──────────────────────────────┘
                        └────── same git revision ──────┘
```

**Rule: never compare a number from one machine against a number from the other unless the
git revision matches.** Record the revision in the results JSON.

The remote machine should run the same command:

```bash
git clone <repo> && cd tensor
uv sync
uv run python -m experiments.p0.harness
```

The new experiments below must be written to degrade honestly on a host with no device —
report `skipped: no device`, never a fabricated zero.

---

## 2. Experiments, in dependency order

### Done and passing (local)

| Experiment | Answers | Status |
|---|---|---|
| `environment` | §25 packaging; §23 first-kernel steps, warm start | ✅ passing |
| `codegen` | §25 IR complexity, lowering stages; §26 source LOC | ✅ 15/15 |
| `pass_trace` | §19 inspection surface; §26 stage count | ✅ 5/5 |
| `torch_dependency` | ADR 0005 — is torch structurally required? | ✅ answered (ADR 0005) |
| `artifact_shape` | **E4 — is the portable tier real?** | ✅ **9/9, answered positive** |
| `artifact_versioning` | **E4a — is a `.tbin` self-identifying?** | ✅ **5/5, answered** |
| `symbolic_shapes` | **E4b — one artifact, many shapes?** | ✅ **4/4, answered positive** |

E4's result is written up in [`../research/e4-artifact-shape.md`](../research/e4-artifact-shape.md).
A serialized TIRx module round-trips byte-identically, **re-targets**, and **reloads in a fresh
interpreter**. Tensor does not need to own a portable IR.

E4a's result is in [`../research/e4a-artifact-versioning.md`](../research/e4a-artifact-versioning.md).
Artifacts do **not** cross a TileLang version boundary in either direction, and carry no usable
version metadata of their own — so the version gate is Tensor's job. The gate is cheap: the
compatibility surface is four statically-extractable op names.

E4b's result is in [`../research/e4b-symbolic-shapes.md`](../research/e4b-symbolic-shapes.md).
A `T.dynamic` extent becomes a **runtime kernel parameter** (`dyn_add_kernel(..., int M)` with
predication), so one artifact serves every shape of that dimension. This *corrects* E4's
"shapes are baked" — that is true only of the default static kernel. The cache key must record
the symbolic/static choice per dimension, not just concrete shapes.

One experiment remains unrun locally:

- **E5 — `cache_behaviour`.** What TileLang already caches, whether a hit skips lowering,
  and what its key includes. Tells us whether Tensor writes a cache or just keys one.
- **E5b — `symbolic_tiling_boundary`** (new, from E4b). `T.gemm` requires static tile
  dimensions, so symbolic extents work for elementwise/reduction but probably not for tiled
  GEMM. Where exactly that boundary lies decides what a user can write symbolically.

Current harness total: **45 ok, 2 expected-failure, 0 unexpected** (47 measurements). The two
expected failures are the `torch_blocked` and `torch_stubbed` probes — their failing *is* the
result, and the harness distinguishes that from a broken experiment.

Also on hand, run directly rather than through the harness:
`experiments/p0/link_check.py` (PE import tables), `experiments/p0/cross_version_probe.py` and
`make_artifact.py` (the E4a cross-version matrix, which needs a second interpreter).

Also on hand: `experiments/p0/link_check.py`, a PE import-table parser showing the shipped
native libraries contain no torch symbol. Run it directly; it is not part of the harness.

### Next: the two experiments that decide the module system

These are the highest-value unrun work, because §11/§12 (the module format) is the one place
Tensor would be building something TileLang does not already have.

**E4 — `artifact_shape`.** What does a compiled TileLang program actually consist of, and
what must a `.tbin` carry?

Emit, for one kernel: the generated source, the TIRx module, the pass list, the header
directory, and the ABI/target metadata. Measure the total size of each. Answer: what is the
*minimum* thing to persist that still permits fusible re-specialization (§12), versus what
is only needed to load and execute.

Explicitly test the fusible/opaque split: serialize the post-`LowerTileOp` TIRx and confirm
it can be re-scheduled and re-lowered. If yes, Tensor's portable artifact can be a thin,
versioned envelope around TIRx serialization. If it fails, determine whether
pinned frontend IR plus opaque execution is sufficient before considering a
new IR. E15 measured a post-`LowerTileOp` re-lowering rejection; this does not
establish that a new IR is necessary.

**E5 — `cache_behaviour`.** TileLang has `enable_cache`/`disable_cache` and
`is_cache_enabled`, and its JIT caches. Measure: cold compile, warm compile, cache hit
latency, what the key includes, and whether a hit actually skips lowering. §17 needs this
before any cache design, because the likely honest answer is *"TileLang already caches;
Tensor's job is to key, version, and invalidate it"* rather than *"write a cache"*.

### Then: the provider and runtime boundary

**E6 — `provider_contract`.** Register a minimal second backend against
`tilelang.backend.register_backend(BackendModule(...))` and drive it end to end. The
`cpu` backend is the cheapest real choice. Success = a backend that is not CUDA, registered
through the documented manifest, compiling a real kernel. This is the Risk 3 test: if the
provider ABI cannot survive a second, genuinely different implementation, it is CUDA-shaped
regardless of how well it is named.

**E7 — `abi_surface`.** Inventory what a non-Python host actually needs, and check it
against §15 (streams, events) and §16 (async, signals, memory ordering). Start from
`apache-tvm-ffi` (3.4 MB) rather than designing from scratch. Spike the claim that TIRx is
usable from C++/Rust before relying on it — `import tvm` failing outside `import tilelang`
is a warning sign, not a detail.

**E8 — `framework_contract`.** `torch.compile(backend=...)` and AOTAutograd as the §13
integration point. The real question from §25: FX directly, AOTAutograd, or something
normalized? Cheap to probe, and it constrains the compiler interface early.

### Full CUDA toolkit and NVIDIA execution

E9 needs a complete CUDA build toolkit, not a GPU. The Actions producer
demonstrates that separation. E10/E11 and the E12 consumer need NVIDIA hardware.

**E9 — `full_compile`.** Compile latency with `nvcc` included, cold and warm, per kernel and
target. §25's compile-latency question, properly answered. Local numbers are a floor only.

**E10 — `numerics`.** Run every kernel, compare against a reference implementation. This is
what makes the kernels real rather than plausible-looking. Until this runs, the workload set
is structurally correct and semantically unproven.

**E11 — `perf`.** Kernel throughput vs a tuned baseline on the available A10G,
including the completed static-versus-symbolic comparison. §23's runtime metrics.
At the user's request on 2026-09-29, cross-GPU `sm_80`/`sm_90`/`sm_100`
benchmarking is deferred and is not a Phase 0 exit requirement. That future
comparison can test whether TMA-vs-cp.async lowering differences pay off on
the corresponding hardware. Cross-target compilation remains a separate probe.

**E12 — `warm_start`.** Load a prebuilt artifact and measure time-to-first-kernel, with
TileLang's cache warm and cold. The §23 number that constrains the whole `tensor` UX.

> **E9–E12 are what the ADR 0006 prototyping surface exists to serve.** All four are
> "allocate, fill, launch, compare, time" — exactly the primitive set defined there. The
> first useful version of that surface should be built *to run these experiments*, not
> after them. Doing it in that order means the runtime ABI gets validated against real use
> before Phase 2 tries to freeze it, and E10/E11 are never blocked on building a harness.
>
> Numerics, throughput and loading need a real NVIDIA device; compilation alone
> does not. The A10G now provides that execution evidence in E14/E15.

---

## 3. What counts as an answer

Guard against the failure mode where a month of experiments produces nothing falsifiable.

An experiment is complete when it produces a number in `results.json` and a line in
`docs/research/` — **including when the answer is unfavourable.** Specifically:

- If TIRx serialization is not fusible, that is a *result*. It redirects Phase 2, it does
  not fail the project.
- If the second provider cannot be registered cleanly, that is a result about §9/§10.
- If compile latency with `nvcc` is 10 s, the product needs a daemon, and that is a result
  about the §5 UX goal.

**Unverified is a valid state. Vague is not.**

---

## 4. Explicit non-experiments

Named so they do not get smuggled in:

- **A new IR.** §20 lists "a new universal compiler IR" as a non-goal. E4 is designed to
  find out whether one is *forced* on us; it is not a licence to start writing one.
- **A second hardware family at MVP.** Phase 5 in §22. Not a Phase 0 concern.
- **RDNA2.** There is no supported path (see ground truth §7). Not on the roadmap.
- **Distributed anything.** §16 and §3.1 are explicit.
- **Autotuning infrastructure.** §18 says bounded and optional; E5 tells us what TileLang's
  existing `autotune` already does before we add anything.

---

## 5. Risks this design carries

1. **TIRx churn.** TIRx is ~3 months old, `Development Status :: 4 - Beta`, and
   `src/tirx` took commits on 2026-09-26, 27 and 28 including an Op registry introduction
   and typed OpDef registration. Pin exact versions; expect IR-node-level breakage, not just
   API changes. **Design so the compiler is replaceable** — which is §3.4's own argument,
   now with evidence behind it.
2. **TileLang has no published stability policy** (UNVERIFIED that one is absent). Its
   v0.1.13 notes "removes several legacy APIs." Treat every internal import as private.
3. **Source emission is weaker than full compilation and numerics.** E10 now
   validates twelve cases on A10G; E15 finds actual nvcc failures on several
   other targets despite successful source emission.
4. **Version drift between proposal and reality.** The proposal already contains one stale
   claim (TileLang/TIRx relationship timing). Re-verify against installed packages at the
   start of each phase, not from memory or blogs.
