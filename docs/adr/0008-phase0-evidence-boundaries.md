# ADR 0008 — Bound the product contract by measured Phase 0 behavior

**Status:** Accepted, amended by [ADR 0009](0009-complete-phase0-with-scoped-provider-and-composition.md) · 2026-09-29
**Evidence:** [E15](../research/e15-phase0-validation.md), exact experiment
commit `37fd520ab7d03cc2476aec73a658783d18ee0335`.

## Context

Execution and the remaining local architecture probes now produce real data.
They expose both useful upstream machinery and restrictions hidden by source
emission. Existing ADRs remain design choices rather than claims that every
product promise has been implemented.

## Decisions for the load-bearing boundaries

1. **Artifact representation:** keep a versioned envelope around exact-version
   frontend TIRx as a source tier; use opaque executable bundles for the MVP.
   Post-`LowerTileOp` IR failed standard re-lowering. Fusion/composition is
   deferred and remains unverified. No new portable IR is justified by this
   failure alone. Exact SM matching remains the experimental runtime policy.
2. **Compiler and cache identity:** wrap the existing TileLang caches. Add
   explicit source, lock/toolchain, FFI, ABI, target-feature and specialization
   identity; do not infer a cache hit from elapsed time. Compilation may
   mutate frontend IR, and repeated calls must be measured. Generic `sm_90`
   and `sm_100` are not interchangeable with their architecture-specific `a`
   targets. A source-emission success is insufficient capability evidence.
3. **Runtime and provider boundary:** build on borrowed DLPack tensor views,
   native TVM FFI calls and provider-owned device execution. CPU Cython with
   scalar lowering is the measured second execution model. CPU TVM FFI,
   built-in target takeover, foreign streams, neutral events and signals
   remain restrictions or open work. The provider ABI must not be frozen yet.
   C++ execution and IR loading are demonstrated; Rust and a fully native
   compiler pipeline are not.
4. **Framework boundary:** start with supported inference FX patterns in a
   separate adapter. Preserve fallback and graph breaks. AOTAutograd supplies
   an observed normalized forward/backward boundary, but the spike executes
   those graphs through boxed reference code; it is not a training backend.

## Consequences

Two-host compiler-free execution is now a measured milestone. Phase 1 can be
planned around that evidence, but the repository continues to label Phase 0
as incomplete against its full original scope. The literal new-provider
registration test is not a passing gate. No product packages are added as
part of this validation work.

**Scope amendment — 2026-09-29:** the user explicitly deferred cross-GPU
benchmarking. It is no longer a Phase 0 completion requirement. Performance
evidence remains limited to A10G; static-versus-symbolic benchmarking on that
device remains planned. Cross-target source emission/compilation does not
establish runtime performance elsewhere. Cross-GPU performance stays
unverified until future hardware validation, rather than being marked passed.
The provider, artifact and native/ABI gaps remain unchanged.
