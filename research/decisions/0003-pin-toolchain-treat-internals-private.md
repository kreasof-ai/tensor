# ADR 0003 — Pin the toolchain exactly; treat TileLang internals as private

**Status:** Accepted · 2026-09-29

## Context

Ground truth §5 documented four API errors encountered during routine use of a *single*
release, TileLang 0.1.14:

- `T.prim_func` / `T.Tensor` live on `tilelang.language`, not top-level `tilelang`
- `T.serial(n)` requires `(start, n)`
- `T.Pipelined()` requires `start`
- there is no `T.__syncthreads`; it is `T.sync_threads`

None were announced as breaking changes. TileLang has no published API-stability policy
(UNVERIFIED that one is absent rather than merely unfound), its classifier is
`Development Status :: 4 - Beta`, and its v0.1.13 notes say it "removes several legacy
APIs."

**E4a then found a second, worse class of problem: a loose *transitive* pin that breaks the
native load.** Building a comparison environment for tilelang 0.1.11 failed with
`OSError: [WinError 127] The specified procedure could not be found` — an unreadable error.
Cause: 0.1.11 declares `apache-tvm-ffi>=0.1.10,~=0.1.0`, so pip resolved **0.1.14.post1**,
three minors newer than the native library was built against. The PE import tables of the
two releases are identical, so the failure is a missing *export*. Pinning
`apache-tvm-ffi==0.1.11` fixed it immediately. tilelang 0.1.14 has since tightened its own
pin to `<0.1.13`.

E4a also measured the artifact-level impact: a serialized TIRx module does **not** cross a
TileLang version boundary in either direction, and the only version metadata it carries is
`{"tvm_version": "0.25.dev0"}` — a dev tag that is *identical* across releases. See
[`../research/e4a-artifact-versioning.md`](../research/e4a-artifact-versioning.md).

Underneath, TIRx is roughly three months old and `src/tirx` took commits on 2026-09-26, 27
and 28 — including introducing an Op registry and typed OpDef signature registration. Those
are breaking-class changes to the IR substrate, not just API surface. Concretely, across
three TileLang minor releases the op `region` moved namespace (`tl.region` →
`tl.tileop.region`), which is enough to make every artifact unreadable.

## Decision

1. Pin `tilelang` to an exact version. No ranges in the lockfile, in CI, or in the MVP
   install path.
2. **Pin `apache-tvm-ffi` to an exact version as well**, and never let a resolver widen it.
   A loose transitive pin produces `WinError 127` at `import tilelang` with no indication of
   cause — a failure mode that costs hours and is invisible in a lockfile review.
3. Import only from surfaces that read as public: `tilelang.jit`, `tilelang.language`,
   `tilelang.transform`, `tilelang.tools.*`. Treat `tilelang/backend/*`, `tilelang/engine/*`,
   and the per-backend packages as private even where they are convenient — with the
   documented exception of the `register_backend(BackendModule(...))` manifest, which *is*
   the supported extension point and which Tensor must depend on.
4. Write the TileLang compatibility profile (§7) against observed behaviour of a pinned
   version, never against documentation or memory.
5. Keep every TileLang touchpoint behind one adapter module, so replacing the substrate is
   a bounded change rather than a rewrite.
6. **Version-gate every artifact, in this order:** TileLang version, then FFI version, then
   the artifact's extracted op set, then attempt the load. All but the last are cheap and
   fail before deserialization. E4a measured that artifacts carry no usable version metadata
   of their own, so this gate is Tensor's responsibility, not TileLang's.

## Consequences

- No automatic upgrades. Deliberate and cheap: the adapter module is where breakage is
  absorbed.
- Some convenient internals are off-limits. `tilelang.lower` is the one most likely to be
  wanted for `tensor inspect`; if it turns out to be private API, `compile_only` plus
  `lower_trace` is the supported alternative and is sufficient (both are in `tools`).
- Every artifact Tensor caches is invalidated by a TileLang or FFI upgrade. Measured and
  accepted: the break is loud and detectable in advance, which is better than a silent one,
  but it means the cache key must carry both versions. Users will notice a cold cache after
  upgrading. That is a real product cost, not an implementation detail.
- This ADR is the concrete form of §3.4's "TIRx is an implementation substrate, not the
  public module ABI". The substrate is young enough that this is not a formality: three
  minor releases moved an op namespace, and one loose transitive pin produced a native
  failure with no readable cause.
