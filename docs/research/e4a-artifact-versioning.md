# E4a — Artifact Versioning and the Compatibility Surface

**Experiment:** `python -m experiments.p0.harness --only artifact_versioning` (5/5)
**Cross-version tests:** `experiments/p0/cross_version_probe.py`, run against
tilelang 0.1.14 and 0.1.11 runtimes
**Run:** 2026-09-29 · no GPU

Follows [`e4-artifact-shape.md`](e4-artifact-shape.md), which established that a serialized
TIRx artifact round-trips and re-targets. This asks the follow-up: **does it survive a
TileLang upgrade?**

---

## Verdict

> **No. Artifacts do not cross a TileLang version boundary in either direction. But the
> break is small, clean, and statically detectable — so the version gate is implementable.**

---

## The cross-version matrix

Artifacts generated in one TileLang release, loaded in another:

| Artifact | Runtime | Result |
|---|---|---|
| 0.1.14 | 0.1.14 | ✅ OK — `sm_80=08912d4fef6d1aa0/207L sm_90=5358ab907bc2133f/133L` |
| 0.1.14 | 0.1.11 | ❌ `Operator tl.region is not registered` |
| 0.1.11 | 0.1.14 | ❌ `Operator tl.tileop.region is not registered` |

Both directions fail. Neither fails silently, and both name the exact missing operator.

## The metadata is useless as a gate

```json
"metadata": {"tvm_version": "0.25.dev0"}
```

That is the **entire** self-description. Measured across two real releases:

| | 0.1.14 artifact | 0.1.11 artifact |
|---|---|---|
| `tvm_version` | `0.25.dev0` | `0.25.dev0` — **identical** |
| byte size | 50.3 KB | 46.8 KB |
| node count | 406 | 379 |
| format version | **absent** | **absent** |
| TileLang version | **absent** | **absent** |

`0.25.dev0` is a development tag that does not move between releases. **A cache keyed on
the artifact's own metadata would happily serve an incompatible artifact.** This is the
single most important finding for §11/§12: the artifact is *not* self-identifying, and
Tensor must supply that identity itself.

## The compatibility surface is small and extractable

The observed cross-version failures involve op names. Those are statically
enumerable from the node graph without loading anything:

| | ops |
|---|---|
| 0.1.14 | `tl.region`, `tl.tileop.copy`, `tl.tileop.fill`, `tl.tileop.gemm` |
| 0.1.11 | `tl.tileop.region`, `tl.tileop.copy`, `tl.tileop.fill`, `tl.tileop.gemm` |

**Four ops. Across three minor releases, the only difference is that `region` moved from
the `tl.` namespace to `tl.tileop.`.** Everything else is byte-for-byte the same op set.

This is good news for the design in two ways:

1. **The gate is precise and cheap.** Extract four op names, check them against the
   runtime's op registry, fail before attempting a load. No deserialization required.
2. **The observed load failure is a namespace mismatch** — which is why both failure
   messages are so legible. It is also *mechanically* recoverable via a remap table, though
   relying on that across future versions would be fragile. Use it as a diagnostic, not a
   strategy.

This matrix does not prove that op names are the only compatibility surface or
that matching names imply identical semantics. IR-node schema, compiler passes,
native ABI and compilation options can also change. Exact toolchain pins and
execution validation remain necessary; the op gate diagnoses the measured
failure rather than proving general compatibility.

## Malformed artifacts fail loudly

| Input | Result |
|---|---|
| truncated JSON | `JSONDecodeError` with position |
| valid JSON, unknown node type | `InternalError: Cannot find type 'NotARealNode'` — **names the cause** |
| not JSON at all | `JSONDecodeError` |

No silent corruption. That is what makes it safe to implement a gate: the failure mode of
getting it wrong is a loud error, not a subtly wrong kernel.

## A packaging hazard found on the way

Setting up the 0.1.11 comparison runtime failed with an **opaque native error**:

```
OSError: [WinError 127] The specified procedure could not be found
```

Root cause, confirmed by fixing it: tilelang 0.1.11 declares
`apache-tvm-ffi>=0.1.10,~=0.1.0`. `~=0.1.0` means `<0.2.0`, so pip resolved **0.1.14.post1** —
three minor versions newer than the 0.1.11 native library was built against. The PE import
tables of 0.1.11 and 0.1.14 are **identical**, so the failure is a missing *export*, not a
missing DLL. Pinning `apache-tvm-ffi==0.1.11` made 0.1.11 load immediately.

**A loose transitive constraint produces an unexplained native load failure.** tilelang
0.1.14 tightened its own pin to `>=0.1.11,<0.1.13` (resolving 0.1.12) — evidence that
upstream is already reacting to this class of problem.

---

## What a `.tbin` manifest must carry

Derived from the measurements, not invented:

1. **`format_version`** — Tensor's own artifact format version. Distinct from the package
   version, per the `apache-tvm-ffi` `{major, minor, patch}` and DLPack major/minor precedent.
2. **`tilelang_version`, exact** — demonstrated necessary; there is no substitutable signal.
3. **`tvm_ffi_version`, exact** — demonstrated necessary; a loose pin breaks the native load.
4. **`op_set`** — the four extracted names. This is the precise, checkable contract.
5. **Target specialization** (arch, shapes) — from E4; shapes are baked at the frontend.
6. **A content hash** — so the cache key can be a function of the artifact's bytes rather
   than of claims inside it.

The gate itself: compare (2) and (3) first, then verify (4) against the runtime registry,
then attempt the load. Three cheap checks before the expensive one.

## What this changes

- **The §11 module format now has a concrete, testable spec** rather than an open question.
- **ADR 0003 gets a much stronger justification.** Pinning exactly is not only about Python
  API drift (four documented errors in one release); it is also that a *transitive* loose pin
  can produce a native failure nobody can read.
- **The risk is bounded and known.** One namespace rename in three releases, cleanly
  reported, statically checkable in advance. That is a far better position than "unknown
  compatibility".
- **Still open:** whether a *newer* runtime will accept *older* artifacts under a remap
  policy. E4a says they do not today, and a remap is a deliberate, version-scoped decision
  rather than a default.
