# ADR 0006 — The tensor surface is a kernel-author prototyping layer, not a tensor library

**Status:** Accepted · 2026-09-29
**Supersedes the framing in:** the "basic tensor/nn interface" discussion; narrows it.

## Context

The question was whether Tensor should ship a basic tensor/nn interface, by analogy to Bun
providing a built-in HTTP server. Ground truth established why the question arises at all:
torch is 71% of the install (495 MB of 695 MB), `import tilelang` costs 4.2 s, and there is
**no torch-free path to using TileLang today**.

Two possible audiences:

- **(a) non-PyTorch users doing inference** — a small tensor library, NumPy-interop-first,
  competing with PyTorch on ergonomics.
- **(b) kernel authors prototyping** — buffers, launches, reference checks, timing.

**Decision: (b).** The audience is Tensor's own community. A kernel author is the person who
benefits most from the project's real strength — the compiler — and is least served by a
second, worse PyTorch.

This choice has a consequence that is easy to miss, and it is the main reason to prefer it.

## The consequence: this is not gated on E4

A tensor algebra or nn layer needs **fusible** modules to be competitive. If serialized TIRx
turns out not to be re-specializable, every op becomes an opaque binary and a Python-level
tensor interface would be slower than PyTorch eager. That argument made the nn layer wait on
E4.

A prototyping surface needs only **opaque executables** — which exist either way. So:

| Surface | Gated on E4 (fusibility)? | Can start |
|---|---|---|
| Prototyping primitives: buffer, launch, timing, numerics | **No** | Phase 1 |
| Tensor algebra, `nn` layers | Yes | after E4 |

**This surface is robust to a negative E4.** If fusibility fails, everything above still
stands.

## The second consequence: it forces the runtime ABI into existence

The prototyping surface's primitives *are* the §15/§16 runtime ABI — device, buffer, stream,
event, launch, copy — with a real user instead of an abstract design. Building it early means
the Phase 2 ABI is validated by actual use before it is frozen, rather than designed in a
vacuum and discovered not to fit.

It is also the harness E9–E12 need. Numerical validation (§E10) and performance (§E11) are
exactly "allocate, fill, launch, compare, time." Making that a product surface and an
experiment harness is the same work done once.

## Scope

**In** — the minimum to prototype a kernel without PyTorch:

- device buffers with shape, dtype, strides, and a byte-level view
- construction from NumPy (zero-copy where possible) and to NumPy
- construction from DLPack, so `torch` / `jax` / `np` interop is free
- allocation: `zeros`, `ones`, `randn`, `arange`, `full`
- a compiled kernel, launched with bound arguments
- `assert_close` against a NumPy reference
- `bench` with warmup and iteration count
- shape specialization and cache introspection

**Out** — deliberately, and this is the part that keeps it honest:

- broadcasting, autograd, optimizers, layers, `nn.Module`, composition
- a general tensor algebra; if an op is not needed to prototype a kernel, it does not exist
- replacing PyTorch. The goal is that a kernel author *never needs to install it*, not that
  an application author prefers Tensor over it.

**No autograd.** It is the scope trap: it is where "basic tensor/nn" silently becomes
"a framework", and it would consume the project. Inference-only, consistent with §22
Phase 4's inference-first stance.

## Extension

The basic ops are **examples, not a closed set**. A user must be able to add an op without
forking, monkeypatching, or asking for a release.

**Two tiers, with different current status:**

**Tier 1 — a new standalone kernel. Open today.** `tx.build("my_op.py")` accepts any
TileLang-compatible source. There is no op registry to extend, no plugin to register, and no
API surface to design — the extension mechanism *is* the compile path. Verified escape
hatches reachable from user code: TileLang source, `Tx.cuda.*` / `Tx.ptx.*` backend
intrinsics, and raw device-source injection (`_load_cuda_source`). This covers §3.6's
"escape hatches are intentional" and lets frontier work happen before any abstraction exists.

**Tier 2 — a fused composition of several ops into one kernel. Not available to users
today.** Verified against TileLang 0.1.14: there is no user-facing fusion API — no `fuse`,
`fusion`, or megakernel entry point in `tilelang`, `tilelang.transform`, or `tilelang.contrib`.
`par_compile` compiles *independent* kernels concurrently; it is not fusion. TIRx describes
megakernel stitching internally ("re-offsetting shared memory, renaming barriers, reassigning
warp roles, interleaving pipelines across tasks") but exposes no Python surface for it.

**Tier 2 is exactly what E4 decides.** A positive E4 gives Tensor a serialized, re-specializable
IR on which a composition layer can be built. A negative E4 means Tier 2 does not exist, and
the surface stays Tier 1 — still open, still sufficient for most kernel-author work, just not
fused. **This is the main reason Tier 1 is worth building first.**

**The design principle that makes both tiers durable:** the basic ops must be built through
the same user-reachable path as everything else. If `add` is an ordinary module artifact
rather than a special-cased builtin, extension is automatic — there is no extension API to
design, version, or get wrong. The Bun analogue is exact: `Bun.serve` is not special-cased,
it is built-in code written in the same language users write in.

Corollary, and a rule to hold ourselves to: **do not hardcode the basic ops into the runtime.**
A builtin is a closed library, and closed libraries are exactly what people fork.

**Discovery** is module resolution (§11): a user's ops are a package with a `tensor.json` and
named exports, installed with `tensor add`. Not a plugin scan, not import-path convention —
the ecosystem survey's top finding was that entry-point groups are the only mechanism that
lets a third party plug in without patching.

**What stays deliberately closed:** a *new backend or a new memory space* is a provider-level
concern (§3.6, TileLang's `register_backend`), not an op-level one. A user op should not be
able to invent an address space.

## Sketch

```python
import numpy as np
import tensor as tx

a = tx.from_numpy(np.random.randn(M, K).astype(np.float16))
b = tx.randn((K, N), "float16")
c = tx.zeros((M, N), "float32")

kernel = tx.build("attention.py", target="sm_90")   # TileLang-compatible source
out = kernel(q, k, v, block=(128, 64))               # returns buffers, not an object graph

tx.assert_close(out.to_numpy(), reference, rtol=1e-2)
tx.bench(kernel, (q, k, v), warmup=10, iters=100)
```

Note what is absent: no `Tensor` class with operator overloads, no graph, no layers. What is
present is everything needed to answer "is my kernel correct, and is it fast?"

## Consequences

- **Narrower product, better fit.** This does not create the "install Tensor, get a tensor
  stack" story — that was audience (a). It creates "work on kernels without installing
  PyTorch," which is a smaller claim and an achievable one.
- **It does not fix the torch install problem.** ADR 0005 is orthogonal and still needs the
  upstream TileLang change. A torch-free *install* remains blocked on that.
- **The audience is small** — kernel authors, i.e. roughly TileLang's existing user base.
  That is a deliberate bet: serve the people the compiler is for, rather than a wider
  audience the compiler cannot yet serve well.
- **It is still the thing that would make Tensor's own development faster**, because the
  experiments need it. Treat it as infrastructure with a public face, not as a feature.
- **Durable naming:** `tensor.prototype` or a top-level `tx` namespace. It should read as a
  workbench, not as a framework, so that nobody assumes a layer is missing by accident.
