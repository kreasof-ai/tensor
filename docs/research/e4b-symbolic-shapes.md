# E4b — Symbolic Shapes

**Experiment:** `python -m experiments.p0.harness --only symbolic_shapes` (4/4)
**Run:** 2026-09-29 · no GPU · artifacts in `experiments/p0/out/_e4b/`

**Validation update:** the current harness uses a 128-column tile instead of
512 columns, reducing the three float16 shared buffers from 384 KiB to 96 KiB.
It checks the runtime parameter and byte-identical symbolic serialization
round-trip. The numbers below describe the original source-emission probe;
neither version has been launched on a GPU.

Follows [`e4-artifact-shape.md`](e4-artifact-shape.md), which concluded that shapes are
baked at the frontend. **That conclusion was right about the default kernel and wrong as a
general statement.** TileLang has a second, better mode.

---

## Verdict

> **A `T.dynamic` extent becomes a runtime kernel parameter. One artifact serves every
> shape of that dimension. E4's "not re-shapeable" qualifier holds only for static shapes.**

---

## The measurement

The decisive test is not a diff between two shapes. It is **where the extent appears in the
generated kernel's signature.**

Authoring `T.Tensor((M, 512), ...)` with `M = T.dynamic("M")` produces:

```c
extern "C" __global__ void __launch_bounds__(128, 1) dyn_add_kernel(
    const half_t* __restrict__ A,
    const half_t* __restrict__ B,
    half_t* __restrict__ C,
    int M)                                            // ← runtime parameter
```

and every access is predicated against it:

```c
if (((((int)blockIdx.x) * 128) + (i * 2) + (((int)threadIdx.x) >> 6)) < M) {
    condval = *(uint4*)(A + (...));
```

Three predication checks in 75 generated lines. The grid becomes `ceildiv(M, 128)` at
launch; offsets use `blockIdx.x * 128` with no baked extent.

**`M_is_kernel_parameter = True`.** The artifact is shape-agnostic by construction.

## Original substitution probe — not evidence of runtime shape reuse

Substituting a concrete value into the serialized symbolic artifact and re-lowering:

| Substituted M | digest |
|---|---|
| 512 | `9dd1353e64ac172c` |
| 256 | `9dd1353e64ac172c` |
| 1024 | `9dd1353e64ac172c` |

Identical output does not establish that a substitution took effect or that a
kernel executes correctly at these shapes. The harness now replaces this
probe with `roundtrip_runtime_extent`, comparing original and reloaded source
and checking that `int M` survives. Runtime shape reuse remains unverified.

## The contrast that makes it meaningful

| | static extent (Python `int`) | `T.dynamic` extent |
|---|---|---|
| Appears in kernel signature | ❌ no | ✅ yes, `int M` |
| Baked into offsets / grid | ✅ yes | ❌ no |
| Access guarded at runtime | ❌ no | ✅ predication, 3 checks |
| One artifact serves many shapes | ❌ no | ✅ yes |
| Generated lines (same op) | 85 | 75 |

Static extents are folded into address arithmetic and grid constants at codegen. Symbolic
extents are not.

The symbolic artifact also serializes normally: **36.7 KB**, with the same
`{"tvm_version": "0.25.dev0"}` metadata and the same four-op compatibility surface as any
other artifact (E4a). So it participates in the same version gate.

## What this changes

**E4's write-up was too strong and is corrected here.** The accurate statement is a
*choice the author makes per dimension*:

| | default (static) | `T.dynamic` (symbolic) |
|---|---|---|
| Cache key must carry | every concrete shape | only the symbolic/static choice |
| Artifact reuse | one shape | every shape of that dim |
| Generated code | fully folded, faster | one register + predication, slightly slower |

**Design consequences:**

1. **The module manifest must record which dimensions are symbolic, not the concrete shapes.**
   A cache key of `(source, target, shape)` is right for static kernels and badly wrong for
   symbolic ones — it would miss every reuse.
2. **Tensor should make the symbolic form reachable and attractive.** It is a strictly better
   default for dimensions that vary at runtime, and a worse one when full unrolling matters.
   That is a real tradeoff the user should make knowingly, not one Tensor should hide.
3. **This is a §25 question now answered.** "Can a user-defined op be shape-polymorphic?"
   Yes — natively, at the TileLang frontend, with no new IR.

## What it does not settle

- **No numerics.** Nothing was executed. A predicated dynamic-shape kernel is exactly the
  kind of thing that is subtly wrong at a boundary (e.g. `M = 127`, `M = 129`) and E10 must
  cover non-multiple-of-tile sizes.
- **No performance data.** Predication and an extra register are not free. E11 must compare
  static vs symbolic on the same kernel. Until then, do not claim symbolic is free.
- **Grid computation is asserted, not verified.** The generated code uses `blockIdx.x * 128`
  and `< M`; that the *launch* passes `ceildiv(M, 128)` is inferred from the source, not
  observed. Verify on the NVIDIA box.
- **Interaction with tiling is untested.** `T.gemm` requires static tile dimensions — the
  compiler explicitly errors on symbolic ones. So a symbolic extent is fine for elementwise
  and reductions, and probably not for tiled GEMM. **Where the boundary lies is the natural
  next question**, and it directly affects what a user can write symbolically.
