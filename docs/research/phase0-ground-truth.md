# Phase 0 — Ground Truth

**Measured:** 2026-09-29 · **Host:** Windows 11 (10.0.26200), AMD Ryzen 5 5600 (6C/12T), 32 GB, Radeon RX 6700 XT · **No NVIDIA GPU, no CUDA toolkit, no MSVC, no CMake.**

Everything below was either executed on that host or read from a cited source. Claims that
could not be verified are marked **UNVERIFIED** rather than asserted.

Reproduce with `uv run python -m experiments.p0.harness` (26/26 measurements passing).
Raw data: `experiments/p0/out/results.json`.

---

## 1. The headline: most of Phase 0 does not need the GPU

The proposal's Phase 0 asks how much of `tensorc` already exists inside TileLang and TIRx.
The first thing worth knowing is *where* that question can be answered.

**All five workload kernels compile to real CUDA C++ on a machine with no GPU and no CUDA
toolchain — 15/15, 0.07 s to 1.41 s each.** The generated code is not a stub. `gemm_relu`
on `sm_80` produces 207 lines containing `tl::cp_async_gs<16>`, `tl::mma_sync<...,16,8,16>`,
`__syncthreads`, and hand-computed swizzled shared-memory offsets.

The mechanism is `tilelang.tools.compile_only.compile_kernel_source(func, target)`, which
lowers with `enable_device_compile=False` and returns source. It needs no device and no
`nvcc`. This is the same code path TileLang's own `python -m tilelang.tools.compile_only`
CLI uses, and it is also the path `tilelang.compile_only` defaults to with `--target c`.

So the split of work is:

| Needs a real device | Does not |
|---|---|
| kernel performance, TFLOPS, launch overhead | IR inspection, pass counting, source emission |
| warm-start latency of loaded artifacts | compile latency |
| numerics validation | diagnostics quality, cache behaviour, artifact shape |
| NVIDIA provider | packaging, CLI, module system, everything above |

This is the single most useful thing established so far: **the experiments that decide
Tensor's architecture are not blocked on hardware you do not have.**

---

## 2. What TIRx and TileLang already provide

The proposal's §3.4, §8, and §19 describe capabilities that now exist upstream. Building
them again would be waste.

### TIRx is real, and it is the design the proposal was reaching for

TIRx was announced 2026-06-22 as the next-generation kernel IR in Apache TVM, shipped in
`apache-tvm` 0.26.0, and lives at `tvm.tirx` (`from tvm.script import tirx as Tx`).

- It is a **parallel IR, not a rename of `tir`**. `tvm.compile(mod, target, tir_pipeline="tirx")`
  runs a `tirx_pipeline` of 19 passes; the result is consumed by codegen without conversion
  to the `tvm.tir` object model. Legacy `tir` remains a separate path.
- It exposes the three-layer authoring model the proposal describes: core `Tx.*`, tile
  primitives `Tx.tile.*` (→ `TilePrimitiveCall`), backend ops `Tx.cuda.*` / `Tx.ptx.*`.
- The hardware-feature ladder of §3.6 — *raw intrinsic → target capability → provider
  operation → portable primitive* — is **already implemented and enforced**: backend
  intrinsics are table-driven and emit `tirx.ptx.*` calls; promotion to a portable
  `tirx.tile.<name>` op happens later, and each backend independently registers variants.
  `TilePrimitiveDispatcher` calls a global FFI hook and splices the chosen `PrimFunc` in;
  a surviving `TilePrimitiveCall` is a fatal verifier error.
- Backends own their own complete lowering sequence, and new backends register through a
  manifest — not by branching in the engine.

TileLang **is** on TIRx in 0.1.14 (verified: `import tilelang; import tvm.tirx` succeeds,
and `tilelang/cuda/pipeline.py` imports `tvm.tirx` directly).

> Correction to a claim worth flagging: the TIRx announcement blog (2026-06-22) says the
> TileLang integration is future work — *"we are working with the TileLang community to bring
> TIRx as a new minimal foundation."* TileLang's own changelog dates its TIRx migration to
> 2026-05-20 and the installed 0.1.14 demonstrably ships `tvm.tirx`. **Docs written during
> the transition lag reality. Verify against the installed package, not the blog.**

### Inspection tooling already exists

`tilelang.tools.lower_trace.enable(trace_dir=...)` wraps a real compile and writes
`NN_<PassName>_before.tir` / `_after.tir` for **every pass**. Measured:

| Kernel | Pass steps | Distinct passes | TIRx IR dumped |
|---|---|---|---|
| `fused_elementwise` | 78 | 62 | 558 KB |
| `gemm_relu` | 84 | 63 | 2,018 KB |
| `flash_attention` | 82 | 63 | 1,504 KB |
| `row_softmax_reduce` | 78 | 62 | 337 KB |
| `gather_rows` | 78 | 62 | 366 KB |

**~63 distinct passes for one GEMM**, including `LowerTileOp` (TIRx tile dispatch),
`LayoutInference`, `PipelinePlanning`, `LowerHopperIntrin`, `InjectTcgen05Fence`,
`LowerBlackwell2SM`, `ReducerPlanAndMaterialize`, `LowerThreadAllreduce`, `ThreadSync`.

There is also `tilelang/tools/pass_visualizer/`, `tools/pass_timing.py`, and
`tools/Analyzer.py`.

**Implication:** `tensor inspect --stage X` is a thin wrapper over an existing capability.
The interesting product work is the *presentation and stability* of it, not the mechanism.
Do not build a competing IR dumper.

### Six backends are already registered

`tilelang.backend.list_backends()` returns `rocm, cuda, cutedsl, cpu, metal, webgpu`, each
declaring its target kinds, pass pipeline, device codegens, and execution backends:

| Backend | target kinds | pipelines | execution backends |
|---|---|---|---|
| `rocm` | hip | hip | tvm_ffi, cython |
| `cuda` | cuda | cuda | tvm_ffi, nvrtc, cython |
| `cutedsl` | cuda | cuda | cutedsl |
| `cpu` | c, llvm | c, llvm | cython, tvm_ffi |
| `metal` | metal | metal | torch, tvm_ffi |
| `webgpu` | webgpu | webgpu | tvm_ffi |

Note `cutedsl`: TileLang can already emit NVIDIA CuTe DSL, which is one of the systems
§26 asks you to compare against. And `cpu` claims an `llvm` pipeline, though the release
wheel does not ship an LLVM-enabled TVM (`USE_LLVM=ON` requires a source build — **UNVERIFIED
at runtime, not attempted**).

---

## 3. Is torch required to compile? (ADR 0005)

torch is 71% of the installed footprint, so this was measured directly rather than assumed.
Three probes compile a kernel to CUDA source in an isolated process
(`harness.py --only torch_dependency`):

| Probe | Result |
|---|---|
| `normal` | ✅ compiles, 23 lines of CUDA; torch loaded **eagerly** during `import tilelang` |
| `torch_blocked` | ❌ fails *inside* `import tilelang` — `ModuleNotFoundError` |
| `torch_stubbed` | ❌ fails in `tvm_ffi/_optional_torch_c_dlpack.py:193` at `torch.cuda.is_available()` |

**But the native libraries are torch-free.** `experiments/p0/link_check.py` parses the PE
import directory of the shipped binaries:

| Library | Size | torch/c10 in import table |
|---|---|---|
| `tvm_compiler.dll` | 60.0 MB | **none** |
| `tvm_runtime.dll` | 3.4 MB | **none** |
| `tilelang_cython_wrapper.pyd` | 0.1 MB | **none** |

**So the dependency is entirely at the Python layer.** The trigger is one line —
`tilelang/__init__.py:135`:

```python
import torch  # preload torch to avoid dlopen errors
```

A dlopen-ordering precaution, in a binary whose import table shows no torch symbols. There
are also 17 modules with top-level `import torch`, including `cuda/target.py` and
`language/dtypes.py` — core paths, not only interop.

**Conclusion:** a torch-free *binary* is achievable now; a torch-free *install* needs the
`torch` → extra change upstream (ADR 0005). Until then `pip install tensor` still pulls
torch via TileLang's metadata, and should not be claimed otherwise.

---

## 4. Measured numbers

### Packaging footprint (Windows)

`pip install tilelang numpy` → 20 packages, **218.9 s** wall (3 min 13 s of it uv resolving
and fetching, 24.4 s installing).

| Package | Downloaded wheel | Installed on disk |
|---|---|---|
| `tilelang` 0.1.14 | 27.2 MB (`cp39-abi3-win_amd64`) | 130.0 MB |
| `torch` 2.14.0 | 118.4 MB | 495.4 MB |
| `numpy` 2.5.3 | — | 44.0 MB |
| `z3-solver` 4.15.4.0 | 16.3 MB | 21.2 MB |
| `apache-tvm-ffi` 0.1.12 | 3.4 MB | ~0 MB |
| `torch-c-dlpack-ext` 0.1.5 | — | 4.7 MB |
| **Total** | **~230 MB** | **695.3 MB** |

**This answers §25's packaging question favourably on Windows.** A user downloads ~230 MB
and runs a kernel. The tile compiler itself is the *small* part.

The risk is `torch`: **71% of the installed footprint is torch**, a hard dependency of
tilelang, and its Linux install pulls a further ~3 GB of `nvidia-*` wheels. Any `tensor`
that promises a lean install must confront this directly. The obvious split is a
**codegen-only mode with no torch** — but note that tilelang's DLPack tensor bridge
(`torch-c-dlpack-ext`, 4.7 MB) is the interop path, so dropping torch means owning a tensor
ABI from day one.

### Import cost — a real threat to `tensor run`

| Measurement | Time |
|---|---|
| bare interpreter start | ~0.00 s |
| `import torch` | 1.65 s |
| `import tilelang` (warm `.pyc`) | **4.19 – 4.46 s** |
| `import tilelang` (first ever, cold `.pyc`) | **23.34 s** |

The 23 s figure is real and will be what a first-time user experiences. Even warm, ~4.2 s
is a large fixed cost for a CLI whose headline command is `tensor run kernel.py`.

**This is a concrete Phase 1 design constraint, not a future concern:** a `tensor` CLI that
imports the compiler eagerly pays 4+ seconds on every invocation. The obvious answer is a
long-lived daemon or a pre-forked server, which conflicts with the "users install Tensor,
not a toolchain" simplicity goal and needs measuring. **Add `tensor run` warm-start latency
to the Phase 1 metric set** (§23 lists it; give it a number).

### Compile latency (source emission, no device)

| Kernel | sm_80 | sm_90 | sm_100 |
|---|---|---|---|
| `fused_elementwise` | 0.61 s | 0.43 s | 0.47 s |
| `gemm_relu` | **1.41 s** | 0.29 s | 0.29 s |
| `flash_attention` | 0.86 s | 0.16 s | 0.21 s |
| `gather_rows` | 0.20 s | 0.10 s | 0.10 s |
| `row_softmax_reduce` | 0.07 s | 0.07 s | 0.07 s |

Sub-second for everything except a pipelined GEMM. Source emission is fast; these numbers
are a **floor**, not the full compile, because the vendor toolchain step (`nvcc` → PTX →
cubin) has not run. §25's "can the selected path meet the desired interactive budget"
therefore remains **open** — it must be re-measured on the NVIDIA box with the toolchain
included. The useful takeaway is that a big constant sits *upstream* of `nvcc`, not in it.

### The target genuinely changes the lowering strategy

Not just a different `--arch` flag. For `gemm_relu`:

| Target | Source | MMA | Async copy | TMA | Barriers |
|---|---|---|---|---|---|
| sm_80 | 207 lines / 17.0 KB | 6 × `mma_sync` | 12 × `cp_async` | 0 | 4 |
| sm_90 | 133 lines / 7.5 KB | 0 | 0 | 8 | 1 |
| sm_100 | 131 lines / 8.0 KB | 2 × `mma_sync` | 0 | 8 | 1 |

Pre-Hopper stages through `cp.async` with a 3-stage software pipeline; Hopper+ uses **TMA
tensor maps** (`__grid_constant__ const CUtensorMap`). Generated source *shrinks 36%* on
sm_90 because TMA is more declarative than hand-expanded async copies.

The `mma_instr=0` on sm_90 is a measurement artifact of the marker regex, not a missing
feature — sm_90 routes MMA through a different template path. **Do not trust structural
markers as a performance proxy; they are a smoke test for "is this real code".**

This is good evidence that the "capability-based, not vendor-name-based" provider selection
of §10 is workable in practice: `have_tma`, `have_mbarrier`, `have_pdl` are queried as
capability predicates inside the CUDA pipeline, not hardcoded per arch.

---

## 5. Friction found while building the harness

These are small, but each one cost real time and each would cost a user real time.

1. **`tvm` is not an installed package.** It is vendored at `tilelang/3rdparty/tvm/python/`
   and only lands on `sys.path` as a side effect of `import tilelang`. `import tvm` alone
   fails. Anything that wants TIRx directly — a Rust host, a C++ runtime, a separate
   process — must go through tilelang or replicate the path bootstrap. This materially
   weakens the "TIRx is exposed through TVM FFI across Python, C++, and Rust" claim
   (**UNVERIFIED** in practice; no spike done yet).

2. **`determine_target()` raises with no GPU**: `ValueError: No registered target detector
   found an available target.` Any `tensor` command that sniffs a target must handle the
   no-device case gracefully rather than crashing.

3. **The `@tilelang.jit` decorator auto-detects a target at call time** and therefore cannot
   be used for compile-only work on a machine without a device. Use the raw `PrimFunc` plus
   `compile_kernel_source`.

4. **API drift within 0.1.14 is real.** `T.prim_func` and `T.Tensor` are on
   `tilelang.language`, not top-level `tilelang`. `T.serial(n)` requires `(start, n)`.
   `T.Pipelined()` requires `start`. There is no `T.__syncthreads` (it is `T.sync_threads`).
   All four bit during normal use — and none of it is documented as breaking, because there
   is no published stability policy (**UNVERIFIED that one is absent**, not that it is).
   **Pin exact versions. Write the compatibility profile against observed behaviour, not
   documentation.**

5. **Generated code is not self-contained.** Every emitted kernel starts with
   `#include <tl_templates/cuda/...>` (6–10 includes). A `.cu` file on its own does not
   compile; the artifact must ship the header tree. This is a first-class input to the §11
   module layout decision and is easy to miss.

6. **Generated code leaks absolute build paths.** `#line` directives embed the full local
   path of the source and of TileLang's own `.py` files:
   `#line 122 "D:\...\site-packages\tilelang\cuda\op\gemm..."`. Good for diagnostics
   (this is the "source-aware compiler diagnostics" feature working), but it makes
   artifacts non-reproducible across machines and will perturb §17 cache keys.

7. **Generated CUDA contains `_MSC_VER` guards.** The codegen already accounts for MSVC
   toolchains on Windows, which is a good sign for the Windows-first story.

---

## 6. What remains unverified

- **Everything requiring execution.** No kernel has been run. No numerics validated, no
  performance measured, no artifact loaded. The compile-coverage kernels are structurally
  plausible but untested.
- **The NVIDIA box numbers.** Full compile latency (with `nvcc`), cold vs warm start,
  launch overhead, and kernel performance are all still open. §25's compile-latency
  question is explicitly *not* answered by this document.
- **AOT artifact round-trip.** There is no `export_library` equivalent in TileLang. The
  closest is source emission plus shelling out to `nvcc` yourself. Whether the cython
  execution backend's binary cache survives a process boundary is **UNVERIFIED** — the
  relevant file is `tilelang/jit/adapter/libgen.py`, which was not read.
- **RDNA2 / gfx1031.** No supported path exists (see §7). Not attempted, deliberately.
- **C++/Rust FFI to TIRx.** Asserted by the TIRx blog, unproven. Needs a spike.
- **Ecosystem comparison (§26) and packaging precedents** — research in progress; not
  included here.

---

## 7. RDNA2 is a dead end as a local target — and that is fine

Stated plainly so it does not absorb effort later.

Your GPU is a **Radeon RX 6700 XT = Navi 22 = `gfx1031`** (RDNA2), driven by Adrenalin
32.0.21043.19003 on Windows.

- TileLang's ROCm backend is **Linux-only**; Windows and macOS wheels do not include it.
  There is no ROCm/HIP toolchain for `gfx1031` on Windows.
- TileLang's own issue tracker has **zero** `gfx1030`/RDNA2 references. The RDNA feature
  requests ask for RDNA3 and RDNA4. CI runs on a self-hosted `gfx942` MI300X.
- Upstream ROCm CI on TileLang was **disabled from 2026-05-21 for lack of contributed AMD
  machines** (issue #2844), and the HIP bug list is "awaiting CDNA/RDNA silicon" (#2645).

Options considered and rejected: WSL2 + ROCm (`gfx1031` dropped from supported lists),
the HIP SDK on Windows (unsupported, and TileLang does not target it), WebGPU (experimental
in TileLang, and DX12/WebGPU on RDNA2 is a poor performance bet).

**Conclusion: do not plan around running TileLang kernels on the local RDNA2.** The local
machine's job is compile/IR/packaging research, which §1 shows it does well. When AMD
support is wanted, the honest routes are (a) the `cpu` backend as a cheap second provider
for ABI validation, and (b) a rented MI300X or RDNA4 box — not this machine.

That said, this shapes the plan in a useful way: **the `cpu` backend is the cheapest
possible second provider**, and Risk 3 (provider ABI becoming CUDA-shaped) demands two
substantially different implementations *before* the ABI is stabilised. `cuda` + `cpu` is
enough to start that test locally, and it costs nothing extra.

---

## Sources

- https://tvm.apache.org/2026/06/22/tirx
- https://tvm.apache.org/docs/tirx/overview.html · `/arch/lowering_pipeline.rst` · `/arch/tile_dispatch.rst` · `/arch/backends.rst`
- https://github.com/apache/tvm · https://github.com/tile-ai/tilelang
- https://tilelang.com/get_started/Installation.html
- https://pypi.org/pypi/tilelang/json · `/apache-tvm/json` · `/torch/json` · `/z3-solver/json`
- Measured locally: `experiments/p0/out/results.json`, `experiments/p0/out/_traces/`

