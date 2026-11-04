# AGENTS.md — Tensor

Read this file completely before modifying `tensor`. Tensor is a compiler-and-runtime foundation (`TileLang kernel → Tensor compiler → .tbin → Tensor runtime → device`) with strict separation between compilation, runtime execution, and downstream model engines.

---

## 1. Scope & Ownership Boundary (`tensor` vs `akas`)

- **`tensor` owns:**
  - `src/tensor/compiler/`: frontend signature/domain/view extraction, source fingerprinting (`export_source`), schedule exploration (`search.py`, `tuning.py`), build caching (`cache.py`), and backend lowering (`cuda/`, `webgpu/`, `cpu.py`).
  - `src/tensor/artifacts/`: `.tbin` executable format, manifest/checksum validation, portable TIRx preflight, and inspection.
  - `src/tensor/runtime/`: buffer/view/prefix/column-copy lifetimes, bound call descriptors (`abi.py`), symbolic shape domain checking (`domain.py`), DLPack interop (`dlpack.py`), and manual backward (`manual.py`).
  - `src/tensor/providers/`: hardware device allocation, streams, CUDA graph capture/replay, and WebGPU prepared command plans.
  - `src/tensor/modules/`: `.tpack` manifests, guarded AOT shape portfolios (`portfolio.py`), lock graphs, and registry transport.
  - `packages/`: optional `tensor-torch` adapter and `tensor-nn` training templates.
- **Compiler features only:** Any change added to `tensor` for downstream workloads must be a **general compiler, ABI, or runtime feature** (e.g., Hopper MMA/TMA lowering attributes, checked buffer views, shape domains, cache conditioning).
- **Never add model-specific or challenge-specific kernels to `tensor`** (see commit `7997197`). Model architectures, checkpoint formats, quantization codecs, MoE/attention model kernels, and serving belong in `akas`.

---

## 2. Do Not Duplicate Existing Tensor Facilities

Before adding new modules or helpers, check what `tensor` already provides:

| Capability | Existing Canonical Location | Notes |
| --- | --- | --- |
| Schedule beam search & profiles | `tensor.compiler.search` (`ScheduleSearch`, `ScheduleProfile`) | Proposes schedules; caller owns compilation, correctness, and timing. |
| Correctness-gated kernel tuning & CUDA event timers | `tensor.compiler.tuning` (`measure`, `measure_cuda`, `measure_cuda_sequence`, `cuda_working_set`, `tune`, `tune_shapes`, `select_regions`) | Includes L2 working-set conditioning (`cuda_working_set`) and DP region selection for dynamic shapes. |
| Opt-in CUDA GEMM prelowering | `tensor.compiler.cuda.prelower`, `tensor.compiler.cuda.matmul`, `tensor.compiler.cuda.schedules` | Supports `matmul`, `matmul_partials`, `matmul_reduce`, `orientation` (`weight` vs `activation`), FP16/BF16 Tensor Core and FP32 SIMT FMA. |
| CUDA lowering & Hopper attributes | `tensor.compiler.cuda.lowering`, `tensor.compiler.cuda.build` | Honours `tensor.cuda.gemm_mode` (`"auto"` / `"mma"`) and `tensor.cuda.tma` (`sm_90a`). |
| WebGPU schedules & WGSL passes | `tensor.compiler.webgpu.schedules/*`, `tensor.compiler.webgpu.passes/*`, `tensor.providers.webgpu.plan` | One schedule family per file in `schedules/`; barrier/uniformity passes in `passes/`. |
| Dependency-tracked source export | `tensor.compiler.frontend.source:export_source` | Embeds implementation module SHA-256 hashes into specialization files. |
| Buffer views, prefixes & column copies | `tensor.runtime.buffer` (`view`, `prefix`, `copy_from`, `copy_columns_from`) | Zero-copy views/prefixes and ordered device-side pitched/row copies. |

---

## 3. Strict Architectural & Packaging Invariants (Enforced by Tests & CI)

### A. Compiler-Free Core Runtime & CLI (`tests/integration/test_layout.py`, `tools/validation/distribution_check.py`)
- The core `tensor-workspace` wheel depends **only on `numpy`**.
- Importing `tensor`, `tensor.cli`, `tensor.runtime.*`, `tensor.providers.*`, `tensor.artifacts.*`, or `tensor.modules.*` **must never** import `tensor.compiler`, `tilelang`, `tvm`, `tvm_ffi`, `torch`, `triton`, `wgpu`, `tensor_nn`, `tensor_torch`, or `akas`.
- Frontend deserialization and TileLang/TVM imports may only occur inside `src/tensor/compiler/` during an explicit build/compile/inspect request.

### B. Canonical Module Layout (`tests/integration/test_layout.py`)
- Keep all code in the grouped canonical hierarchy (`runtime.*`, `providers.*`, `artifacts.*`, `modules.*`, `compiler.*`, `cli.*`).
- **Never** reintroduce retired flat module aliases (`tensor.abi`, `tensor.cuda`, `tensor.nn`, `tensor.compiler.lowering`, `tensor.compiler.entry`, `tensor.compiler.nvrtc`, `tensor.compiler.webgpu_lowering`, `tensor.compiler.webgpu_templates`, `tensor.artifacts.modules`, `tensor.artifacts.registry`, `tensor.providers.webgpu_contract`, `tensor.cli.commands`).

### C. Compiler Identity & Artifact Integrity (`src/tensor/artifacts/format.py`, `src/tensor/compiler/cuda/build.py`)
- Any change to lowering passes, pass configs, prelowering (`prelower.py`, `matmul.py`, `schedules.py`), or codegen attributes **must** participate in the compiler cache key and artifact metadata (`lowering_identity()`, `pass_configs`, `source_sha256`). Stale `.tbin` files must never be silently reused across different lowering semantics.
- Prelowering (`lower_cuda_operations`) must verify `tvm.ir.structural_equal` against the canonical logical operation before substituting a schedule. Never allow an attribute annotation to replace arbitrary computation or silently cast FP32 inputs to TF32/FP16.
- Format versions and runtime capabilities (`format_version`, `runtime_abi`, `required_capabilities` such as `cuda_tensor_maps`) must fail fast with a clear error when loaded by an incompatible consumer or device.
- Bound call descriptors (`Executable.bind`) validate buffer shapes, dtypes, alignment, symbolic `shape_domain` bounds/relations, and hardware launch limits (grid/block dimensions, threads per CTA, dynamic shared memory) before output allocation, launch, or CUDA graph capture.

### D. Documentation vs Historical Research Archive
- `docs/` describes current usage and contracts only.
- `research/` (`research/studies/`, `research/decisions/0001..0017`, `research/plans/`, `research/phase0/`) contains sealed historical studies, evidence bytes, and architecture decision records. Do not overwrite historical evidence or dump scratch tuning logs into `docs/` or `research/`.
- All disposable build outputs, NVRTC downloads (`build/nvrtc-12.9`), and temporary wheels belong in `build/`.

---

## 4. Required Verification Before Completing Work

Run these checks locally before declaring any change to `tensor` complete:

```sh
# 1. Fast layout, compiler-free import guard, and runtime ABI tests
uv run --locked python -m pytest \
  tests/runtime \
  tests/artifacts \
  tests/integration/test_layout.py \
  tests/integration/test_doctor.py

# 2. Compiler and provider unit/regression tests
uv run --locked python -m pytest tests/compiler tests/providers tests/modules

# 3. Full distribution & isolated compiler-free consumer validation (when touching packaging/runtime/exports)
uv run --locked --no-default-groups python tools/validation/distribution_check.py
```
