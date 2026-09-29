# E13 — Opaque CUDA artifact prototype

**Host:** Windows 11, RX 6700 XT, no NVIDIA driver, no `nvcc`.
**Date:** 2026-09-29. **Status:** local checks passed; GPU checks unverified.

## Findings

The experiment now has a producer, an independent compiler-free consumer,
and a later-NVIDIA runbook. All code remains under `experiments/p0/`; no public
runtime ABI or product package is being frozen.

The source bundle for `relu(2*a+b)` at size 129 is about **3.6 MB compressed**.
Its header payload is about **27.2 MB uncompressed**. Generated TileLang CUDA
includes CUTLASS/CuTe headers through `tl_templates/cuda/common.h`, so the
original E4 measurement of only the 896.7 KB TileLang template tree was not a
complete source distribution footprint. Both trees and license notices are
bundled. The cubin tier excludes those headers and retains the notices; its
real size has not been measured because compilation needs `nvcc`.

The source bundle reloads and verifies every payload hash in a Python 3.12
environment containing **only NumPy 2.5.3**, with TileLang, TVM, TVM FFI and
PyTorch absent. The consumer also actively rejects importing those modules.
This establishes reader isolation. It does not establish CUDA execution.

The local suite reports **18 passed, 6 skipped**. It covers corrupted and
incompatible artifacts, unsafe archive paths, preservation of previous
artifacts, import isolation, CUDA argument-address packing, stream ordering,
cleanup on failures, and genuine lowering of partial-tile workloads for
`sm_80`, `sm_90` and `sm_100`. Driver test doubles establish host-side
contracts, not generated-kernel correctness. The six opt-in NVIDIA checks
exercise the opaque size matrix and corrected workload numerics.

The focused harness reports **33 ok, 0 unexpected** for `codegen`,
`artifact_shape`, `artifact_versioning` and `symbolic_shapes`, including all
five corrected workloads across three CUDA targets. The artifact round-trip
now compares both target outputs, and the fresh-process reload must reproduce
both expected digests. Reports include source hash, lock hash, Git revision,
dirty state, Python version and installed package versions.

## Workload changes

- Elementwise uses guarded direct accesses and no shared buffers.
- Row sum replaces the uninitialized, racy scalar accumulation and out-of-row
  indexing with a padded, zero-filled TileLang reduction. It is named
  `row_sum`, rather than implying softmax.
- Gather tiles both rows and columns, uses N as the source row count, guards
  partial tiles, and writes zero for invalid indices.
- GEMM guards bias accesses in partial output tiles.
- Attention now uses `[heads, sequence, head_dim]` Q/K/V/O, scaled QK scores,
  masked key tails, online maximum/normalizer rescaling, and probability-times-V
  accumulation. It remains a numerically unverified correctness candidate.
- The symbolic source probe uses less shared memory and verifies serialization
  of its runtime extent. Identical output from a substitution probe is no
  longer treated as proof of runtime shape reuse.

## Still unverified

`nvcc` compilation, actual GPU numerics, real resource limits and launch
behavior, timing, two-host executable transfer, symbolic-shape execution,
module fusion, and cross-provider portability. A no-device run writes an
explicit skipped result rather than a successful numerical result.

The two-host milestone needs a CUDA toolkit on the producer (no GPU required)
and an NVIDIA driver/device on the consumer. Building on the future NVIDIA
host first is a useful interim check but does not establish executable
transfer between hosts.

## Reproduce

```powershell
uv sync --locked
uv run --locked python -m pytest
uv run --locked python -m experiments.p0.harness --only codegen artifact_shape artifact_versioning symbolic_shapes --out experiments/p0/out/milestone-results.json
uv run --locked python -m experiments.p0.artifact_build prepare --size 129 --arch sm_80 --out experiments/p0/out/source-129.zip
```

Raw local results live under ignored `experiments/p0/out/`. The later-NVIDIA
commands and measurement definitions are in the
[validation runbook](../plan/opaque-artifact-validation.md).
