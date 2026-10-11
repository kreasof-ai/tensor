# E14 — NVIDIA execution and compiler-free startup

**Measured:** 2026-09-29. **Host:** Linux x86-64, NVIDIA A10G (`sm_86`,
23,028 MiB), NVIDIA driver 595.91.07. Python 3.12.14, TileLang 0.1.14,
TVM FFI 0.1.12, PyTorch 2.14.0, NumPy 2.5.3. CUDA compiler 12.9.86,
runtime headers 12.9.79, CCCL 12.9.27; host `g++` reports 13.3.0.

**Result:** actual cubin compilation, opaque execution, and all twelve
workload-reference cases pass. This is a **single-host** experiment with
independent producer/compiler/consumer processes. Two-host executable transfer
remains unverified.

The retained [measurement data](data/e14-cuda-execution.json) contains every
timing sample, workload error/tolerance, artifact hash, source and lock hash,
Git revision/dirty state, compiler options and package versions. Full local
reports and bundles live under ignored `experiments/p0/out/a10g-20260929/`.

## Build environment failure and fix

The environment's initial `/opt/conda/bin/nvcc --version` succeeded, but actual
compilation failed with `cuda_runtime.h: No such file or directory`.
Matching NVIDIA redistribution components were downloaded, SHA-256 checked
against the [CUDA 12.9.1 manifest](https://developer.download.nvidia.com/compute/cuda/redist/redistrib_12.9.1.json),
and assembled locally under `experiments/p0/out/cuda-12.9/`: `cuda_nvcc`,
`cuda_cudart` and `cuda_cccl`. NVIDIA describes these components in its
[archive installation guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#tarball-and-zip-archive-deliverables).
The system toolkit and project dependency pins were not changed.

`artifact_build doctor` now compiles a small kernel using runtime, half/FP8
and CCCL headers and verifies the resulting ELF cubin. It requires neither
GPU access nor compiler Python packages. Compiler selection is explicit
`--nvcc`, then `CUDA_HOME`/`CUDA_PATH`, then `PATH`. Invalid configured roots
fail with a diagnostic; missing-header failures retain the original compiler
message and explain the required development components.

## Opaque executable results

All five source bundles were emitted in the compiler environment. All five
cubins were then compiled and packaged in the NumPy-only environment with
imports of TileLang, TVM, TVM FFI and PyTorch actively blocked. That environment
contains only NumPy 2.5.3, and the four compiler packages are absent.

Each executable was validated in three fresh consumer interpreters (seeds
0/1/2, twenty warm iterations per run). All **15 runs pass**, with zero maximum
absolute error and no compiler imports.

| Extent | Bundle bytes | Source-to-bundle build (s) | Median first result (ms) | Median module load (ms) | Median warm launch + sync (µs) |
|---|---:|---:|---:|---:|---:|
| 1 | 4,564 | 1.503 | 233.66 | 0.126 | 11.86 |
| 127 | 4,607 | 1.502 | 237.77 | 0.123 | 11.98 |
| 128 | 4,544 | 1.491 | 236.89 | 0.156 | 11.97 |
| 129 | 4,655 | 1.528 | 242.53 | 0.133 | 11.97 |
| 1025 | 4,654 | 1.497 | 233.99 | 0.128 | 12.02 |

Build time includes source-bundle validation/extraction, `nvcc`, and executable
packaging; it excludes TileLang source emission. Source ZIPs are about 3.6 MB.
Executable ZIPs contain the cubin, manifest and redistribution notices without
compiler headers.

First-result time starts at consumer module entry: it includes imports,
NumPy input/reference preparation, artifact verification, context setup,
load/allocation/upload, launch/synchronization and download, but excludes
interpreter startup. Module loading is measured separately. Warm timing is
host launch plus stream synchronization, **not GPU-only throughput**. Fresh
interpreters do not guarantee cold filesystem or driver caches. The workload
is the fixed float32 `relu(2*a+b)` kernel; these numbers are not a GEMM or
attention startup baseline or a comparison with direct TileLang.

## Workload correctness

The existing TileLang/PyTorch device adapter ran all twelve NumPy-reference
cases with the TileLang cache disabled:

- Elementwise: aligned and partial tiles; maximum absolute error 0.
- GEMM + bias + ReLU: `64×64×32` and `65×67×33`; maximum error 0.0009765625,
  with `rtol=atol=0.01`.
- Row sums: widths 1/127/128/129/1024; maximum error 0.000003814697265625,
  with `rtol=atol=0.00001`.
- Gather: partial tiles and invalid indices; exact NumPy agreement.
- Non-causal attention: sequence lengths 64/65, two heads, head dimension 64;
  maximum error 0.00048828125, with `rtol=atol=0.01`.

The full suite reports **32 passed, 0 skipped** in 54.33 seconds, including the
new compiler selection/error cases and a real compiler probe with compiler
imports blocked. This establishes these cases on A10G, rather than general
correctness across shapes, architectures or dtypes.

## Reproduce

Install a full matching toolkit and select its root. These commands use the
local toolkit and NumPy-only environment created for this experiment:

```bash
export CUDA_HOME="$PWD/experiments/p0/out/cuda-12.9"
uv sync --locked
uv venv --python 3.12 experiments/p0/out/review-runtime
uv pip install --python experiments/p0/out/review-runtime/bin/python numpy==2.5.3
experiments/p0/out/review-runtime/bin/python -m experiments.p0.artifact_build doctor --arch sm_86
TILELANG_DISABLE_CACHE=1 TENSOR_P0_CUDA=1 TENSOR_P0_RUNTIME_PYTHON="$PWD/experiments/p0/out/review-runtime/bin/python" uv run --locked python -m pytest
TILELANG_DISABLE_CACHE=1 uv run --locked python -m experiments.p0.numerics --report experiments/p0/out/numerics-a10g.json
```

For a fresh artifact matrix, choose an unused output directory (bundles use
exclusive creation), then emit in the compiler environment and compile in the
NumPy-only environment:

```bash
for size in 1 127 128 129 1025; do
    uv run --locked python -m experiments.p0.artifact_build prepare --size "$size" --arch sm_86 --out "experiments/p0/out/reproduce-a10g/source-$size.zip" || break
    experiments/p0/out/review-runtime/bin/python -m experiments.p0.artifact_build compile "experiments/p0/out/reproduce-a10g/source-$size.zip" --out "experiments/p0/out/reproduce-a10g/elementwise-$size.tbin" || break
done
experiments/p0/out/review-runtime/bin/python -m experiments.p0.artifact_check experiments/p0/out/reproduce-a10g/elementwise-{1,127,128,129,1025}.tbin --runtime-python experiments/p0/out/review-runtime/bin/python --runs 3 --report experiments/p0/out/reproduce-a10g/artifact-matrix.json
```

The [two-host runbook](../plan/opaque-artifact-validation.md) defines the next
transfer check. Fusion, symbolic-shape execution, a second provider, tuned
performance, and a torch-free producer remain separate work.
