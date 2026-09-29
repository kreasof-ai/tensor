# Opaque CUDA artifact validation

This is a Phase 0 experiment, not a public Tensor API. The milestone is:
build a correct kernel on host A, transfer the executable to host B, launch it
without importing the compiler, compare with NumPy, and record startup time.

**Current state:** source generation, bundle reload, compiler isolation and
host-side driver contracts pass locally. CUDA compilation, real GPU numerics,
startup latency and two-host executable transfer remain **UNVERIFIED**.

The first kernel computes `c = maximum(2*a + b, 0)` on contiguous float32
vectors. It has three pointer arguments, 128 threads per block, explicit tail
guards, and zero shared memory. It deliberately does not exercise fusion,
symbolic shapes, streams supplied by another framework, or a second provider.

## What is implemented

- `artifact_build prepare`: TileLang to a transferable source ZIP, including
  TileLang and CUTLASS/CuTe headers and redistribution notices. No GPU/toolkit.
- `artifact_build compile`: source ZIP to a precompiled cubin bundle with
  `nvcc`. Only stdlib Python and the CUDA build toolchain are needed.
- `artifact_build build`: both stages on one host. No GPU is needed to build.
- `artifact_run inspect`: checks the manifest and every payload hash.
- `artifact_run doctor`: reports the device architecture or an explicit skip.
- `artifact_run validate`: direct CUDA Driver API load/launch, NumPy reference,
  first-result timing and warm launch-plus-synchronization timing. Imports of
  TileLang, TVM, TVM FFI and PyTorch are actively blocked.
- `artifact_check`: launches fresh consumer processes and saves their reports.
- `numerics`: twelve NVIDIA-only checks for all five corrected workloads.

Source bundles and executable bundles have different manifest kinds. The
consumer rejects a source bundle. The experimental cubin contract requires
an exact SM match and is independently format-versioned. Producer package
versions, lock hash, Git revision, dirty state and source hash are recorded;
the consumer does not need those compiler packages installed.

## Local checks

From the repository root:

```powershell
uv sync --locked
uv run --locked python -m pytest
uv run --locked python -m experiments.p0.artifact_build prepare --size 129 --arch sm_80 --out experiments/p0/out/source-129.zip
uv run --locked python -m experiments.p0.artifact_run inspect experiments/p0/out/source-129.zip
```

Outputs are created exclusively: choose a new filename for another build.
`uv run --locked python -m experiments.p0.artifact_run doctor` reports a skip
on the current Windows/AMD host. Direct Python CLI exit codes are 0 for a
successful check, 1 for a failure, and 2 for unavailable hardware.

## Full two-host executable milestone

Host A needs the locked Python environment, `nvcc`, and the host C++ compiler
required by the CUDA toolkit. It does not need an NVIDIA GPU. Host B needs
64-bit Python 3.12, NumPy and an NVIDIA driver. Use Linux for the initial GPU
check; on Windows run builds from a configured MSVC developer shell.

On **B**, clone or copy this repository and create a consumer environment
containing only NumPy. Example commands for a Linux host:

```bash
uv venv --python 3.12 experiments/p0/out/runtime-venv
uv pip install --python experiments/p0/out/runtime-venv/bin/python numpy==2.5.3
experiments/p0/out/runtime-venv/bin/python -m experiments.p0.artifact_run doctor
```

On Windows, replace `bin/python` with `Scripts/python.exe`. Use the exact SM
reported by doctor in the following build commands (`sm_80` is an example).
Use the consumer interpreter directly: `uv run` uses the compiler workspace.

On **A**, build the boundary-size matrix:

```powershell
foreach ($size in 1,127,128,129,1025) {
    uv run --locked python -m experiments.p0.artifact_build build --size $size --arch sm_80 --out "experiments/p0/out/elementwise-$size.tbin"
    if ($LASTEXITCODE -ne 0) { throw "Build failed for size $size" }
}
```

Transfer the `.tbin` files to **B**. Keep the same experiment scripts on both
hosts; the reports record the producer's source and lock hashes. On B:

```bash
experiments/p0/out/runtime-venv/bin/python -m experiments.p0.artifact_run validate experiments/p0/out/elementwise-129.tbin --report experiments/p0/out/first-result.json
experiments/p0/out/runtime-venv/bin/python -m experiments.p0.artifact_check experiments/p0/out/elementwise-*.tbin --runtime-python experiments/p0/out/runtime-venv/bin/python --runs 3 --report experiments/p0/out/artifact-matrix.json
```

A passing report must show NumPy agreement, no compiler imports, matching SM,
and real timing values for each boundary size. Check the producer/consumer
hostnames to confirm the two-host condition; a one-host run validates launch
and isolation but does not establish cross-host transfer.

`process_entry_to_first_result_seconds` includes consumer module imports,
NumPy import, artifact validation, context setup, loading, allocation, input
upload, launch, synchronization and result download. It starts at Python
module entry, so it excludes interpreter startup. `process_wall_seconds`
includes interpreter startup **and** reference checks, warm iterations,
reporting and teardown. It is not time to first result. Warm timing measures
host launch plus stream synchronization, not GPU-only kernel throughput.
Repeated processes are fresh interpreters, not guaranteed cold filesystem or
driver caches. Record machine state before comparing timings.

## First GPU check with the existing source bundle

The local host has no `nvcc`. As an interim check, transfer its source ZIP to
the future NVIDIA host and compile there using the clean consumer interpreter:

```bash
experiments/p0/out/runtime-venv/bin/python -m experiments.p0.artifact_build compile experiments/p0/out/source-129.zip --out experiments/p0/out/elementwise-129.tbin
experiments/p0/out/runtime-venv/bin/python -m experiments.p0.artifact_run validate experiments/p0/out/elementwise-129.tbin --report experiments/p0/out/first-result.json
```

This checks source transfer, compilation and compiler-free execution. It does
not complete the separate two-host **executable** transfer milestone.

## Validate the corrected workload suite on NVIDIA

In the compiler environment on the NVIDIA host:

```bash
uv sync --locked
uv run --locked python -m experiments.p0.numerics --report experiments/p0/out/numerics-nvidia.json
TENSOR_P0_CUDA=1 TENSOR_P0_RUNTIME_PYTHON=experiments/p0/out/runtime-venv/bin/python uv run --locked python -m pytest tests/test_nvidia.py
```

The numeric suite uses PyTorch only as TileLang's existing device-buffer
client. Expected values come from NumPy. It checks elementwise and GEMM partial
tiles, reductions at widths 1/127/128/129/1024, gather tails and invalid indices,
and attention at sequence lengths 64/65. The opaque consumer uses no PyTorch.

Passing these checks establishes correctness for these cases. It does not
establish a tuned performance baseline, fusion, a general provider ABI, or a
torch-free compiler. Those remain separate experiments.
