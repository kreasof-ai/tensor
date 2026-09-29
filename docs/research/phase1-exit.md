# Phase 1 exit — CUDA CLI, runtime dimensions and GPU interop

**Status: complete for the single-device NVIDIA scope.** The five CLI commands,
content-addressed cache and kernel workbench execute on the NVIDIA A10G
(`sm_86`). The product now handles runtime scalar arguments and symbolic buffer
dimensions, imports GPU DLPack buffers, and orders work across foreign CUDA
streams. The provider-neutral stable ABI is the next phase.

## Acceptance evidence

Product implementation commit:
`87803a66adbf20bce4f2523faa0e9575bf969f87`.
[GitHub Actions run 36627654075](https://github.com/kreasof-ai/tensor/actions/runs/36627654075)
passed all three jobs: a GPU-free producer and clean-wheel installation checks
on Ubuntu and Windows. The producer built five v2 artifacts: static elementwise,
static GEMM/bias/ReLU, dynamic affine, dynamic GEMM and an int64 scalar offset.

The consumer installed the downloaded wheel into a fresh Python 3.12
environment containing exactly `tensor-workspace==0.1.0` and `numpy==2.5.3`.
The [raw transfer report](data/phase1-exit.json) verifies distinct hostnames,
clean matching commits, source and lock hashes, every artifact hash, and the
wheel hash. An import guard rejects TileLang, TVM, TVM FFI and PyTorch imports
throughout the consumer execution.

All **22 numerical/compatibility cases**, the CLI scalar execution check, and
**8 expected runtime failures** passed. The numerical cases cover:

| Profile | Cases | Maximum absolute error |
|---|---:|---:|
| Static elementwise | 1 | 0.0 |
| Dynamic affine: lengths 1, 127, 128, 129, 1025 × two scalar values | 10 | 4.77e-7 |
| Dynamic GEMM: rows 1, 31, 32, 33, 65 | 5 | 1.22e-4 |
| Static GEMM plus bias/ReLU | 1 | 5.96e-8 |
| Positive and negative int64 offsets beyond 32 bits | 2 | 0.0 |
| Legacy and versioned GPU DLPack imports | 2 | 0.0 |
| Existing v1 product artifact | 1 | 0.0 |

Float32 dynamic affine comparisons use `rtol=1e-6, atol=1e-6`; float16 GEMM
comparisons use `rtol=1e-2, atol=1e-2`. Scalar offsets use exact equality. These
are scoped numerical probes with fixed inputs, not exhaustive kernel proofs.

The complete local suite, with both Phase 0 and Phase 1 CUDA checks enabled,
passed **72 tests, zero skips, in 43.05 s**. This includes product tests using a
PyTorch client on independent producer and consumer streams. The tests delay
the producer, import its allocations, wait for later updates, launch Tensor,
and consume the output on another stream before a Tensor host synchronization.
They check unchanged buffer addresses, both owned and borrowed launch streams,
preserved primary contexts, and borrowed streams usable after Tensor cleanup.
Separate tests check managed-tensor deleters run once on successful imports
and rejected metadata, for legacy and versioned capsules.

## Product contracts

New `.tbin` files use format version 2. In addition to the v1 cubin, frontend
TIRx, compiler versions, notices and hashes, they record:

- frontend buffer and scalar arguments;
- the actual lowered CUDA parameter order and scalar widths;
- integer symbols, shape and launch expressions;
- required pointer alignment.

The producer checks its metadata against the emitted CUDA signature. The
consumer infers direct dimension variables from input shapes, checks all
shared dimensions and explicitly supplied values, allocates the corresponding
outputs, and binds implicit CUDA dimension arguments. Expressions are bounded
JSON trees for integer arithmetic and lossless casts; no Python expression
evaluation or compiler imports are used. Integer overflow, invalid geometry,
missing dimensions and incompatible buffer shapes fail before launch.

Scalar arguments support bool, signed/unsigned 8–64-bit integers and
float32/float64. Buffers additionally support float16. One compiled dynamic
binary serves each workload's tested shapes; consumers never recompile it.
The updated consumer also reads existing v1 static artifacts. Older v1
consumers reject v2 artifacts through the explicit format version check.

GPU DLPack imports follow the
[DLPack Python ownership and stream protocol](https://dmlc.github.io/dlpack/latest/python_spec.html)
and [legacy/versioned structure layouts](https://dmlc.github.io/dlpack/latest/c_api.html).
The importer negotiates the version, passes its stream to the producer,
consumes the capsule once, and retains the managed tensor until release. GPU
imports share storage; CPU imports upload a contiguous copy. CUDA sessions
retain the primary context and either create a stream or borrow an external
handle. `wait_for` and `handoff` establish event dependencies. Cleanup waits
for session and handed-off consumer work, releases managed tensors once,
destroys only owned streams/modules, releases its context reference, and
restores the previous context.

## Installation metrics

Starting with downloaded wheel/artifacts, Python 3.12, `uv`, and a working
NVIDIA driver on the target GPU, first execution takes **four commands**:

1. Create the consumer virtual environment.
2. Install the Tensor wheel; NumPy is resolved automatically.
3. Generate the example `.npy` inputs.
4. Run `tensor run`.

Retrieving the Actions bundle with `gh run download` adds one command and
requires an authenticated GitHub CLI. Counting only application packages,
the user explicitly selects **one package**, the Tensor wheel, and manually
installs **zero dependency packages**. The resulting environment has **two
Python distributions**. Python, `uv`, the NVIDIA driver and the available GPU
are prerequisites outside those counts.

```bash
uv venv --python 3.12 build/consumer
uv pip install --python build/consumer/bin/python downloaded/dist/tensor_workspace-0.1.0-py3-none-any.whl
build/consumer/bin/python -c 'import numpy as np; np.save("a.npy", np.arange(129, dtype="float32")); np.save("b.npy", np.ones(129, dtype="float32"))'
build/consumer/bin/python -m tensor run downloaded/build/transfer/dynamic_affine.tbin \
  --input a=a.npy --input b=b.npy --scalar scale=2.5 --out-dir build/result
```

A producer needs the pinned compiler packages, full CUDA build components and
a host C++ compiler. `uv sync --locked` installs the compiler dependencies;
`tools/bootstrap_cuda.py` supplies the selected NVIDIA build components.
Ubuntu/Windows CI validates installation and inspection without a GPU. GPU
execution evidence is the A10G consumer.

## Final latency measurements

The [raw metrics](data/phase1-metrics.json) use the 129-element static float32
example, an isolated cold/warm cache, three fresh consumer processes, and 100
launches after 10 warmups. The consumer contains only Tensor and NumPy.

| Measurement | Result |
|---|---:|
| Cold/warm build, inside command | 1.659 / 0.308 s |
| Cold/warm build, whole process | 4.295 / 2.975 s |
| Cold `nvcc` stage | 1.352 s |
| Consumer process wall, three runs | 0.733 / 0.414 / 0.414 s |
| First result after command entry, three runs | 0.433 / 0.188 / 0.189 s |
| Static median host enqueue | 49.4 µs |
| Static median launch plus stream sync | 54.7 µs |
| Static maximum absolute error | 0.0 |

The warm build skips `nvcc` but still imports and lowers with the compiler.
First-result timing excludes interpreter startup. The first fresh process is
slower; driver/filesystem caches were not reset. Per-launch shape binding,
scalar conversion, context activation and validation remain on the host path.
The earlier v1 static baseline measured about 24 µs enqueue, so the current
typed runtime's additional host cost is visible rather than hidden.

The transferred dynamic affine benchmark measured 69.6 µs median host enqueue
and 74.8 µs median launch plus synchronization. Its CLI first-result time of
90.1 ms was measured after other checks had warmed the context. It is not a
cold-start comparison. These host timings are not GPU kernel duration or
cross-GPU throughput claims.

The eight diagnostic probes identify missing scalar arguments, conflicting
input dimensions, conflicting explicit dimensions, non-finite scalars, wrong
buffer dtype, integer overflow, noninteger scalar input, and unknown arguments.
The initial five CLI failure probes are retained in
[the historical baseline](phase1-cli-baseline.md). This is a cause-identification
check, not a usability study.

## Reproduction and phase boundary

```bash
TENSOR_P0_CUDA=1 TENSOR_P1_CUDA=1 CUDA_HOME=/path/to/cuda-12.9 \
  uv run --locked python -m pytest -o addopts='' -q
gh workflow run phase1-artifact-transfer.yml --ref main -f arch=sm_86
gh run download RUN_ID -n tensor-cuda-sm_86 -D build/downloaded
uv venv --python 3.12 build/transfer-consumer
uv pip install --python build/transfer-consumer/bin/python \
  build/downloaded/dist/tensor_workspace-0.1.0-py3-none-any.whl
build/transfer-consumer/bin/python tools/phase1_transfer_check.py \
  build/downloaded --out build/transfer-check.json
```

Use the producer's exact commit and a clean checkout for the transfer audit.
An optional `--legacy-artifact PATH` adds the v1 compatibility case; the
recorded report includes that case. Normal reproduction without the archived
v1 artifact runs 21 numerical cases plus CLI execution and eight failures.

The supported scope is one CUDA kernel per artifact, exact SM matching,
positive dimensions and contiguous buffers with the declared alignment. GPU
DLPack imports require writable same-device allocations in the primary
context; foreign stream handles must outlive the session. General strided or
zero-size buffers, cross-device copies, symbolic GEMM tile sizes, DLPack output
export, and provider-neutral stable ABI/stream descriptors are outside this
Phase 1 profile. The original Phase 1 CLI and the explicitly requested scalar,
symbolic and GPU-import work are complete. Phase 2 can stabilize the measured
runtime contracts; it need not reopen the CLI work.
