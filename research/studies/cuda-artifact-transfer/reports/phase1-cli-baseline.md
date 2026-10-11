# Phase 1 CLI baseline — A10G

**Historical v1 baseline.** Phase 1 is now complete with typed scalar arguments,
symbolic shapes and GPU DLPack/stream interoperability. The final acceptance
checks and current measurements are in [the exit report](phase1-exit.md).
The earlier measurements below are retained as the initial static baseline.

**Run:** 2026-09-29, Linux, Python 3.12.14, NVIDIA A10G (`sm_86`), pinned
TileLang 0.1.14 and TVM FFI 0.1.12 producer. The consumer virtual environment
contained only `tensor-workspace==0.1.0` and `numpy==2.5.3` (confirmed by
`uv pip list`). The local CUDA 12.9 build components were selected with
`CUDA_HOME`. These measurements use the 129-element float32 example in
`examples/elementwise.py`; they do not predict other workload sizes.

## Reproduction

```bash
uv sync --locked
uv run --locked python tools/bootstrap_cuda.py --out build/cuda-12.9
export CUDA_HOME="$PWD/build/cuda-12.9"
uv build --wheel
uv venv --python 3.12 build/consumer-venv
uv pip install --python build/consumer-venv/bin/python dist/tensor_workspace-0.1.0-py3-none-any.whl
uv run --locked python scripts/validation/phase1_measure.py \
  --runtime-python build/consumer-venv/bin/python --out build/phase1-metrics.json
```

The report script uses an isolated cache for a cold and a warm build, runs the
wheel in three fresh consumer processes, verifies all 129 values against NumPy,
and benchmarks 100 launches after 10 warmups. It times the same host in this
report; the separate GitHub Actions producer is the two-host check.

| Measurement | Result |
|---|---:|
| Cold build, inside command | 1.670 s |
| Warm build, inside command | 0.304 s |
| Cold build, whole process | 4.264 s |
| Warm build, whole process | 2.832 s |
| Cold `nvcc` stage | 1.354 s |
| Consumer process wall, three runs | 0.734 / 0.412 / 0.466 s |
| First result after Python module entry, three runs | 0.439 / 0.187 / 0.190 s |
| Median host enqueue | 24.0 µs |
| Median host launch plus stream sync | 29.2 µs |
| Maximum absolute error | 0.0 |

The warm build skips `nvcc`; it still imports TileLang and lowers the frontend
to validate the cache key. The consumer process wall includes interpreter
startup, NumPy import, artifact verification, context creation, execution,
output serialization and cleanup. First-result time starts inside the `run`
command, so it excludes interpreter startup. Enqueue timing includes Python
argument validation and CUDA Driver API launch; it is not GPU kernel duration.
The first fresh consumer process was slower than the next two; filesystem and
driver caches were not reset. These are single-run A10G observations, not
throughput guarantees. The build cache key for this run was
`544021380da14ae80e5f86a4bd061af56e5d513b51869b83504b3fbb72f0bdb2`.

## Diagnostic probes

Five deliberate failures all exited 1 with a named cause and no Python
traceback: missing `tensor_export()`, invalid target, wrong input shape, missing
input, and corrupt cubin hash. Their messages identified the required export,
an example SM target, expected shape/dtype, expected input names, and the
corrupt member respectively. This is a 5/5 cause-identification check, not a
user study or a claim that all compiler diagnostics are clear.

## Initial v1 boundary

The first product export profile accepts one static kernel with pointer-only
buffer arguments. Grid, block and dynamic shared memory are extracted from
lowered TileLang metadata. A 64×64 float16 GEMM plus bias/ReLU artifact also
passed a NumPy comparison on the A10G, with maximum absolute error
`5.96e-8` for the tested inputs. Artifacts require an exact SM match.
The wheel can execute without TileLang, TVM FFI, PyTorch or
CUDA development headers; a build still needs the pinned compiler packages
and a full CUDA toolkit. GPU DLPack borrowing and broader symbolic/scalar
exports were still open at this baseline; the exit report records their completion.

## Two-host product artifact transfer

[GitHub Actions run 36616282091](https://github.com/kreasof-ai/tensor/actions/runs/36616282091)
built the product `.tbin` and wheel on the GPU-free Ubuntu producer
`runnervmtr4k5` at commit `4cfb607d0445997a206f15c28237ba578846294a`.
The consumer host was `default`, an NVIDIA A10G (`sm_86`). The downloaded
artifact and wheel SHA-256 values matched `producer.json`:

| File | SHA-256 |
|---|---|
| `elementwise.tbin` | `dfb86bc484641086e8f59bb4ee71993eac76bff34c6cf336a78007706273fd98` |
| `tensor_workspace-0.1.0-py3-none-any.whl` | `54485c8d6830ba3fc6a0a6ab9b25d5354fabe1c8c00624d8aecbf177c6d7f865` |

A new Python 3.12 virtual environment installed that downloaded wheel with
only `numpy==2.5.3` and `tensor-workspace==0.1.0`. Its `tensor doctor` reported
`run_ready` despite missing compiler packages and CUDA development headers.
`tensor run` consumed the downloaded cubin and produced all 129 values with
maximum absolute error `0.0` against NumPy. A fresh in-process consumer check
confirmed no `tilelang`, `tvm`, or `torch` imports after execution. The CLI
reported 0.191 s from command entry to first result on this run. A 100-launch
benchmark reported 23.6 µs median host enqueue and 29.1 µs median launch plus
stream synchronization. These are one-host consumer measurements of a
two-host transfer, not a cross-GPU performance claim.

To repeat the transfer check, run the workflow with `arch=sm_86`, download its
`tensor-cuda-sm_86` artifact, compare `producer.json` against both downloaded
files and the local commit/hostname, install its wheel into a clean Python 3.12
environment, then run its `.tbin` with `a.npy` and `b.npy` float32 arrays of
shape `(129,)`. Check `c.npy` against `maximum(2*a+b, 0)`.

## Clean-install smoke beyond the A10G host

[GitHub Actions run 36616925428](https://github.com/kreasof-ai/tensor/actions/runs/36616925428)
passed a second GPU-free product build followed by clean-wheel installs on
Ubuntu 24.04 and Windows runners. Each consumer job installed only NumPy and
Tensor, verified both downloaded payload hashes and the `sm_86` manifest, and
confirmed no compiler imports. This establishes cross-platform installation
and inspection of the pure-Python wheel, **not** CUDA execution on Windows or
another GPU. CUDA execution evidence remains the transferred A10G run above.

## Broader export boundary at the initial baseline

The initial v1 artifact deliberately had only static buffer shapes and pointer
arguments. A TileLang `T.dynamic` elementwise kernel lowers successfully, but
its CUDA signature adds an `int size` parameter and its grid extent is the
runtime expression `(size + 127) // 128`. The v1 product rejected that
signature and symbolic grid instead of silently launching it with static
metadata. Supporting it requires a typed scalar argument descriptor, a binding
between buffer shape and that scalar, and a runtime launch-expression contract.
The v2 artifact and runtime now implement that binding contract; see
[the Phase 1 exit report](phase1-exit.md).
