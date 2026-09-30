# Phase 2 validation runbook

Use Python 3.12 and the repository lock. NVRTC compilation does not need a
GPU or CUDA driver when an exact target is supplied. Execution needs the
driver and a GPU matching that target. The measured target is A10G `sm_86`.

## Producer and consumer

```bash
uv sync --locked
uv run --locked python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
uv run --locked tensor doctor --target sm_86 --compiler nvrtc --json
uv run --locked python scripts/validation/phase2_producer.py --out build/phase2-transfer
uv build --wheel
uv venv --python 3.12 build/phase2-consumer
uv pip install --python build/phase2-consumer/bin/python \
  dist/tensor_workspace-0.1.0-py3-none-any.whl
```

Use a new producer output directory. The producer builds elementwise, GEMM
with ReLU, dynamic affine, dynamic GEMM and int64 offset; it checks cold/warm
cache behavior and repairs a deliberately corrupt image. It prohibits
external compiler subprocesses. `--require-isolated` additionally rejects
a driver library or any compiler tool on PATH; run that mode in a clean
container/host, rather than merely hiding the tools on the normal producer.

The local isolated run used the official
`python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f`
base with libstdc++6, libgomp1 and strace. The pinned installed compiler
packages were copied to `/packages`, the NVRTC bundle to `/nvrtc`, and
source/examples/producer script to `/workspace`. No GPU devices or driver
libraries were mounted. With `PYTHONPATH=/workspace/src:/packages`,
`TENSOR_NVRTC_HOME=/nvrtc` and `CUDA_HOME=/missing-toolkit`, the check was:

```bash
strace -f -e trace=openat -s 4096 -o /workspace/files.trace \
  python /workspace/producer.py --source-root /workspace \
  --out /workspace/output --require-isolated
```

The successful header reads must be confined to `/nvrtc/include/`,
`/packages/tilelang/src/` and `/packages/tilelang/3rdparty/cutlass/include/`.
Missing-header probes in strace are not successful dependency reads.
Retain the full trace and its hash with the report. The
[local report](../research/phase2-validation.md) records the observed audit.
`python scripts/validation/phase2_audit_trace.py PATH_TO_TRACE` checks successful reads,
including extensionless headers and files opened for compiler-input hashing;
repeat `--root` to specify different explicit include roots.

For consumer acceptance, place the five `.tbin` files, wheel and a
`producer.json` in one bundle directory. The provenance schema is the same
as [the Phase 1 transfer runbook](opaque-artifact-validation.md): base commit,
dirty state, physical hostname, lock hash, target, wheel filename/hash and
artifact filename/hashes. Keep container hostname separate from physical
hostname, so a local container is not counted as a second host. Run:

```bash
build/phase2-consumer/bin/python scripts/validation/phase1_transfer_check.py \
  build/phase2-transfer --allow-same-host --out build/phase2-consumer.json
```

This checks 21 numerical/interop cases and eight diagnostic cases with
compiler imports prohibited. If a retained CUDA v1 elementwise artifact
with declared outputs is available, add `--legacy-artifact PATH` for the
22nd compatibility case. For an actual two-host run, use clean matching
checkouts and omit `--allow-same-host`. The new
`.github/workflows/phase2-nvrtc.yml` builds on Linux/Windows and checks wheel
inspection in a fresh Tensor/NumPy environment; GPU execution is separate.

For Linux/Windows Actions archives, use `scripts/validation/phase2_transfer_check.py` instead
of manufacturing a Phase 1 provenance manifest. Save `gh run view RUN_ID --json
conclusion,event,headSha,jobs,status,url` and the run's Actions artifact API
records into one JSON document with an `artifacts` array. Download each archive
with `gh api repos/kreasof-ai/tensor/actions/artifacts/ARTIFACT_ID/zip > build/ci.zip`.
Install its included wheel in a fresh consumer environment, then run:

```bash
build/phase2-consumer/bin/python scripts/validation/phase2_transfer_check.py build/ci.zip \
  --ci-record build/ci.json --platform linux --out build/ci-gpu.json
```

Use `--platform windows` for the Windows producer. The check verifies archive
digests, exact producer source revision and LF/CRLF checkout hashes, distinct
physical hosts, five artifact hashes, 21 numerical/interop cases, eight
diagnostics, executable/workspace/event lifetime checks and CLI execution.
It reports whether the installed consumer package matches the included wheel;
a newer compatible consumer is allowed and identified explicitly.

## Runtime ABI and native hosts

The default tests exercise ABI 1.1 version/capability rejection, 64-bit scalar
payloads, shared CPU binding, failed loads, executable/event identity,
zero-workspace validation, native descriptor rejection and C++ CPU hosting:

```bash
uv run --locked python -m pytest -o addopts='' -q
uv run --locked tensor build examples/elementwise.py --provider cpu \
  --out build/phase2-cpu.tbin
uv run --locked python - <<'PY'
from pathlib import Path
from tensor.artifacts.format import read_artifact
_,files=read_artifact('build/phase2-cpu.tbin')
Path('build/phase2-kernel.so').write_bytes(files['kernel.so'])
_,files=read_artifact('build/phase2-transfer/elementwise.tbin')
Path('build/phase2-kernel.cubin').write_bytes(files['kernel.cubin'])
PY
c++ -std=c++17 -O2 -I src/tensor/include src/tensor/native/host.cpp \
  -ldl -o build/phase2-native-host
build/phase2-native-host cpu build/phase2-kernel.so
build/phase2-native-host cuda build/phase2-kernel.cubin elementwise_kernel
experiments/p0/out/rust-1.98.1/bin/rustc -O src/tensor/native/host.rs \
  -o build/phase2-rust-host
build/phase2-rust-host build/phase2-kernel.so
ldd build/phase2-native-host
ldd build/phase2-rust-host
```

The CPU producer needs a host C++ compiler; the NVRTC CUDA producer does not.
Install the retained pinned Rust compiler with `tools/bootstrap_rust.py` if
needed. Native hosts are Linux validation programs using extracted, verified
images; `ldd` should show only standard system runtimes, without Python,
TVM FFI, PyTorch or executable compiler libraries.

## Compiler comparison and GPU regressions

Use a complete CUDA 12.9 nvcc installation for the comparison path. This
requirement belongs to nvcc and the historical Phase 0 checks:

```bash
uv run --locked python tools/bootstrap_cuda.py --out build/cuda-12.9
export CUDA_HOME="$PWD/build/cuda-12.9"
TENSOR_P0_CUDA=1 TENSOR_P1_CUDA=1 TENSOR_P2_CUDA=1 \
  uv run --locked python -m pytest -o addopts='' -q
uv run --locked python scripts/validation/phase2_measure.py \
  --runtime-python build/phase2-consumer/bin/python \
  --nvcc "$CUDA_HOME/bin/nvcc" --out build/phase2-comparison
```

Measurement uses separate fresh compiler processes/caches, the same target,
and a clean consumer. It verifies identical snapshots and measures host
enqueue and launch plus stream synchronization. It does not measure pure
GPU kernel duration or reset system/compiler/driver caches. On a host without
the GPU/toolchains, opt-in tests skip; report skips explicitly.

Direct PTX experiments are outside this runbook and remain post-Tensor-v1 work.
