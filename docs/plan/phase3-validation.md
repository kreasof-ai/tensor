# Phase 3 validation runbook

Use the locked Python 3.12 producer and local NVRTC bundle from the
[Phase 2 runbook](phase2-validation.md). Module packaging and binary selection
do not require GPU access or compiler imports. Native execution uses the
existing provider requirements.

## Regression and local package acceptance

```bash
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
export CUDA_HOME="$PWD/experiments/p0/out/cuda-12.9"
TENSOR_P0_CUDA=1 TENSOR_P1_CUDA=1 TENSOR_P2_CUDA=1 TENSOR_P3_CUDA=1 \
  uv run --locked python -m pytest -o addopts='' -q
uv run --locked python tools/phase2_producer.py --out build/phase3-kernels --target sm_86
uv run --locked python tools/phase3_producer.py \
  --artifacts build/phase3-kernels --out build/phase3-transfer
uv build --wheel
uv venv --python 3.12 build/phase3-consumer
uv pip install --python build/phase3-consumer/bin/python \
  dist/tensor_workspace-0.1.0-py3-none-any.whl
```

Use fresh output directories. Module tests cover deterministic packaging,
relocated transitive closures, cycles/conflicts, frozen lock changes, failed
updates, corrupt snapshots, capability rejection, helpers, compiler-cache
recovery and CLI reference handling. CUDA opt-in tests compile source modules
through NVRTC and retarget sm_80 affine/GEMM TIRx to the actual device.

## Remote Linux/Windows acceptance

The `NVRTC runtime and module transfer` workflow uses the existing
`phase2-nvrtc.yml` matrix. It builds five profiles, packages `tensor-ops` and its
`tensor-base` dependency, creates the consumer wheel and installs/inspects
exports with exactly Tensor and NumPy under a compiler import guard.

Save a successful run's `conclusion,event,headSha,jobs,status,url` fields from
`gh run view --json`, plus its Actions API artifact records in an `artifacts`
array. Download both archives with `gh api .../actions/artifacts/ID/zip > FILE`.
Verify the archive digest before extracting its wheel; install that included
wheel into a separate clean consumer environment per producer platform.

```bash
build/phase3-consumer/bin/python tools/phase3_transfer_check.py build/linux.zip \
  --ci-record build/phase3-ci.json --platform linux --out build/linux-gpu.json
```

Repeat with `--platform windows` and the Windows-produced wheel/archive. The
checker verifies archive digests, producer revision and clean state, package
hashes and all five source hashes against exact LF/CRLF checkout bytes. It
checks the Phase 2 direct-artifact baseline, then installs a relocated module
closure, deletes the source package, repeats frozen installation, and executes
21 numerical/interop cases and eight diagnostics through installed exports.
Module CLI run/bench must work, all selections must be packaged binaries, and
the generated-image cache must remain empty. Compiler imports are prohibited.

Retain full GPU test output, raw CI records, per-platform acceptance and source
hashes. Record skips on GPU-free runners and the difference between Windows
production and Linux GPU execution. No cross-SM binary compatibility follows
from successful portable TIRx retargeting.

## PyPI transport acceptance

Run the registry tests with the optional publisher extra. They use only local
HTTP loopback endpoints, including an actual Twine multipart upload; no public
index is modified and no real publisher credentials are required.

```bash
uv run --locked --extra publish python -m pytest tests/test_registry.py -o addopts='' -q
tensor publish build/phase3-transfer/ops.tpack --dry-run --out-dir build/phase3-registry
build/phase3-consumer/bin/python tools/phase3_registry_check.py \
  build/phase3-registry/tensor_module_tensor_ops-0.1.0-py3-none-any.whl \
  --execute --out build/phase3-registry-gpu.json
```

Install the current runtime wheel into the clean consumer first. The checker
serves the prepared transport wheel through a local HTML Simple Index, adds it,
restores the frozen project into another empty cache, stops the server and
repeats frozen installation offline. It selects all five exact-target exports
under the compiler import guard. `--execute` additionally runs the existing
21 numerical/interop cases and eight diagnostics on the sm_86 A10G. CI omits
that flag on GPU-free Linux/Windows runners.
