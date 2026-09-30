# Tensor

A product layer for high-performance tensor programs: one CLI, a small runtime ABI,
a capability-based provider model, and first-class compiled tensor modules.

The full architectural proposal lives in [`proposal.md`](proposal.md).

**Status: Phases 0–3 complete within their measured profiles.**
CUDA builds default to a pinned NVRTC bundle, with no installed CUDA toolkit
or host compiler required. CUDA and a CPU validation provider share call ABI
1.1, including executable/event identities and zero external workspace,
independently versioned from the v3 artifact envelope. See the
[Phase 2 exit report](docs/research/phase2-exit.md) and
[runtime contract](docs/runtime-abi.md). Direct PTX is experimental work after
Tensor v1. The CLI has `doctor`, `build`, `inspect`, `run`, `bench`, and cache
inspection; CUDA supports symbolic shapes, scalar arguments and GPU DLPack.
Phase 0 answered the question the proposal closes on:

> How much of `tensorc` already exists in TileLang and TIRx, and what minimal layer is
> actually missing between those systems and the developer experience we want?

Phase 3's offline module system is complete.
`tensor.json` names exports, `tensor.lock` pins dependency content, and `tensor
pack` bundles the complete closure. `add`, `install`, `resolve`, and
`module-name::export_name` references work with compiler-free packaged binaries;
missing targets compile through bundled TIRx/source only when explicitly
requested. The GPU suite passes **104 tests with zero skips**; Linux and
Windows module packages each pass execution on the separate A10G consumer.
See the [module guide](docs/modules.md) and [Phase 3 exit report](docs/research/phase3-exit.md).

**Phase 0 two-host executable transfer passed:** GitHub Actions built five opaque
artifacts and an A10G (`sm_86`) runs them in fifteen fresh NumPy-only processes.
The validation harness measures caches, independent CPU provider execution,
C++ and Rust hosts, bounded composition, symbolic dimensions, PyTorch
frontends, full compilation, GPU baselines and foreign CUDA stream ordering.
See [E15](docs/research/e15-phase0-validation.md) and the
[Phase 0 exit report](docs/research/e16-phase0-exit.md).
The Phase 0 GPU-enabled regression suite passed **40 tests with zero skips**.
Its scoped exit gates passed. General fusion and cross-GPU benchmarking remain
future work; Phase 0 performance evidence uses the available A10G.

---

## What we know so far

The initial measurements below used the original compilation probes, before
the workload corrections. They are historical observations, not numerics or
performance results for the current kernels. Measured on a Windows box with an
**RX 6700 XT and no CUDA** (full data in
[`docs/research/`](docs/research/)):

- **All 5 workload kernels compile to real CUDA C++ with no GPU and no CUDA toolchain**
  (15/15, 0.07–1.41 s each) — via `compile_kernel_source`, which needs no device. The
  architecture decisions do not wait on hardware.
- **~63 distinct lowering passes** for one GEMM, with TIRx IR dumped at every stage
  (2 MB per kernel). `tensor inspect` is a wrapper over an existing capability.
- **Packaging risk is `torch`, not TileLang.** ~230 MB downloaded, 695 MB installed, of
  which torch is 71%. The tile compiler's own wheel is 27 MB.
- **A torch-free compiler is a candidate, not yet demonstrated.** `tvm_compiler.dll` and `tvm_runtime.dll` have *no*
  torch in their PE import tables — the dependency is entirely Python-layer, and traces to
  one `import torch  # preload torch to avoid dlopen errors` line. ADR 0005 proposes a
  client-side adapter. The current TileLang producer still imports PyTorch.
- **`import tilelang` costs 4.2 s** warm and **23 s on first run**. That is a hard
  constraint on `tensor run`, not a future problem.
- **TileLang already registers six backends** (`rocm, cuda, cutedsl, cpu, metal, webgpu`)
  through a documented manifest, and already offers a CuTe DSL target.
- **RDNA2 (gfx1031) is not a viable local target** — ROCm is Linux-only and unsupported for
  that arch. Not on the roadmap; see ADR 0002.

---

## Repository layout

```text
proposal.md              the architectural proposal
docs/
  research/              findings, with measured numbers and sources
  plan/                  experiment designs and the roadmap
  adr/                   architecture decision records
experiments/
  p0/                    the retained Phase 0 experiment harness
src/tensor/              product CLI, compiler adapters and runtime providers
```

The product package is separate from the experiment code. [ADR 0009](docs/adr/0009-complete-phase0-with-scoped-provider-and-composition.md)
opened Phase 1 product work after the exit gates passed.

## Product CLI

Install the pinned compiler environment and NVRTC libraries/headers:

```bash
uv sync --locked
uv run --locked python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
uv run --locked tensor doctor
uv run --locked tensor doctor --json
```

Doctor detects the NVIDIA device and uses its exact SM target. It checks the
pinned TileLang and TVM FFI versions, the registered CUDA backend, the driver,
and a real NVRTC cubin compilation with explicitly bundled headers.
On a GPU-free build host, supply an explicit target such as `--target sm_86`.
Use `--nvrtc-home` instead of the environment variable if preferred. NVRTC
is a separate local compiler bundle; it is not part of the consumer wheel.
To use a full CUDA toolkit instead, select `--compiler nvcc` with
`CUDA_HOME`/`CUDA_PATH`, or supply `--nvcc /path/to/nvcc`. Exit code 0 means the host can
build for the target, run artifacts on the detected device, or both; the
summary distinguishes `build ready`, `run ready`, and `ready`. Code 1 means
neither path is ready.

Build a standalone TileLang kernel:

```bash
uv run --locked tensor build examples/elementwise.py --out build/elementwise.tbin
```

The source must define `tensor_export()` returning `{"kernel": PrimFunc,
"outputs": ["result_name"]}`. Tensor reads grid, block and dynamic shared
memory from the lowered kernel. An optional explicit `launch` description is
checked against those values. Declare outputs to use `tensor run` and the
Python call API.
The CUDA profile accepts one kernel with contiguous buffers, static or symbolic
shapes, and typed scalar arguments. The included elementwise, float16 GEMM and
int64 scalar examples execute on the A10G.
`--target sm_XX` permits a GPU-free build host; without it, build targets device
0. Outputs are created exclusively. The `.tbin` contains a cubin, serialized
frontend TIRx, exact compiler versions, source and payload hashes, and notices.
New artifacts use `tensor.module` envelope version 3, which declares the runtime
provider, call ABI, capabilities and compiler provenance, as well as lowered
argument order, launch expressions and pointer alignment. The consumer also
reads existing CUDA envelope versions 1 and 2. CUDA images still require the
exact SM used at build time; NVRTC does not add cross-GPU portability.

On Linux x86-64, `tensor build ... --provider cpu` creates a native image with
the host C++ compiler. `tensor run` and `bench` infer its provider from the
artifact. This CPU implementation validates the shared runtime contract;
it supports contiguous buffers, typed scalars and direct symbolic extents.

The build cache is keyed by source, frontend IR, exact compiler versions and
target, executable compiler identity, and bundled compiler headers. `tensor build`
reports `cache_hit`, and `tensor cache` reports the entry count and bytes.
Override the cache with `--cache-dir` or `TENSOR_CACHE_DIR`.

Run and time an artifact with NumPy `.npy` inputs:

```bash
uv run --locked python - <<'PY'
import numpy as np
from pathlib import Path
Path("build/inputs").mkdir(parents=True, exist_ok=True)
for name in ("a", "b"):
    np.save(f"build/inputs/{name}.npy", np.arange(129, dtype="float32"))
PY
uv run --locked tensor run build/elementwise.tbin \
  --input a=build/inputs/a.npy --input b=build/inputs/b.npy \
  --out-dir build/results
uv run --locked tensor bench build/elementwise.tbin \
  --input a=build/inputs/a.npy --input b=build/inputs/b.npy
uv run --locked tensor inspect build/elementwise.tbin --stage manifest
uv run --locked tensor inspect examples/elementwise.py --stage passes --target sm_86 \
  --out build/trace
```

For a symbolic length and a runtime scalar, build `examples/dynamic_affine.py`
and supply `--scalar scale=2.5` to `run` or `bench`. Tensor infers `size` from
the inputs; the same binary serves lengths 1, 127, 128, 129 and 1025. The Python
equivalent is `kernel(a, b, scale=2.5)`. You can supply a dimension explicitly
with `size=129` or `--scalar size=129`; it must agree with all input shapes.
Scalar arguments support bool, signed/unsigned 8–64-bit integers and
float32/float64, with overflow and finite-value checks.

`run` and `bench` also accept a `.py` source and compile it first. A built
artifact runs with only the Tensor wheel, NumPy, and an NVIDIA driver; compiler
packages and CUDA development headers are absent from the consumer path.
The product artifact and wheel have also transferred from a GPU-free GitHub
Actions producer to the A10G consumer with matching hashes and exact results.
Linux and Windows Actions producers each build all five NVRTC profiles; their
artifacts and included wheels pass GPU acceptance on the separate Linux A10G
consumer. The full GPU suite passes 89 tests with zero skips. GPU execution on
Windows itself has not been tested.
For a kernel-author Python session:

```python
import tensor as tx
with tx.Device() as device:
    kernel = device.load("build/elementwise.tbin")
    a = device.arange(129)
    b = device.ones((129,))
    c = kernel(a, b)
    tx.assert_close(c, 2 * a.to_numpy() + b.to_numpy())
    print(tx.bench(kernel, (a, b, c)))
```

The workbench exposes owned and borrowed device buffers, shape/dtype/strides,
byte snapshots, NumPy upload/download, CPU DLPack upload, GPU DLPack borrowing,
`zeros`, `ones`, `full`, `randn`, `arange`, numerical checks, and timing.
`bench` reports host enqueue and launch-plus-stream-synchronization times.
Buffer address, shape, dtype and strides are immutable. Loaded kernels expose
`descriptor` and `workspace_requirements()`; `release()` waits for submitted
work and unloads the image. Sessions resolve descriptor snapshots with
`get_executable()` and `get_event()`, rejecting stale or foreign identities.

`device.from_dlpack(gpu_tensor)` borrows a contiguous writable tensor without a
copy or framework import. It retains the producer's managed tensor until
release. The session retains the CUDA primary context so framework allocations
and streams are accessible. Run the import in the producer's stream context;
DLPack arranges the dependency onto Tensor's stream. For later producer updates,
use `device.wait_for(producer_stream_handle)`. After launching into a borrowed
output, call `device.handoff(consumer_stream_handle)` before consuming it there.
`Device(stream=foreign_handle)` launches directly on a borrowed stream and
preserves its ownership. Foreign streams must outlive the session. Session
cleanup waits for its work and handed-off consumer streams, releases imported
managed tensors exactly once, and restores the previous context.

The current CUDA profile requires positive extents, an exact SM match,
contiguous storage, and the artifact's pointer alignment. GPU execution is
measured on A10G. [The Phase 1 exit report](docs/research/phase1-exit.md)
records acceptance checks, installation counts, latency, diagnostics and transfer evidence.

## Development environment

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned rather than 3.13/3.14
because tilelang declares `torch-c-dlpack-ext; python_version < "3.14"` — on 3.14 the
DLPack tensor bridge is silently dropped.

```powershell
.\tools\bootstrap.ps1                                  # or: uv sync
uv sync --locked
uv run --locked python -m experiments.p0.harness --list
uv run --locked python -m experiments.p0.harness                # all experiments
uv run --locked python -m experiments.p0.harness --only codegen # one experiment
uv run --locked python -m pytest
```

Results land in `experiments/p0/out/`: `results.json`, generated `.cu` sources, and
`out/_traces/<kernel>/` containing the TIRx IR at every lowering stage.
Reports now record Git revision, dirty state, source hash, lock hash and installed
package versions. Python 3.12, TileLang 0.1.14 and TVM FFI 0.1.12 are pinned;
`uv.lock` records the complete resolution and bootstrap uses it.

On a CUDA build host, check the **full compiler installation** before building:

```bash
uv run --locked python -m experiments.p0.artifact_build doctor --arch sm_86
```

This compiles a small CUDA probe without a GPU or TileLang imports. It checks
runtime/CCCL headers and the host compiler as well as `nvcc`. Set `CUDA_HOME` or
`CUDA_PATH` to the toolkit root, or pass `--nvcc /path/to/nvcc`. An explicit
compiler wins, then the configured toolkit, then `nvcc` on `PATH`. The separate
`artifact_run doctor` checks the NVIDIA device/driver. The
[validation runbook](docs/plan/opaque-artifact-validation.md) covers GPU tests
and creating a NumPy-only consumer environment.

Run the retained Phase 0 validation suite in fresh processes with isolated caches:

```bash
uv run --locked python -m experiments.p0.validation --out experiments/p0/out/new-validation-run
```

Use a new directory for each run. On Linux, the pinned CUDA build components
can be installed locally:

```bash
uv run --locked python tools/bootstrap_cuda.py --out experiments/p0/out/cuda-12.9
export CUDA_HOME="$PWD/experiments/p0/out/cuda-12.9"
```

The native-host probe requires g++. The manual GitHub Actions workflow
`P0 artifact transfer producer` builds the five executable boundary cases.
The final exit suite additionally installs a pinned local Rust compiler and
runs the scoped completion gates:

```bash
uv run --locked python tools/bootstrap_rust.py --out experiments/p0/out/rust-1.98.1
export TENSOR_P0_RUSTC="$PWD/experiments/p0/out/rust-1.98.1/bin/rustc"
uv run --locked python -m experiments.p0.phase0_exit --out experiments/p0/out/new-phase0-exit
```

## Where to start reading

| Document | What it gives you |
|---|---|
| [`docs/research/phase0-ground-truth.md`](docs/research/phase0-ground-truth.md) | Measured numbers, what already exists, what is blocked |
| [`docs/research/e14-cuda-execution.md`](docs/research/e14-cuda-execution.md) | NVIDIA correctness, compiled artifacts, and fresh-process startup |
| [`docs/research/e15-phase0-validation.md`](docs/research/e15-phase0-validation.md) | Remaining validation, two-host transfer, and measured restrictions |
| [`docs/research/e16-phase0-exit.md`](docs/research/e16-phase0-exit.md) | Final provider, composition, Rust, stream and static/symbolic gates |
| [`docs/research/ecosystem-and-precedents.md`](docs/research/ecosystem-and-precedents.md) | Triton / CuTe DSL / tinygrad / Pallas comparison, packaging precedents |
| [`docs/plan/phase0-experiment-design.md`](docs/plan/phase0-experiment-design.md) | The 12 experiments and which machine runs each |
| [`docs/plan/roadmap.md`](docs/plan/roadmap.md) | Phases, reordered by what the evidence supports |
| [`docs/adr/`](docs/adr/) | Architecture decisions, each with its evidence |

## Ground rules

- Experiments record measurements, not impressions. Every claim in `docs/research/`
  carries a number, a command, or a source URL.
- "Unverified" is a valid, expected result. See the experiment design.
- Re-verify ground truth against installed packages at each phase boundary. The proposal
  already contained one stale claim, and this stack's knowledge has a shelf life measured
  in weeks.
