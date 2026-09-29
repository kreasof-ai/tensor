# Tensor

A product layer for high-performance tensor programs: one CLI, a small runtime ABI,
a capability-based provider model, and first-class compiled tensor modules.

The full architectural proposal lives in [`proposal.md`](proposal.md).

**Status: Phase 0 — Architecture Validation.** No product code is written yet, by design.
The question being answered right now is the one the proposal closes on:

> How much of `tensorc` already exists in TileLang and TIRx, and what minimal layer is
> actually missing between those systems and the developer experience we want?

**Two-host executable transfer now passes:** GitHub Actions builds five opaque
artifacts and an A10G (`sm_86`) runs them in fifteen fresh NumPy-only processes.
The remaining validation harness measures caches, CPU execution, a C++ host,
symbolic dimensions, PyTorch frontends, full compilation, and GPU baselines.
See [E15 and its raw data](docs/research/e15-phase0-validation.md).
The full GPU-enabled regression suite passes **40 tests with zero skips**.
Phase 0 remains open for fusion, new-provider registration, Rust hosting and
cross-GPU performance; measured failures are recorded explicitly.

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
  p0/                    the Phase 0 experiment harness (this is the active work)
src/                     product code — intentionally not created yet
```

`src/` does not exist yet on purpose (ADR 0001). Creating product packages before Phase 0
concludes would freeze structural decisions the experiments are meant to inform.

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

Run the remaining architecture probes in fresh processes with isolated caches:

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

## Where to start reading

| Document | What it gives you |
|---|---|
| [`docs/research/phase0-ground-truth.md`](docs/research/phase0-ground-truth.md) | Measured numbers, what already exists, what is blocked |
| [`docs/research/e14-cuda-execution.md`](docs/research/e14-cuda-execution.md) | NVIDIA correctness, compiled artifacts, and fresh-process startup |
| [`docs/research/e15-phase0-validation.md`](docs/research/e15-phase0-validation.md) | Remaining validation, two-host transfer, and measured restrictions |
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
