# Tensor

Build tensor kernels once. Run them with a small runtime.

Tensor compiles TileLang/TIRx kernels into `.tbin` artifacts that can run on a
separate machine without the compiler stack. A Python API and one CLI cover
building, inspection, execution, and reusable kernel modules.

**Pre-1.0 software:** the current package version is `0.1.0`. APIs and artifact
compatibility can change; pin versions and rebuild artifacts when upgrading.
See [support and compatibility](docs/guides/compatibility.md) for the tested scope.

## Start here

| Your goal | Next step |
|---|---|
| Explore the CLI without a GPU or compiler | Install the runtime below |
| Build and execute your first kernel | [Quickstart](docs/guides/quickstart.md) |
| Package a TileLang kernel you already wrote | [From TileLang to Tensor](docs/guides/from-tilelang.md) |
| Search schedule choices for a kernel | [Schedule discovery](docs/guides/schedule-search.md) |
| Run an artifact someone shared with you | [Runtime installation](docs/guides/installation.md) |
| Use AMD/Intel/Apple through native wgpu | [WebGPU guide](docs/guides/webgpu.md) |
| Work on Tensor | [Contributing](CONTRIBUTING.md) |

## Install the runtime

Use 64-bit Python **3.12** and [uv](https://docs.astral.sh/uv/). Start from a
checkout; these commands work in PowerShell and a POSIX shell:

```sh
git clone https://github.com/kreasof-ai/tensor.git
cd tensor
uv sync --locked --no-default-groups
uv run --locked --no-default-groups tensor --version
uv run --locked --no-default-groups tensor --help
```

This installs the core and NumPy. Exploring the CLI requires no GPU. Executing
an artifact requires the corresponding driver and hardware. The distribution is
named `tensor-workspace`; the import and CLI are named `tensor`.

To build your own kernels, install the pinned development/compiler environment:

```sh
uv sync --locked
uv run --locked python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
```

Set the NVRTC bundle location in your shell:

```sh
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
```

```powershell
$env:TENSOR_NVRTC_HOME = "$PWD/build/nvrtc-12.9"
```

With an NVIDIA GPU, build and check the included example:

```sh
uv run --locked tensor doctor
uv run --locked tensor build examples/elementwise.py --out build/elementwise.tbin
uv run --locked python examples/run_elementwise.py build/elementwise.tbin
```

The runner compares `relu(2 * a + b)` against NumPy and prints a success message.
Use a new artifact output path each time. A GPU-free producer can build with an
explicit `--target sm_86`; execution requires a GPU with that exact architecture.
The [quickstart](docs/guides/quickstart.md) explains each step and the separate
compiler-free consumer path.

## What you can do

```text
TileLang kernel → TIRx lowering → provider compiler → .tbin artifact
                                                       ↓
                                             Tensor runtime → GPU
```

| Component | Scope | Guide |
|---|---|---|
| Runtime and CLI | CUDA kernels, contiguous buffers, symbolic dimensions, typed scalars, DLPack, streams, inspection, caching | [Runtime](docs/guides/runtime.md) |
| Kernel modules | Named exports, pinned dependencies, `.tpack` archives, offline distribution and PyPI transport wheels | [Modules](docs/guides/modules.md) |
| `tensor-torch` | Optional `torch.compile` inference adapter and custom operators | [PyTorch](docs/guides/pytorch.md) |
| WebGPU | Optional native wgpu provider for bounded elementwise, linear/MLP and forward-attention profiles | [WebGPU](docs/guides/webgpu.md) |
| `tensor-nn` | Optional manual nanoGPT training templates | [Tensor NN](packages/tensor-nn/README.md) |
| `tensor-llm` | Optional single-sequence LFM2 GGUF inference | [Tensor LLM](packages/tensor-llm/README.md) |

CUDA is the primary backend. CUDA compilation uses a local NVRTC bundle;
consumers need only the core wheel, NumPy, an NVIDIA driver, and matching hardware.
WebGPU artifacts carry WGSL and explicit adapter feature/limit requirements.
A Linux x86-64 CPU provider exercises the shared runtime contract.

Standalone autograd, arbitrary graph fusion, complete TileLang/TIRx coverage on
WebGPU, and general GGUF model support are future work. Performance claims apply
to the specific workloads and hardware in the [benchmark reports](docs/research/README.md).

## Documentation and community

- [Documentation](docs/README.md): installation, guides, and API contracts.
- [Contributing](CONTRIBUTING.md): setup, tests, and how to propose a change.
- [Changelog](CHANGELOG.md) and [release guide](docs/releases.md).
- [Path to 1.0](docs/plan/v1.md): planned stability contracts and release milestones.
- [GitHub issues](https://github.com/kreasof-ai/tensor/issues): bugs and ideas.
- [Project history](docs/research/README.md): experiments, measurements, and provenance.

Tensor is licensed under [MIT](LICENSE). Compiler dependencies and model weights
retain their own licenses.
