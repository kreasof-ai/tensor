# Tensor

Build TileLang kernels into portable artifacts. Execute them with a small runtime.

```text
TileLang kernel → Tensor compiler → .tbin → Tensor runtime → device
```

Compilation and execution are separate: consumers need the runtime and their
hardware driver, without TileLang, TVM, or PyTorch. CUDA, WebGPU, and a scoped CPU
provider share buffer and executable contracts. Tensor is pre-1.0 (`0.1.0`);
see [compatibility](docs/compatibility.md) before transferring artifacts.

## Install the CLI

Linux/macOS:

```sh
curl -fsSL https://raw.githubusercontent.com/kreasof-ai/tensor/main/tools/install/install.sh | bash
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/kreasof-ai/tensor/main/tools/install/install.ps1 | iex"
```

Open a new terminal and run `tensor --help`. The installer manages Python 3.12
and installs the runtime under `~/.tensor`, without a system Python or Git.
[Installation](docs/installation.md) covers optional compiler/WebGPU dependencies,
custom paths, updates, removal, and installing Tensor into a Python project.

## First kernel

Use 64-bit Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/kreasof-ai/tensor.git
cd tensor
uv sync --locked
uv run --locked python tools/bootstrap/bootstrap_nvrtc.py --out build/nvrtc-12.9
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
uv run --locked tensor build examples/elementwise.py --out build/elementwise.tbin
uv run --locked python examples/run_elementwise.py build/elementwise.tbin
```

The example checks `relu(2 * a + b)` against NumPy. The
[quickstart](docs/quickstart.md) includes PowerShell, explicit GPU targets, and a
separate compiler-free consumer. To explore the CLI without a compiler or GPU:

```sh
uv sync --locked --no-default-groups
uv run --locked --no-default-groups tensor --help
```

## Read and extend

| Task | Start here |
|---|---|
| Build and execute a kernel | [Quickstart](docs/quickstart.md) |
| Understand the implementation | [Architecture](docs/architecture.md) |
| Import TileLang kernels or search schedules | [Compilation](docs/compilation.md) |
| Allocate buffers, launch kernels, use streams or manual backward | [Runtime](docs/runtime.md) |
| Package named exports and distribute artifacts | [Modules](docs/modules.md) |
| Use WebGPU | [Backends](docs/backends.md) |
| Bind kernels to PyTorch | [Tensor Torch](packages/tensor-torch/README.md) |
| Compose native NN modules and train nanoGPT | [Tensor NN](packages/tensor-nn/README.md) |
| Develop and validate Tensor | [Development](docs/development.md) |

[AKAS](https://github.com/kreasof-ai/akas) owns model inference, generation,
scheduling, and serving. [Loop Latent Transformer](https://github.com/kreasof-ai/loop-latent-transformer)
owns LLT workloads. Tensor supplies their compilation and execution primitives.

The distribution is named `tensor-workspace`; Python imports and the CLI use
`tensor`. Optional bindings/templates have separate distributions. See the
[documentation index](docs/README.md), [changelog](CHANGELOG.md), and
[contribution guide](CONTRIBUTING.md). Historical measurements and design decisions
are retained under [research](research/README.md), separate from current usage.

MIT licensed. Compiler dependencies retain their own licenses.
