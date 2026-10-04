# Install Tensor

[Documentation](../README.md) · [Quickstart](quickstart.md) · [Compatibility](compatibility.md)

Tensor currently ships as version `0.1.0`, with pre-1.0 compatibility expectations.
Use **64-bit Python 3.12**. The distribution name is `tensor-workspace`; import
it as `tensor` and invoke the CLI as `tensor` or `python -m tensor`.
These instructions use a checkout or wheels built from it; they do not assume a
public PyPI release exists.

## Choose an environment

| Environment | Dependencies | Hardware needed |
|---|---|---|
| Core consumer / CLI exploration | Tensor and NumPy | None for import/help/inspection; compatible GPU/driver to execute CUDA |
| CUDA producer | Pinned compiler packages and local NVRTC bundle | None with an explicit target; NVIDIA GPU for local execution |
| WebGPU consumer | Core plus the `webgpu` extra | Native wgpu-compatible adapter and required artifact features |
| Contributor | Locked compiler and development groups | None for ordinary tests; provider tests need their own setup |

Optional `tensor-torch`, `tensor-nn`, and `tensor-llm` distributions add only the
features you choose. Start with the core before attempting model benchmarks.

## Try the runtime from a checkout

Install [uv](https://docs.astral.sh/uv/), clone the repository, and run:

```sh
git clone https://github.com/akbar2habibullah/tensor.git
cd tensor
uv sync --locked --no-default-groups
uv run --locked --no-default-groups tensor --version
uv run --locked --no-default-groups tensor --help
```

The expected version output is `Tensor 0.1.0`. Use `--no-default-groups` on
subsequent `uv run` commands in this environment; omitting it restores the default
compiler/development groups. To switch to the full producer, run `uv sync --locked`
and follow the [quickstart](quickstart.md).

## Install a standalone wheel

Build the core wheel from a checkout:

```sh
uv build --wheel --out-dir build/wheels
uv venv --python 3.12 build/consumer
```

On Linux:

```sh
uv pip install --python build/consumer/bin/python build/wheels/tensor_workspace-0.1.0-py3-none-any.whl
build/consumer/bin/python -m tensor --version
build/consumer/bin/python -m tensor --help
```

On Windows PowerShell:

```powershell
uv pip install --python build/consumer/Scripts/python.exe build/wheels/tensor_workspace-0.1.0-py3-none-any.whl
build/consumer/Scripts/python.exe -m tensor --version
build/consumer/Scripts/python.exe -m tensor --help
```

You can transfer the wheel and artifact to another machine and use
`python -m pip install /path/to/tensor_workspace-0.1.0-py3-none-any.whl` in a
Python 3.12 virtual environment. This installs NumPy from the configured package
index. For disconnected installation, also prepare and transfer a NumPy wheel
for the consuming OS/Python, then install with `--no-index --find-links`.

The installed core imports without TileLang, TVM, Torch, or wgpu. Compilation
requires the producer environment; a `.tbin` executes without it. Run
`python -m tensor doctor --json` to check local CUDA readiness. `run_ready` is a
successful consumer setup even when compiler checks report missing packages.

## Execute a shared artifact

The sender must supply an artifact for your exact CUDA SM and matching Tensor
version. An artifact built for `sm_86` cannot run on another SM. From the
checkout, use the [elementwise runner](../../examples/run_elementwise.py) with
the consumer Python:

```sh
build/consumer/bin/python examples/run_elementwise.py build/elementwise.tbin
```

On Windows substitute `build/consumer/Scripts/python.exe`. General artifacts
use `tensor run` with named `.npy` inputs, or the [Python runtime](runtime.md).

## Troubleshooting

| Symptom | Next step |
|---|---|
| Wrong Python version | Create a Python 3.12 environment with `uv venv --python 3.12` |
| `tensor` command unavailable | Invoke `python -m tensor` with the environment's Python |
| Compiler packages missing in a consumer | Expected for execution; use a producer to build source |
| NVRTC bundle missing on a producer | Run the quickstart bootstrap and set `TENSOR_NVRTC_HOME` in the same shell |
| No CUDA device | Check driver/device access; GPU-free builds require an explicit `--target` |
| Artifact target or ABI rejected | Rebuild for the consuming GPU/version; see [compatibility](compatibility.md) |
| Artifact output already exists | Choose a new path; build outputs intentionally do not overwrite |
| WebGPU feature/limit error | Inspect `tensor doctor --provider webgpu --json`; use a compatible adapter/artifact |

For unresolved issues, follow the reproduction guidance in
[Contributing](../../CONTRIBUTING.md).
