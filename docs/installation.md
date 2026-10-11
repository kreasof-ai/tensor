# Install Tensor

The standalone installer gives you the `tensor` command. It manages Python 3.12
and installs an isolated runtime containing Tensor and NumPy under `~/.tensor`
(`%USERPROFILE%\.tensor` on Windows). It needs no existing Python, Git, or admin
access. Python projects install Tensor as a dependency in their own environment;
the standalone CLI environment is separate.

## One-line CLI install

Linux and macOS, using Bash:

```sh
curl -fsSL https://raw.githubusercontent.com/kreasof-ai/tensor/main/tools/install/install.sh | bash
```

Windows, using PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -c "irm https://raw.githubusercontent.com/kreasof-ai/tensor/main/tools/install/install.ps1 | iex"
```

Open a new terminal, then run:

```sh
tensor --version
tensor --help
```

The installer uses [uv](https://docs.astral.sh/uv/) to manage its environment. It
reuses an available uv executable or installs uv 0.12.9 into Tensor's `bin/`.
It adds that directory to your shell/user PATH. CUDA execution still needs an
NVIDIA driver; WebGPU execution needs a supported native adapter. Installation
and CLI help do not need a GPU.

There are no published release wheels yet. By default the installer builds the
core package from the GitHub `main` source archive. This follows development;
set `TENSOR_REF` to a commit or tag for a fixed source revision. Re-running the
installer refreshes Tensor from the selected source.

## Options

Both installers accept the same environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `TENSOR_INSTALL_DIR` | `~/.tensor` | Installation root, including `bin/`, `tools/`, and managed Python |
| `TENSOR_EXTRAS` | Empty | `compiler`, `webgpu`, or `compiler,webgpu` |
| `TENSOR_REF` | `main` | GitHub commit, tag, or branch |
| `TENSOR_NO_MODIFY_PATH` | `0` | Set to `1` to leave shell/user PATH unchanged |
| `TENSOR_SOURCE` | Archive for `TENSOR_REF` | Override with a package archive or wheel URL for testing/distribution |

For a compiler and WebGPU install:

```sh
curl -fsSL https://raw.githubusercontent.com/kreasof-ai/tensor/main/tools/install/install.sh \
  | TENSOR_EXTRAS=compiler,webgpu bash
```

```powershell
$env:TENSOR_EXTRAS = 'compiler,webgpu'
irm https://raw.githubusercontent.com/kreasof-ai/tensor/main/tools/install/install.ps1 | iex
```

The compiler option installs TileLang and its dependencies. CUDA compilation also
needs [NVRTC or nvcc setup](quickstart.md#prepare-a-cuda-producer). The WebGPU option
installs the pinned native wgpu Python dependency. See [backends](backends.md) for
provider requirements. Set extras again when updating to keep those options.

## Install into a Python project

Use your project's Python 3.12 environment. For example, from a Tensor checkout:

```sh
uv sync --locked --no-default-groups
uv run --locked --no-default-groups tensor --help
```

A development/producer checkout uses `uv sync --locked`; the
[quickstart](quickstart.md) includes examples and compiler setup.

## Remove the CLI

For the default installation, on Linux/macOS:

```sh
UV_TOOL_DIR="$HOME/.tensor/tools" UV_TOOL_BIN_DIR="$HOME/.tensor/bin" \
  uv tool uninstall tensor-workspace
```

On Windows:

```powershell
$env:UV_TOOL_DIR = "$env:USERPROFILE\.tensor\tools"
$env:UV_TOOL_BIN_DIR = "$env:USERPROFILE\.tensor\bin"
uv tool uninstall tensor-workspace
```

This removes Tensor's tool environment and command. The managed Python and uv
remain in the installation root; deleting that root removes them too. Adjust
paths if you used `TENSOR_INSTALL_DIR`.
