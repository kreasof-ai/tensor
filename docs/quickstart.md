# Build and run a kernel

Use Python 3.12 and [uv](https://docs.astral.sh/uv/). Tensor's distribution is
`tensor-workspace`; its import and command are `tensor`. Read
[compatibility](compatibility.md) for the supported scope.

For the standalone CLI, use the [one-line installer](installation.md).
The checkout below provides kernel examples and the development environment.

## Choose the environment

| Environment | Install | Purpose |
|---|---|---|
| Runtime | `uv sync --locked --no-default-groups` | CLI and compiler-free execution |
| Development/producer | `uv sync --locked` | Build kernels and run tests/examples |
| WebGPU runtime | Add `--extra webgpu` | Native wgpu execution |

The runtime depends on NumPy. The producer also installs the pinned TileLang/TVM
compiler stack and development tools. Execution needs the corresponding driver
and matching hardware. Exploring `tensor --help` needs no GPU.

## Prepare a CUDA producer

```sh
uv sync --locked
uv run --locked python tools/bootstrap/bootstrap_nvrtc.py --out build/nvrtc-12.9
```

Set the bundle location in your shell:

```sh
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
```

```powershell
$env:TENSOR_NVRTC_HOME = "$PWD/build/nvrtc-12.9"
```

```sh
uv run --locked tensor doctor
uv run --locked tensor build examples/elementwise.py --out build/elementwise.tbin
uv run --locked tensor inspect build/elementwise.tbin
uv run --locked python examples/run_elementwise.py build/elementwise.tbin
```

The runner validates `relu(2 * a + b)`. Output artifacts must use fresh paths.
A producer without a GPU can pass `--target sm_86`; use the exact architecture
of the consuming device. Target selection does not establish compatibility with
another GPU architecture.

## Execute from Python

```python
import tensor as tx

with tx.Device() as device:
    kernel = device.load("build/elementwise.tbin")
    a = device.randn((129,), "float32", seed=1)
    b = device.randn((129,), "float32", seed=2)
    result = kernel(a, b)
    tx.assert_close(result, (2 * a.to_numpy() + b.to_numpy()).clip(0))
```

Match the buffer shape to the kernel's declared `SIZE`; the included example is
129 elements. `kernel(...)` allocates declared outputs. Use `kernel.launch(...)`
with all buffers for repeated calls without output allocation.

## Run and benchmark from the CLI

Prepare NumPy input files matching the artifact, then run:

```sh
uv run --locked tensor run build/elementwise.tbin --input a=a.npy --input b=b.npy --out-dir build/result
uv run --locked tensor bench build/elementwise.tbin --input a=a.npy --input b=b.npy
```

Inspect the artifact to obtain its argument names and shapes. Symbolic dimensions
and typed scalar inputs are described in [Runtime](runtime.md).

## Use a compiler-free consumer

Build a wheel and install it in a separate environment:

```sh
uv build --wheel --out-dir build/wheels
uv venv --python 3.12 build/consumer
uv pip install --python build/consumer/bin/python build/wheels/tensor_workspace-0.1.0-py3-none-any.whl
build/consumer/bin/python examples/run_elementwise.py build/elementwise.tbin
```

On Windows, use `build/consumer/Scripts/python.exe`. A different machine needs
the wheel, artifact, NumPy, and matching hardware/driver. A standalone wheel can
also be installed with `python -m pip install PATH_TO_WHEEL`.

Use [Compilation](compilation.md) for your own TileLang exports,
[Modules](modules.md) for distributing kernel collections, and
[Backends](backends.md) for WebGPU. `tensor doctor --json` reports diagnostics;
set an explicit target when the producer has no GPU.
