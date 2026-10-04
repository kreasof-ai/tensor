# Build and run a kernel

[Documentation](../README.md) · [Python runtime](runtime.md)

This guide builds the included CUDA elementwise kernel, executes it, and prepares
a separate consumer without compiler dependencies. Python 3.12 is required.
Run commands from the repository root.

If you only want to execute a shared artifact, use the
[runtime installation guide](installation.md). See [compatibility](compatibility.md)
for hardware requirements and upgrade expectations.

Already have a TileLang kernel? Follow [From TileLang to Tensor](from-tilelang.md)
for the export hook, JIT/factory adaptation, and runtime argument mapping.

## 1. Prepare the producer

Install [uv](https://docs.astral.sh/uv/) and sync the pinned workspace:

```sh
uv sync --locked
uv run --locked python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
```

Set the bundle location on Linux:

```sh
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
```

Or in Windows PowerShell:

```powershell
$env:TENSOR_NVRTC_HOME = "$PWD/build/nvrtc-12.9"
```

The producer uses pinned TileLang 0.1.14 and TVM FFI 0.1.12, plus the local
NVRTC 12.9 libraries and headers. It needs no system CUDA toolkit or host C++
compiler for this CUDA build path. NVRTC is separate from the consumer wheel.
The dependency lock records the full compiler environment.

```sh
uv run --locked tensor doctor
uv run --locked tensor doctor --json
```

Doctor reports build readiness and run readiness separately. On a GPU-free
producer use `tensor doctor --target sm_86`. You can pass
`--nvrtc-home build/nvrtc-12.9` instead of setting the environment variable.
The optional full-toolkit path uses `--compiler nvcc` and `CUDA_HOME`/`CUDA_PATH`
or an explicit `--nvcc` path.

If you installed additional Torch, Triton, or wgpu dependencies into the
development environment, use `uv run --no-sync` to preserve them. See the
[development guide](../development.md).

## 2. Build an artifact

```sh
uv run --locked tensor build examples/elementwise.py --out build/elementwise.tbin
uv run --locked tensor inspect build/elementwise.tbin --stage manifest
```

Without `--target`, CUDA builds select device 0's exact SM. On a GPU-free host,
add `--target sm_86` or the architecture of your intended consumer. CUDA artifacts
contain one exact-SM cubin; a different GPU architecture requires another build.
Using NVRTC does not remove that compatibility requirement. Builds also retain
frontend TIRx for explicit recompilation in a compatible producer environment.

Use a new output path for each artifact. The content-addressed compiler cache
can still reuse a previous build. Inspect it with `tensor cache`; override it
with `--cache-dir` or `TENSOR_CACHE_DIR`.

The example implements `c = relu(2 * a + b)` for 129 FP32 elements. To write your
own kernel, export a TileLang `PrimFunc` and explicitly name its output buffers:

```python
def tensor_export():
    return {"kernel": elementwise, "outputs": ["c"]}
```

Tensor extracts launch dimensions and shared memory from the lowered kernel.
The CUDA profile supports contiguous buffers, positive static or symbolic
extents, and typed scalar arguments. See
[examples/elementwise.py](../../examples/elementwise.py) and
[examples/dynamic_affine.py](../../examples/dynamic_affine.py).

## 3. Execute from Python

Run the included consumer example:

```sh
uv run --locked python examples/run_elementwise.py build/elementwise.tbin
```

It prints `PASS: 129 elementwise results match NumPy` and the first eight values.
The script imports only Tensor and NumPy, so it also works with the separate
consumer in step 5. Its runtime calls are equivalent to the following:

Save this as a Python script, or run it in `uv run --locked python`:

```python
import numpy as np
import tensor as tx

with tx.Device() as device:
    kernel = device.load("build/elementwise.tbin")
    a = device.arange(129)
    b = device.ones((129,))
    c = kernel(a, b)
    expected = np.maximum(2 * a.to_numpy() + b.to_numpy(), 0)
    tx.assert_close(c, expected)
    print(c.to_numpy())
```

The call allocates declared outputs. `to_numpy()` downloads the result; the
device context owns allocations and loaded kernels. See the
[runtime guide](runtime.md) for output reuse, scalars, timing, and interoperability.

## 4. Run and benchmark from the CLI

Start `uv run --locked python` and prepare `.npy` inputs:

```python
from pathlib import Path
import numpy as np

Path("build/inputs").mkdir(parents=True, exist_ok=True)
for name in ("a", "b"):
    np.save(f"build/inputs/{name}.npy", np.arange(129, dtype="float32"))
```

Return to the shell to execute the artifact:

```sh
uv run --locked tensor run build/elementwise.tbin \
  --input a=build/inputs/a.npy --input b=build/inputs/b.npy --out-dir build/results
uv run --locked tensor bench build/elementwise.tbin \
  --input a=build/inputs/a.npy --input b=build/inputs/b.npy
```

`run` writes named outputs such as `build/results/c.npy`. `bench` reports host
enqueue and launch-plus-synchronization timing. For PowerShell, enter each
multi-line shell command on one line, or replace shell continuations with
PowerShell backticks.

For source inspection, including every lowering pass:

```sh
uv run --locked tensor inspect examples/elementwise.py \
  --stage passes --target sm_86 --out build/trace
```

`run` and `bench` also accept Python source, which compiles before execution.
Use a `.tbin` for the compiler-free consumer path. Named module exports resolve
packaged or cached binaries; local compilation there requires explicit
`--compile`. See the [module guide](modules.md).

## 5. Install a separate consumer

Build the core wheel, then install it into a fresh environment:

```sh
uv build --wheel --out-dir build/wheels
uv venv --python 3.12 build/consumer
uv pip install --python build/consumer/bin/python \
  build/wheels/tensor_workspace-0.1.0-py3-none-any.whl
build/consumer/bin/tensor run build/elementwise.tbin \
  --input a=build/inputs/a.npy --input b=build/inputs/b.npy --out-dir build/consumer-results
```

On Windows the executable paths are `build/consumer/Scripts/python.exe` and
`build/consumer/Scripts/tensor.exe`.

Check the installed version with the consumer Python's `-m tensor --version`.
You can run `examples/run_elementwise.py build/elementwise.tbin` with that same
Python to verify the result without compiler packages.

To run on another machine, transfer the wheel, `.tbin`, and input files. The
consumer needs Python 3.12, NumPy, a compatible NVIDIA driver, and a GPU matching
the artifact's SM. It does not need TileLang, TVM, Torch, NVRTC, CUDA headers,
or a host compiler. Tensor validates runtime ABI, capabilities, and target when
loading the artifact. Driver compatibility remains a requirement; consult
`tensor doctor` and the [runtime contract](../reference/runtime-abi.md).

## Other providers

- **WebGPU:** follow the [WebGPU guide](webgpu.md) to build WGSL on a GPU-free
  producer and install the optional wgpu consumer dependency. Adapter features
  and limits determine which artifacts can execute.
- **CPU validation:** on Linux x86-64, build with `--provider cpu` and a host
  C++ compiler. `run` and `bench` infer the provider from a built artifact.
  In Python use `tx.Device(provider="cpu")`.
