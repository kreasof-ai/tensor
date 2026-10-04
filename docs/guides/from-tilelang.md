# Use an existing TileLang kernel with Tensor

[Documentation](../README.md) · [Installation](installation.md) · [Python runtime](runtime.md)

If you already have a TileLang kernel, Tensor can compile its `PrimFunc` into a
`.tbin` and execute that artifact through its own runtime. Start by exposing the
uncompiled kernel and naming its output buffers. Your consumer then loads the
artifact instead of importing the kernel's Python source or invoking TileLang JIT.

This guide uses CUDA and an addition kernel as a concrete example. Replace the
factory, specialization values, inputs, and correctness reference with your own.
Tensor is pre-1.0; use the pinned compiler environment and check the
[compatibility guide](compatibility.md) before transferring artifacts.

## 1. Prepare a producer

From a Tensor checkout, use Python 3.12 and the locked compiler dependencies:

```sh
uv sync --locked
uv run --locked python tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
```

Set the NVRTC bundle location in your current shell. On Linux:

```sh
export TENSOR_NVRTC_HOME="$PWD/build/nvrtc-12.9"
```

On Windows PowerShell:

```powershell
$env:TENSOR_NVRTC_HOME = "$PWD/build/nvrtc-12.9"
```

Run subsequent commands from the repository root. This producer uses TileLang
0.1.14, TVM FFI 0.1.12, and NVRTC 12.9. If your existing kernel uses a different
TileLang version, first make it construct and lower in this environment. A
working JIT kernel in another environment does not establish Tensor compatibility.
The [quickstart](quickstart.md) covers setup and the optional nvcc path in detail.

You can also keep your kernel in its existing project. In a separate, activated
Python 3.12 environment, install Tensor from a local checkout with its compiler
extra (replace the path with your checkout's absolute path):

```sh
python -m pip install "/path/to/tensor[compiler]"
python /path/to/tensor/tools/bootstrap_nvrtc.py --out build/nvrtc-12.9
```

Set `TENSOR_NVRTC_HOME` as above, then invoke `tensor` and `python` directly
instead of `uv run --locked tensor` and `uv run --locked python`. For example,
`tensor build my_kernel.py --out build/my_kernel.tbin` runs from your own project.
The compiler extra installs Tensor's exact TileLang/TVM FFI versions; use the
separate environment to assess compatibility with your original kernel.

## 2. Expose the uncompiled kernel

Tensor expects a Python file with a zero-argument `tensor_export()` function:

```python
def tensor_export():
    return {"kernel": my_prim_func, "outputs": ["out"]}
```

`kernel` must be the TileLang/TIRx `PrimFunc` itself. For a factory, call it with
the specialization you want to ship:

```python
def tensor_export():
    return {"kernel": make_matmul(128, 128, 64), "outputs": ["out"]}
```

These dimensions are build-time constants. Changing a factory argument requires
a new build unless your kernel explicitly uses supported symbolic dimensions.

If your factory has `@tilelang.jit`, calling the decorated function produces a
TileLang execution wrapper. Expose a plain factory that returns the `PrimFunc`
and return that from `tensor_export()`. Keep JIT compilation, autotuning, test
inputs, and benchmark calls in your existing test harness or behind
`if __name__ == "__main__":`; Tensor executes the export file during a build.
Do not pass a compiled `tilelang.compile(...)` result to Tensor.

For example, [examples/tilelang_add.py](../../examples/tilelang_add.py) contains
an ordinary factory plus the export hook:

```python
import tilelang.language as T


def make_add(n, block=128):
    @T.prim_func
    def add(a: T.Tensor((n,), "float32"),
            b: T.Tensor((n,), "float32"),
            out: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block), threads=block) as bx:
            for lane in T.Parallel(block):
                i = bx * block + lane
                if i < n:
                    out[i] = a[i] + b[i]

    return add


def tensor_export():
    return {"kernel": make_add(1025), "outputs": ["out"]}
```

The algorithm and tiling stay in TileLang. Tensor obtains argument metadata,
launch dimensions, and shared-memory requirements from lowering. Your original
JIT pass configurations and compilation options are not automatically carried
over; validate correctness and performance through the Tensor build path.

To explore factory settings and reuse measured choices in later builds, follow
[schedule discovery](schedule-search.md).

### Translate output indices to buffer names

For `add(a, b, out)`, TileLang's `out_idx=[2]` corresponds to Tensor's
`"outputs": ["out"]`. Use the buffer names in the `PrimFunc`, not integer indices.

| Export | Runtime call |
|---|---|
| `"outputs": ["out"]` | `out = kernel(a, b)` allocates and returns one buffer |
| `"outputs": ["out", "stats"]` | `out, stats = kernel(...)` returns buffers in that listed order |
| No `outputs`, or `"outputs": []` | Allocate buffers yourself and call `kernel.launch(...)` with every argument |

For an in-place update, pass the existing buffer through `launch` so its original
contents are available. Declaring it as an allocating output creates a new buffer.

## 3. Build and inspect an artifact

With an NVIDIA GPU on the producer:

```sh
uv run --locked tensor doctor
uv run --locked tensor inspect examples/tilelang_add.py --stage tirx
uv run --locked tensor build examples/tilelang_add.py --out build/tilelang-add.tbin
uv run --locked tensor inspect build/tilelang-add.tbin --stage manifest
```

Substitute your export file for `examples/tilelang_add.py`. The manifest records
argument names, dtypes, shapes, outputs, launch metadata, and the target. A build
without `--target` selects device 0's exact SM.

On a GPU-free producer, supply the intended consumer's architecture:

```sh
uv run --locked tensor doctor --target sm_86
uv run --locked tensor build examples/tilelang_add.py --target sm_86 --out build/tilelang-add.tbin
```

`sm_86` is an example, not a universal target. The consumer must have that exact
SM. Use a fresh output filename for each build; Tensor refuses to overwrite an
artifact. The compiler cache can reuse a matching previous compilation.

## 4. Run with Tensor buffers and validate

The included consumer checks both allocating and preallocated calls:

```sh
uv run --locked python examples/run_tilelang_add.py build/tilelang-add.tbin
```

Expected output:

```text
PASS: 1025 addition results match NumPy (allocated and reused outputs)
```

The central runtime calls are:

```python
import numpy as np
import tensor as tx

a_host = np.linspace(-1, 1, 1025, dtype=np.float32)
b_host = np.full((1025,), 2, dtype=np.float32)

with tx.Device() as device:
    kernel = device.load("build/tilelang-add.tbin")
    a = device.from_numpy(a_host)
    b = device.from_numpy(b_host)
    out = kernel(a, b)
    tx.assert_close(out, a_host + b_host)

    out = device.empty((1025,), "float32")
    kernel.launch(a, b, out)
    tx.assert_close(out, a_host + b_host)
```

Upload inputs with their declared dtype and shape. A direct kernel call accepts
Tensor buffers; it does not automatically accept Torch tensors or NumPy arrays.
Keep buffers and kernels within the owning device context. `to_numpy()` downloads
results; `tx.assert_close` also downloads a buffer before comparing it.

For your kernel, keep its original independent reference and choose appropriate
tolerances, including any reduced-precision accumulation. A successful build
does not replace a numerical check.

To time repeated calls with output allocation excluded, use this inside the same
device context after allocating `out`:

```python
print(tx.bench(kernel, (a, b, out), warmup=10, iters=100))
```

`launch` and `bench` take all arguments in the original `PrimFunc` order,
including output buffers. Timing reports host enqueue and launch-plus-stream
synchronization; it excludes uploads, downloads, and output allocation here.

## 5. Keep runtime parameters or existing framework storage

Use Python factory arguments for specialization and typed `PrimFunc` scalar
parameters for runtime values. For the included
[dynamic affine kernel](../../examples/dynamic_affine.py), the allocating call is
`out = kernel(a, b, scale=2.5)`: Tensor binds the FP32 scalar and infers symbolic
`size` from the input shapes. Follow the [runtime guide](runtime.md) for building
and calling that separate artifact. The static addition artifact above always
expects 1025 elements.

If your current harness uses CUDA Torch tensors, either upload CPU NumPy data
for an initial correctness check or use CUDA DLPack to borrow compatible device
storage. DLPack requires attention to stream ordering and lifetimes; see
[CUDA interoperability](runtime.md#cuda-dlpack-and-stream-interoperability).
The optional [PyTorch adapter](pytorch.md) handles installed kernels as custom
operators and provides framework-aware stream/allocation handling. Neither route
adds automatic backward support to an arbitrary existing kernel.

## 6. Ship the artifact to a compiler-free consumer

Build the core wheel and install it in a separate Python 3.12 environment using
the [installation guide](installation.md#install-a-standalone-wheel). Transfer
the wheel, `.tbin`, and your consumer script. For this example, transfer
`examples/run_tilelang_add.py`; it imports only Tensor and NumPy.

Run the script with that environment's Python, for example:

```sh
build/consumer/bin/python examples/run_tilelang_add.py build/tilelang-add.tbin
```

On Windows, use `build/consumer/Scripts/python.exe`. The consumer needs matching
CUDA hardware and a compatible NVIDIA driver. It does not need the TileLang
source, NVRTC bundle, or compiler packages. Use [kernel modules](modules.md) when
you need named exports, several specializations, or pinned reusable dependencies.

## Compatibility checks and common problems

The CUDA export profile requires one lowered device kernel, contiguous buffers
with positive extents, supported shape/launch expressions, and typed scalar
parameters. Cluster/cooperative launches require a dedicated adapter and are
rejected by the current path. Tensor's Python API validates shape, dtype,
alignment, device ownership, and scalar ranges before launch.

| Problem | What to change |
|---|---|
| Export is not a TIRx `PrimFunc` | Return the uncompiled kernel from a plain factory, before JIT/compile |
| Output name rejected | Use an actual buffer name from the `PrimFunc` signature |
| Allocating call has no declared outputs | Add `outputs`, or explicitly allocate and use `launch` |
| Shape or dtype mismatch | Match the specialization and declared dtype; build another specialization if needed |
| Noncontiguous frontend buffer rejected | Start with a contiguous signature and contiguous runtime storage |
| Original JIT works but Tensor lowering fails | Reproduce in the pinned environment and inspect `--stage target` or `--stage passes` |
| GPU target mismatch | Rebuild for the consumer's exact SM |

For this simple addition kernel, you can also build a separate WebGPU artifact:

```sh
uv sync --locked --extra webgpu
uv run --locked --extra webgpu tensor build examples/tilelang_add.py --provider webgpu --out build/tilelang-add-webgpu.tbin
uv run --locked --extra webgpu python examples/run_tilelang_add.py build/tilelang-add-webgpu.tbin --provider webgpu
```

This path emits WGSL without NVRTC. General CUDA TileLang kernels can use
operations outside the bounded WebGPU profile; inspect the
[WebGPU guide](webgpu.md) before attempting that migration.
