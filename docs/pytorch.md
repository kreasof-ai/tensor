# PyTorch adapter

`tensor-torch` is a separate wheel. Core `tensor-workspace` still depends only
on NumPy. Install the adapter beside your existing compatible PyTorch install:

```sh
uv build --wheel
uv build --wheel packages/tensor-torch --out-dir build/torch-wheel
pip install dist/tensor_workspace-0.1.0-py3-none-any.whl build/torch-wheel/tensor_torch-0.1.0-*.whl
```

The validated environment is Python 3.12, PyTorch 2.14, CUDA driver 595.91.07,
and NVIDIA A10G (`sm_86`). Metadata permits PyTorch 2.8–2.14; other versions
need their own execution validation. No PyTorch wheel is bundled with Tensor.
Portable adapter wheels include a small native CUDA submission shim built using
Python headers alone. An additional C++ executor can be built against PyTorch
2.14 to move allocation and tensor/stream handling out of Python. Neither build
needs CUDA toolkit headers or libraries. Installed wheels need no host compiler.
A Python/ctypes fallback remains available. The original Phase 4 performance
acceptance applies to portable wheels with the submission shim.

## Native execution

On a builder with Python 3.12, PyTorch 2.14, setuptools, and a C++20 compiler:

```sh
TENSOR_TORCH_BUILD_NATIVE=1 uv build --wheel --no-build-isolation \
  packages/tensor-torch --out-dir build/torch-native-wheel
pip install build/torch-native-wheel/tensor_torch-0.1.0-*.whl
```

The installed adapter automatically selects the matching executor. Set
`TENSOR_TORCH_NATIVE=0` before importing the adapter to use the portable path.
The C++ wheel uses CPython and PyTorch's version-specific ABIs; it has a
`cp312-cp312` tag, rather than the portable wheel's `cp312-abi3` tag. Other
supported PyTorch versions use the portable path. CI builds and imports both
variants on Linux and Windows, without a system CUDA toolkit.

For compiled FX regions, the executor checks tensor metadata and alignment,
allocates outputs through ATen, rebinds kernel pointers, selects the current
PyTorch stream, records allocator usage, and submits through the driver.
Fixed `kernel.prepare(...)` calls also use C++, including storage/metadata
checks and copied scalar arguments. Their input and output tensors remain
retained. FX launch plans retain descriptors and shape specifications, rather
than the tensors from the first invocation. Argument storage is serialized
across concurrent calls; the GIL is released during allocation and submission.

Initial artifact loading, DLPack/ABI validation, graph specialization, and
functional custom-operator registration remain in Python. Ordinary custom-op
calls still use the Python bridge; use compiled FX regions or prepared calls
for the native executor. The outer Dynamo call wrapper also remains in Python.
The executor uses PyTorch's [generic device/stream interface](https://github.com/pytorch/pytorch/blob/main/c10/core/impl/DeviceGuardImplInterface.h)
to access the current stream without CUDA toolkit headers.
See the [measured C++ executor results](research/native-executor.md) for full-call
and isolated launch timings and validation evidence.

## Inference graphs

```python
import torch
import tensor_torch

@torch.compile(backend="tensor")
def affine(a, b):
    return torch.relu(a * 2 + b)

with torch.inference_mode():
    a = torch.randn(129, device="cuda")
    b = torch.randn_like(a)
    result = affine(a, b)
```

A graph-cache miss emits a standalone TileLang kernel, lowers TIRx, and compiles
an exact-SM cubin through NVRTC 12.9. Install the core `compiler` extra and the
local NVRTC bundle on that producer. NVRTC remains the default; direct PTX is
experimental after v1. Cached artifacts run without TileLang, TVM or NVRTC.

Use an instance for explicit options and observable coverage:

```python
compiler = tensor_torch.Backend(cache_dir="build/torch-cache")
compiled = torch.compile(lambda a, b: torch.relu(a * 2 + b), backend=compiler)
# After executing compiled(...):
print(compiler.report)
```

Reports contain compiled region nodes, artifact paths, shape specializations,
cache hits, preparation times, unsupported FX nodes, and semantic fallbacks.
Compiler failures propagate. Unsupported nodes run through their original
PyTorch FX targets, including CPU tensors. PyTorch continues to own graph breaks.
A semantic fallback region has no compiled specialization; graph capture alone
is not evidence that Tensor executed a kernel.

Supported CUDA inference profiles:

* Contiguous, nonempty FP32/FP16 pointwise add/subtract/multiply/divide, negate,
  ReLU, sigmoid and tanh. Broadcasts cover full tensors, trailing bias, and
  singleton tensors. Floating constants and `alpha` are specialized. Intermediate
  FP16 operations round to FP16, and ReLU propagates NaNs.
* Rank-two FP16 matmul/linear, with fused pointwise bias and activation epilogues,
  including M/N/K tails and chains of linear layers. Accumulation is FP32.
* FP16 forward self-attention via functional SDPA: contiguous `[B,H,S,D]`,
  head dimension 64/128, causal or noncausal, tail sequences and batches.
  No custom mask, dropout, GQA, or KV-cache decoding. The implementation uses
  online softmax and never materializes a global score matrix.

CUDA inputs need the frontend's 64-byte pointer alignment. Unaligned views,
unsupported layouts/dtypes/broadcasts, and unsupported SDPA options fall back.
Each region accepts up to 64 concrete shape specializations by default;
`max_specializations` changes that bound. Dynamic Dynamo graphs can reuse one
captured region while selecting cached concrete shape binaries. This is guarded
shape specialization, rather than arbitrary symbolic tiled code generation.
Warm up every specialization before CUDA graph capture.

## Installed kernels as custom operators

```python
kernel = tensor_torch.load("tensor-ops::dynamic_affine", project="my-project")
# Or tensor_torch.load("kernel.tbin")
with torch.inference_mode():
    output = kernel(a, b, 2.0)
    kernel.into(a, b, 2.0, outputs=[torch.empty_like(a)])
```

Run `tensor install` for the project first. Module resolution uses packaged
exact-target images by default; `compile=True` explicitly allows source fallback.

Functional custom ops allocate outputs through PyTorch, declare no input
mutation, and have FakeTensor implementations for symbolic output shapes.
`into` registers an explicit mutable-output operator. Exported kernels must
read input buffers, completely write declared outputs, and obey these contracts.
Outputs cannot alias inputs or each other; arbitrary alias analysis is not
encoded in `.tbin`. Functional kernels do not register a backward formula.
PyTorch raises if their output is differentiated without an autograd registration.
The registration follows [PyTorch's custom-operator contracts](https://docs.pytorch.org/docs/main/library.html).

The bridge validates its first concrete binding through the core DLPack import
and ABI descriptors. Prepared bindings reuse those descriptors with new guarded
pointers. PyTorch owns allocation and producer ordering. Every invocation launches
on the current PyTorch stream and records allocator usage on that stream; there
is no per-call synchronization. As with native PyTorch operations, establish
producer/consumer ordering when using different streams.

```python
with torch.inference_mode():
    call = kernel.prepare(a, b, 2.0)
    call()                 # fixed storage; no output allocation
    output = call.outputs[0]
```

Prepared calls retain all tensors and validate storage/metadata before launch.
In-place data changes are supported. Changing storage, shape, dtype, or strides
invalidates the binding. `tensor_torch.close()` synchronizes devices and unloads
adapter-owned modules; call it after workers finish. It never frees PyTorch
storage. Prepared calls cannot execute after session closure.

## Autograd evaluation

The default backend preserves eager autograd when gradients are enabled and
inputs require gradients. Inference measurements use `torch.inference_mode()`.

`Backend(training=True)` evaluates AOTAutograd by lowering its separate forward
and backward FX graphs. The regression compiles affine/ReLU forward and an
actual backward multiply region, validates both input gradients, and reports
PyTorch fallback for detach/threshold-backward operations. This demonstrates
compiled work in both stages; it is not complete compiled-training coverage.
Training remains experimental. Tensor does not claim a FlashAttention backward.

The inference-first Phase 4 scope is accepted; see the [exit report](research/phase4-exit.md)
for the 20-case benchmark, exact performance gates, cross-platform wheels and
compiler-free GPU transfer evidence.
