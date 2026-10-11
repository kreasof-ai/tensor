# Tensor Torch

`tensor-torch` binds Tensor artifacts to PyTorch storage, streams, and custom
operators. It also provides a bounded `torch.compile` backend. The distribution
is `tensor-torch`; the Python import is `tensor_torch`.

From a Python 3.12 Tensor checkout:

```sh
uv sync --locked
uv pip install -e packages/tensor-torch
```

Use `uv run --no-sync` afterward to preserve the optional Torch installation.
CUDA compilation requires the NVRTC bundle in the
[quickstart](../../docs/guides/quickstart.md). Existing artifacts execute without
the compiler. See the [PyTorch guide](../../docs/guides/pytorch.md) for contracts,
examples, and the optional PyTorch-versioned native executor.

## Reading the binding

Start with `src/tensor_torch/__init__.py`, which exports the public API, then
follow either path:

- `load(...)` → `loading.py` → `kernel.py` → `runtime/launch.py`.
- `torch.compile(..., backend=...)` → `compile/backend.py` →
  `compile/partition.py` → `compile/lowering.py` → `compile/cache.py` → `kernel.py`.

| Module | Responsibility |
| --- | --- |
| `loading.py` | Resolve a `.tbin` or an installed `module::export` reference. |
| `kernel.py` | Expose `Kernel`, allocate outputs, and delegate execution. |
| `signatures.py` | Validate tensors and resolve concrete or symbolic output metadata. |
| `registration.py` | Register functional custom ops, fake execution, and explicit output mutation. |
| `runtime/session.py` | Retain device sessions and synchronize/release adapter resources. |
| `runtime/streams.py` | Find and cache the current PyTorch stream. |
| `runtime/launch.py` | Bind the ABI, borrow DLPack buffers, record storage use, and submit launches. |
| `runtime/prepared.py` | Fixed-storage prepared calls and guarded pointer-rebinding launch plans. |
| `compile/partition.py` | Recognize supported operations and select compilation regions. |
| `compile/lowering.py` | Validate FX regions and emit Tensor source. |
| `compile/cache.py` | Validate and atomically publish compiled artifacts. |
| `compile/reporting.py` | Return compilation and fallback reports. |
| `compile/kernels.py`, `compile/attention.py` | Generic factories used by FX lowering. |
| `native/` | Portable C submission shim and optional Torch-versioned C++ executor. |

Runtime modules operate on artifacts and PyTorch tensors. They do not depend on
FX graph compilation. Core Tensor remains independent of Torch.

The current binding requires contiguous CUDA tensors. Functional outputs are
fully written and disjoint from inputs and one another. Prepared calls retain
storage and reject changed metadata or pointers; graph launch plans rebind
pointers after guards. Execution uses the current PyTorch stream and records
allocator lifetime. `close()` synchronizes before releasing Tensor resources.

Graph compilation supports bounded pointwise, matrix multiplication, and
attention regions. Unsupported operations fall back to PyTorch; compilation
failures propagate. Inference preserves eager autograd through fallback when
gradients are enabled. `Backend(training=True)` uses AOTAutograd to compile
supported forward/backward regions; it does not supply arbitrary kernel backward
rules. A dedicated `autograd.py` can be added when explicit backward registration
becomes part of the binding API.

## LLT ownership

LLT operators, autograd implementations, AdamW, KV cache, recurrent operations,
and workload-specific templates now live in the sibling
`loop-latent-transformer/packages/llt-tensor` package (`llt_tensor`). Their tests
and benchmarks moved with them. That package preserves a private copy of the
previous launch bridge and native sources, and does not import `tensor_torch`.
It can continue to run while this binding is rewritten.

Historical LLT research reports and hashed source snapshots in Tensor remain as
recorded. Their old module names and benchmark commands describe the original
runs; use the LLT package README for current execution paths.

## Validation

Tests are grouped by loading, registration, runtime, compilation, and autograd
behavior. From the Tensor checkout:

```sh
python -m pytest packages/tensor-torch/tests
TENSOR_P4_CUDA=1 python -m pytest packages/tensor-torch/tests
```

The default wheel can build a portable submission extension with a host C
compiler. The C++ executor is a separate opt-in build described in the guide.
Tensor Torch is licensed under [MIT](LICENSE).
