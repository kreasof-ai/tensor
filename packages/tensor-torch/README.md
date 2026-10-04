# Tensor Torch

Optional PyTorch adapter for Tensor's pre-1.0 runtime. It provides a
`torch.compile` inference backend and registered custom operators for installed
kernel modules. It does not provide general autograd or arbitrary graph fusion.

From a Python 3.12 checkout, install the pinned Tensor workspace and adapter:

```sh
uv sync --locked
uv pip install -e packages/tensor-torch
```

Use `uv run --no-sync` for subsequent commands to preserve the optional Torch
installation. CUDA compilation also requires the NVRTC bundle described in the
[quickstart](../../docs/guides/quickstart.md).

See the [PyTorch guide](../../docs/guides/pytorch.md) for supported regions,
custom operators, examples, and the optional PyTorch-versioned native executor.
The default wheel can use a portable submission extension when a host C compiler
is available during the build. The full C++ executor is a separate opt-in build.

The distribution is `tensor-torch`; the Python import is `tensor_torch`.
Install a matching `tensor-workspace` wheel alongside it when consuming a local
release bundle. Tensor Torch is licensed under [MIT](LICENSE).
