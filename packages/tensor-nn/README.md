# Tensor NN

Optional static training templates for Tensor, including a manually differentiated
nanoGPT model. Consumers use Tensor, Tensor NN and NumPy; compiler and framework
packages are producer/reference dependencies.

Install both local distributions from the repository:

```sh
uv pip install -e . -e packages/tensor-nn
```

The canonical API is `from tensor_nn import GPTConfig, NanoGPT`. Existing
`tensor.nn` imports delegate to this installed package. Manual backward itself
remains in core: `from tensor import ManualFunction, BackwardContext`.

Build/validation/benchmark commands are under `benchmarks/nanogpt`. New bundles
record hashes of both distributions' implementation modules. Pre-reorganization
bundles require their original wheels; compile a new bundle for the new layout.

See [repository layout](../../docs/development.md) and the
[training report](../../docs/research/phase6-nanogpt.md).
