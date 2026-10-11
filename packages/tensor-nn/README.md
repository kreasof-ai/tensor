# Tensor NN

`tensor-nn` is a collection of readable, executable templates for building neural
network modules on Tensor. nanoGPT is the worked example: it composes the modules,
records forward contexts, writes the backward pass explicitly, and runs a training
loop. Runtime execution uses Tensor and NumPy. PyTorch supplies independent
reference checks in `benchmarks/training/nanogpt/reference.py`.

## Start with the example

From the Tensor checkout, using Python 3.12:

```sh
uv sync --locked
```

The workspace installs `tensor-nn` alongside Tensor. Alternatively, install the
local distributions with `python -m pip install -e . -e packages/tensor-nn`.
To compile the small diagnostic model, install Tensor's compiler dependencies
and configure the NVRTC bundle as described in the
[quickstart](../../docs/quickstart.md). Then build for your GPU's SM:

```sh
uv run --no-sync python -m benchmarks.nanogpt.producer \
  --diagnostic --target sm_89 --out build/nanogpt-example
uv run --no-sync python packages/tensor-nn/examples/train_nanogpt.py \
  --bundle build/nanogpt-example --steps 10
```

`sm_89` targets the L40S diagnostic environment. Use the target matching your
CUDA GPU. The producer requires a fresh output directory. The resulting bundle
contains compiled artifacts and fingerprints of the runtime and module templates;
consumers can execute it without importing Torch, TileLang, or TVM.

The training example spells out the central contract:

```python
logits, model_context = model(model.tokens)
losses, loss_context = model.criterion(logits, model.targets)
logit_gradient, _ = model.criterion.backward(loss_context, model.loss_grad)
model.backward(model_context, logit_gradient)
model.optimizer.step()
```

`model.step()` is a convenience wrapper around the same sequence. Each module
owns static output and scratch buffers. A forward call returns its output and
context; backward consumes that context. Consume the previous context, or call
`discard_contexts()`, before another forward. Gradients and residual accumulation
are explicit. Keep saved inputs and weights unchanged until backward consumes
their contexts. `model.reset()` discards pending contexts and restores the initial
weights and optimizer state.

## Reading and extending the templates

```text
src/tensor_nn/
├── module.py          # Composition, named parameters, forward/backward contexts
├── parameter.py       # Master weights, compute copies, gradients, training state
├── library.py         # Verified artifact bundles and reusable Tensor calls
├── optim.py           # Gradient unscaling, global clipping, AdamW
├── templates/
│   ├── linear.py
│   ├── normalization.py
│   ├── activation.py
│   ├── attention.py
│   ├── embedding.py
│   └── loss.py
└── kernels/           # Matching Tensor DSL factories, grouped by operation

examples/
├── train_nanogpt.py    # Explicit forward, backward and optimizer loop
└── nanogpt/
    ├── config.py      # Static geometry and deterministic initialization
    ├── plan.py        # Kernel specializations required by that geometry
    ├── bundle.py      # Validate nanoGPT configuration and kernel coverage
    ├── block.py       # Transformer composition and residual derivatives
    └── model.py       # Embeddings, blocks, tied head, loss and training
```

`tensor_nn` exposes the reusable templates. The separately named
`tensor_nn_examples.nanogpt` package maps to `examples/nanogpt` and is included
in distributions so the example also works outside a checkout. Importing
`tensor_nn` does not load the model.

Read [the model](examples/nanogpt/model.py), then
[one transformer block](examples/nanogpt/block.py), then
[the linear template](src/tensor_nn/templates/linear.py) and its
[kernel factory](src/tensor_nn/kernels/linear.py). The
[training script](examples/train_nanogpt.py) provides the full caller-side loop.

To add a module, subclass `Module`, allocate its buffers, and bind your forward
and backward callbacks. Call `register_parameter` and `register_module` for its
parameters and children. `named_parameters()` walks those registrations and
counts a shared `Parameter` object once. nanoGPT passes the same token parameter
to its embedding and output projection: classifier gradients are written first,
then embedding backward adds its scatter gradients into that buffer.

Parameters retain their logical shapes while runtime buffers are flattened. A
linear weight has shape `(out_features, in_features)`; a layer-norm weight is a
width-sized vector. `Parameter.from_numpy(device, values)` allocates the FP32
master weight, FP16 compute copy, gradient, and optimizer state.

The templates export `Linear`, `LayerNorm`, `GELU`, `CausalSelfAttention`,
`TokenPositionEmbedding`, and `CrossEntropy`. Linear can fuse a GELU or residual
addition. Each callback declares the kernels it runs and the buffers it saves.
Add new factory specializations to the producer plan and include new implementation
modules in `provenance.py` when extending the example.

## Example scope and checks

The nanoGPT profile is a static, bias-free, pre-normalized causal transformer with
learned positional embeddings, exact GELU, tied token/classifier weights, and
zero dropout. It uses FP16 compute with FP32 master weights, reductions,
gradients, and AdamW state. Shapes are fixed by `GPTConfig` and the compiled
bundle; width, sequence, and vocabulary must be multiples of sixteen, and head
dimensions must be multiples of eight. Rebuild bundles after changing templates.

The diagnostic configuration is one layer, two heads, width 32, vocabulary 256,
and sequence 16. The default configuration retains the existing larger training
profile. These templates demonstrate explicit manual differentiation; Tensor's
`ManualFunction` and `BackwardContext` define the underlying buffer contract.

```sh
uv run --no-sync python -m pytest packages/tensor-nn/tests
uv run --no-sync python -m benchmarks.nanogpt.validate \
  --bundle build/nanogpt-example --steps 10 --out build/nanogpt-check.json
```

The second command uses PyTorch to compare loss, every parameter gradient, global
norm, weights, and optimizer state. See the historical
[training report](../../research/studies/nanogpt-training/reports/phase6-nanogpt.md) for the original measured
profile. Its results and old source identities remain unchanged.
