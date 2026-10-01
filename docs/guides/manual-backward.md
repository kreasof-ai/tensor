# Manual forward and backward

[Documentation](../README.md) · [Python runtime](runtime.md)

`tensor.ManualFunction` pairs two callbacks without importing a framework or
constructing an automatic differentiation graph. Each callback launches normal
Tensor executables. The application supplies output gradients and chooses the
order of backward calls, accumulation, recomputation and optimizer updates.

For example, given precompiled square and square-gradient kernels with arguments
`(x, y)` and `(x, dy, dx)` respectively:

```python
import tensor as tx

with tx.Device() as device:
    square = device.load("square.tbin")
    square_backward = device.load("square_backward.tbin")
    x = device.randn((1024,), "float32", seed=1)
    y = device.empty(x.shape, x.dtype)
    dx = device.empty(x.shape, x.dtype)
    dy = device.ones(x.shape, x.dtype)

    def forward(ctx, x):
        ctx.save_for_backward(x)
        square.launch(x, y)
        return y

    def backward(ctx, dy):
        saved_x, = ctx.saved_tensors
        square_backward.launch(saved_x, dy, dx)
        return dx

    operation = tx.ManualFunction(forward, backward, name="square")
    output, context = operation.forward(x)
    input_gradient, = operation.backward(context, dy)
    device.synchronize()
```

Forward returns a buffer or tuple of buffers; backward returns one buffer or
`None` per input. Every supplied output gradient must match its output's shape,
dtype and device session. Returned input gradients follow the same rule;
integer inputs must return `None`. Callbacks can use `ctx.metadata` for non-buffer
state. Save buffers through `ctx.save_for_backward` so their lifetime is explicit.

Each operation permits one outstanding forward context, protecting reused
workspace from a second forward. Consume it through backward or call
`context.discard()`. A backward callback consumes its context even if it fails,
because it may already have written part of a gradient. Argument/liveness checks
that fail before entering the callback leave the context available for correction
or discard. Saved buffers must stay live and their contents must stay unchanged
until consumption; the interface retains references but does not version writes.
Separate operation instances can have separate outstanding contexts.

This contract validates ownership and metadata; callbacks implement the actual
derivative. They also own parameter-gradient accumulation and stream ordering
when crossing sessions or frameworks. The ordinary Tensor session checks and
stream rules still apply.

`tensor_nn.NanoGPT`, from the optional `tensor-nn` distribution, demonstrates an
explicit reverse tape using this interface. Its static buffers and precompiled kernels execute embedding gradients, linear
and attention backward, LayerNorm, GELU, cross-entropy, clipping and AdamW.
See the [Phase 6 report](../research/phase6-nanogpt.md) for compilation, clean
consumer execution and numerical checks. This is a bounded training template;
the manual interface is available independently of that template.
