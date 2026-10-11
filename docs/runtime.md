# Runtime


Use `import tensor as tx` to execute precompiled kernels. The core runtime
depends only on NumPy; loading a CUDA `.tbin` does not import the compiler or
a tensor framework.

## Buffers and kernel calls

```python
import tensor as tx

with tx.Device() as device:
    kernel = device.load("build/elementwise.tbin")
    a = device.arange(129)
    b = device.ones((129,))
    result = kernel(a, b)
    print(result.to_numpy())

    # Reuse an output allocation for repeated calls.
    kernel.launch(a, b, result)
    device.synchronize()
```

`Device()` selects CUDA by default. `provider="cpu"` selects the Linux CPU
validation implementation; `provider="webgpu"` selects the optional native wgpu
provider. Load an artifact built for that provider. A device session owns its
buffers and executables; keep them within the session's lifetime.

The allocating call uses the artifact's declared outputs. `launch` binds all
kernel arguments, including outputs. Buffers expose immutable shape, dtype,
strides, and address metadata. Arguments must satisfy the artifact's dtype,
extent, contiguous-storage, and pointer-alignment requirements.

Device helpers include `empty`, `zeros`, `ones`, `full`, `randn`, `arange`, and
NumPy upload/download. `tx.assert_close` checks results against a host reference.
Copies and kernel launches use the session's execution ordering.

## Symbolic dimensions and scalars

Build [dynamic_affine.py](../examples/dynamic_affine.py) using the quickstart
commands, with a fresh output such as `build/dynamic_affine.tbin`:

```python
with tx.Device() as device:
    kernel = device.load("build/dynamic_affine.tbin")
    a = device.arange(1025)
    b = device.ones((1025,))
    result = kernel(a, b, scale=2.5)
```

Tensor infers symbolic `size` from the input shapes. An explicit `size=1025`
must agree with every bound buffer. The same binary handles different positive
lengths within its declared profile. The CLI equivalents are `--scalar scale=2.5`
and, when needed, `--scalar size=1025`.

Typed scalar arguments support bool, signed/unsigned integers from 8 to 64 bits,
and FP32/FP64. Binding checks overflow, finite floating-point values, and shape
consistency before launch.

## Timing

```python
with tx.Device() as device:
    kernel = device.load("build/elementwise.tbin")
    a = device.arange(129)
    b = device.ones((129,))
    c = device.empty((129,), "float32")
    print(tx.bench(kernel, (a, b, c), warmup=10, iters=100))
```

The benchmark reuses supplied buffers and reports host enqueue and execution
plus stream synchronization. These are different from a call that allocates an
output or an end-to-end framework invocation. The
[latency scaling report](../research/studies/latency-scaling/reports/latency-scaling.md) explains the measurement
protocols used to compare those paths.

## CUDA DLPack and stream interoperability

`device.from_dlpack(gpu_tensor)` borrows contiguous writable CUDA storage without
a copy or a framework import in Tensor. The runtime retains the producer's
managed tensor until release. FP16 and BF16 retain their two-byte storage.
`torch.from_dlpack(buffer)` can also borrow Tensor-owned CUDA storage: exporting
retains the buffer and blocks its release/session close until all consumers drop
it. `device.from_numpy(values, dtype="bfloat16")` rounds FP32 to BF16;
`buffer.to_numpy()` decodes BF16 to FP32 and `to_bytes()` returns raw storage.
CPU DLPack imports upload into device-owned storage.
WebGPU does not support borrowing external GPU pointers in its current profile.

For CUDA, import within the producer's stream context so DLPack can arrange the
initial dependency onto Tensor's stream. For later producer writes, call
`device.wait_for(producer_stream_handle)` before launching. Before another
stream consumes a borrowed output, call `device.handoff(consumer_stream_handle)`.

`tx.Device(stream=foreign_handle)` launches directly on a borrowed CUDA stream.
The caller owns that stream and must keep it alive until the session closes.
Tensor retains the primary CUDA context and restores the previous context after
its operations. Cleanup waits for submitted work and handed-off consumer streams
before releasing imported managed tensors.

The [PyTorch adapter](../packages/tensor-torch/README.md) provides current-stream submission and allocator
lifetime handling for Torch-owned tensors, including prepared calls and an
optional C++ executor.

## Lifetimes and native descriptors

Loaded kernels expose `descriptor` and `workspace_requirements()`.
`release()` waits for submitted work before unloading an image. Sessions resolve
executable and event identities with `get_executable()` and `get_event()`;
stale or foreign identities are rejected.

The [runtime ABI reference](reference/runtime-abi.md) defines buffer, scalar,
call, stream, executable, event, workspace, and error descriptors, plus native
C++/Rust hosting and provider-specific capabilities. Use that contract when
integrating a host outside the Python workbench.

For explicit derivatives, use the independent
[manual backward interface](runtime.md).

## Manual forward and backward


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

`tensor_nn_examples.nanogpt.NanoGPT`, from the optional `tensor-nn` distribution, demonstrates an
explicit reverse tape using this interface. Its static buffers and precompiled kernels execute embedding gradients, linear
and attention backward, LayerNorm, GELU, cross-entropy, clipping and AdamW.
See the [Phase 6 report](../research/studies/nanogpt-training/reports/phase6-nanogpt.md) for compilation, clean
consumer execution and numerical checks. This is a bounded training template;
the manual interface is available independently of that template.
