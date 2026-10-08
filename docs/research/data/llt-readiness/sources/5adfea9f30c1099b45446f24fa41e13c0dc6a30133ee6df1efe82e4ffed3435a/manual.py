"""Explicit, framework-independent forward/backward contracts; no differentiation engine."""
from __future__ import annotations

from tensor.runtime import Buffer


class BackwardContext:
    """Own references to saved buffers until backward or explicit discard."""

    def __init__(self, operation, inputs):
        self.operation = operation
        self.inputs = inputs
        self.saved_tensors = ()
        self.metadata = {}
        self.outputs = ()
        self.consumed = False

    def save_for_backward(self, *buffers):
        if self.consumed or any(not isinstance(b, Buffer) for b in buffers):
            raise ValueError("save_for_backward requires live Tensor buffers")
        for buffer in buffers:
            buffer._check()
            if buffer.device is not self.inputs[0].device:
                raise ValueError("saved buffers must share the input device session")
        self.saved_tensors = tuple(buffers)

    def discard(self):
        """Drop the saved references without executing backward."""
        self.consumed = True
        self.saved_tensors = self.inputs = self.outputs = ()
        self.metadata.clear()
        if self.operation._pending is self:
            self.operation._pending = None


class ManualFunction:
    """A user-authored pair of callbacks, with one outstanding forward context.

    forward(context, *inputs) returns a Buffer or tuple of Buffers.
    backward(context, *output_gradients) returns one Buffer/None per input.
    Gradients must match their input's shape, dtype and device. Integer inputs
    have no gradients. Accumulation, recomputation and kernel selection belong
    to the callbacks. Buffer contents must remain unchanged while saved.
    """

    def __init__(self, forward, backward, *, name=None):
        if not callable(forward) or not callable(backward):
            raise TypeError("forward and backward must be callable")
        self.forward_callback, self.backward_callback = forward, backward
        self.name = name or getattr(forward, "__name__", "manual")
        self._pending = None

    def forward(self, *inputs):
        if self._pending is not None:
            raise RuntimeError("consume or discard the previous backward context first")
        if not inputs or any(not isinstance(b, Buffer) for b in inputs):
            raise TypeError("manual functions require Tensor buffer inputs")
        for buffer in inputs:
            buffer._check()
            if buffer.device is not inputs[0].device:
                raise ValueError("inputs must share a device session")
        context = BackwardContext(self, inputs)
        try:
            output = self.forward_callback(context, *inputs)
            outputs = output if isinstance(output, tuple) else (output,)
            if not outputs or any(not isinstance(b, Buffer) for b in outputs):
                raise TypeError("forward must return Tensor buffers")
            for buffer in outputs:
                buffer._check()
                if buffer.device is not inputs[0].device:
                    raise ValueError("outputs must share the input device session")
            context.outputs = outputs
            self._pending = context
            return output, context
        except BaseException:
            context.discard()
            raise

    def backward(self, context, *gradients):
        if not isinstance(context, BackwardContext) or context.operation is not self:
            raise ValueError("backward context belongs to another operation")
        if context.consumed or self._pending is not context:
            raise RuntimeError("backward context is consumed or stale")
        if len(gradients) != len(context.outputs):
            raise ValueError("one gradient is required per forward output")
        for value, gradient in zip(context.outputs, gradients):
            self._match(value, gradient)
        for buffer in (*context.inputs, *context.saved_tensors):
            buffer._check()
        # A failed callback can have partially written gradients: consume its
        # context too, rather than permitting accidental double accumulation.
        try:
            result = self.backward_callback(context, *gradients)
            result = result if isinstance(result, tuple) else (result,)
            if len(result) != len(context.inputs):
                raise ValueError("backward must return one gradient or None per input")
            for value, gradient in zip(context.inputs, result):
                if gradient is not None:
                    if value.dtype.kind != 'f':
                        raise ValueError("integer inputs cannot have gradients")
                    self._match(value, gradient)
            return result
        finally:
            context.discard()

    @staticmethod
    def _match(value, gradient):
        if (not isinstance(gradient, Buffer) or gradient.device is not value.device
                or gradient.shape != value.shape or gradient.dtype != value.dtype):
            raise ValueError("gradient must match buffer shape, dtype and device")
        value._check()
        gradient._check()
