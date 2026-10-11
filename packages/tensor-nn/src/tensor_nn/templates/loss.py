"""Cross entropy with an explicit, loss-scaled gradient for the logits."""

import math
import numpy as np
from tensor_nn.module import Module


class CrossEntropy(Module):
    def __init__(self, library, *, rows, vocab):
        super().__init__(library, name="cross_entropy")
        self.geometry = dict(r=rows, v=vocab)
        self.output = self.empty(rows, "float32")
        self.parts = self.empty(rows * math.ceil(vocab / 1024) * 2, "float32")
        self.lse = self.empty(rows, "float32")
        self.gradient = self.empty(rows * vocab)
        self.bind(self._forward, self._backward)

    def _forward(self, ctx, logits, targets):
        self.run("ce_parts", self.geometry, logits, self.parts)
        self.run(
            "ce_loss", self.geometry, logits, targets, self.parts, self.lse, self.output
        )
        ctx.save_for_backward(logits, targets, self.lse)
        return self.output

    def _backward(self, ctx, gradient):
        logits, targets, lse = ctx.saved_tensors
        self.run(
            "ce_backward", self.geometry, logits, targets, lse, gradient, self.gradient
        )
        return self.gradient, None

    def read(self):
        losses = self.output.to_numpy()
        if not np.isfinite(losses).all():
            raise FloatingPointError("non-finite training loss")
        return float(np.mean(losses, dtype=np.float64))
