"""GELU module template with allocated outputs and explicit backward."""

from tensor_nn.module import Module


class GELU(Module):
    def __init__(self, library, n, *, name="gelu"):
        super().__init__(library, name=name)
        y = self.empty(n)
        dx = self.empty(n)

        def forward(ctx, x):
            ctx.save_for_backward(x)
            self.run("gelu", dict(n=n), x, y)
            return y

        def backward(ctx, dy):
            self.run("gelu_backward", dict(n=n), ctx.saved_tensors[0], dy, dx)
            return dx

        self.bind(forward, backward)
