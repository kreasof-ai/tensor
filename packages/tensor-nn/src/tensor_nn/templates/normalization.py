"""LayerNorm module template with allocated outputs and explicit backward."""

from tensor_nn.module import Module


class LayerNorm(Module):
    def __init__(self, library, rows, width, parameter, *, name="layer_norm"):
        super().__init__(library, name=name)
        self.register_parameter("weight", parameter)
        self.weight = parameter
        r, c = rows, width
        param = parameter
        y = self.empty(r * c)
        normal = self.empty(r * c, "float32")
        inverse = self.empty(r, "float32")
        dx = self.empty(r * c)
        parts = self.empty(r * c, "float32")

        def forward(ctx, x, weight):
            self.run("norm", dict(r=r, c=c), x, weight, y, normal, inverse)
            ctx.save_for_backward(weight, normal, inverse)
            return y

        def backward(ctx, dy):
            weight, normal, inverse = ctx.saved_tensors
            self.run(
                "norm_backward", dict(r=r, c=c), dy, weight, normal, inverse, dx, parts
            )
            self.run("column_sum", dict(r=r, c=c), parts, param.grad)
            return dx, param.grad

        self.bind(forward, backward)

    def forward(self, x):
        return self.function.forward(x, self.weight.weight)
