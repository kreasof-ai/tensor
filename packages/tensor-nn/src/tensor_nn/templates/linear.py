"""Linear module template with allocated outputs and explicit backward."""

from tensor_nn.module import Module


class Linear(Module):
    def __init__(
        self,
        library,
        rows,
        parameter,
        *,
        name="linear",
        activation=False,
        residual=False,
    ):
        if activation and residual:
            raise ValueError("choose either a GELU or residual epilogue")
        super().__init__(library, name=name)
        self.residual = residual
        self.register_parameter("weight", parameter)
        self.weight = parameter
        param = parameter
        o, i = param.shape
        r = rows
        y = self.empty(r * o)
        dx = self.empty(r * i)
        dw = self.empty(o * i)
        pre = self.empty(r * o) if activation else None
        dpre = self.empty(r * o) if activation else None

        def forward(ctx, x, weight, *skip):
            ctx.save_for_backward(x, weight, param.half)
            kind = (
                "gemm_gelu" if activation else "gemm_residual" if residual else "gemm"
            )
            extra = (pre,) if activation else skip if residual else ()
            self.run(
                kind, dict(batch=1, m=r, k=i, cols=o, tb=True), x, param.half, *extra, y
            )
            return y

        def backward(ctx, dy):
            x, _, half = ctx.saved_tensors
            original_gradient = dy
            if activation:
                self.run("gelu_backward", dict(n=r * o), pre, dy, dpre)
                dy = dpre
            self.run("gemm", dict(batch=1, m=r, k=o, cols=i), dy, half, dx)
            self.run("gemm", dict(batch=1, m=o, k=r, cols=i, ta=True), dy, x, dw)
            self.run("cast", dict(n=o * i), dw, param.grad)
            return (dx, param.grad, original_gradient) if residual else (dx, param.grad)

        self.bind(forward, backward)

    def forward(self, x, *skip):
        if len(skip) != int(self.residual):
            raise ValueError("a residual projection requires exactly one skip buffer")
        return self.function.forward(x, self.weight.weight, *skip)
