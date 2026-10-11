"""One pre-normalized transformer block, including both residual derivatives."""

from tensor_nn.module import Module
from tensor_nn.templates import LayerNorm, Linear, CausalSelfAttention


class TransformerBlock(Module):
    def __init__(self, library, config, parameters, *, layer):
        super().__init__(library, name=f"block_{layer}")
        rows, width = config.rows, config.width

        def weight(name):
            return parameters[f"{layer}.{name}"]

        self.ln1 = self.register_module(
            "ln1", LayerNorm(library, rows, width, weight("ln1"))
        )
        self.qkv = self.register_module("qkv", Linear(library, rows, weight("qkv")))
        self.attention = self.register_module(
            "attention",
            CausalSelfAttention(
                library,
                batch=config.batch,
                sequence=config.sequence,
                width=width,
                heads=config.heads,
            ),
        )
        self.projection = self.register_module(
            "projection", Linear(library, rows, weight("proj"), residual=True)
        )
        self.ln2 = self.register_module(
            "ln2", LayerNorm(library, rows, width, weight("ln2"))
        )
        self.fc = self.register_module(
            "fc", Linear(library, rows, weight("fc"), activation=True)
        )
        self.fc_projection = self.register_module(
            "fc_projection", Linear(library, rows, weight("fcproj"), residual=True)
        )
        self.elements = rows * width
        self.grad1 = self.empty(self.elements)
        self.grad2 = self.empty(self.elements)
        self.bind(self._forward, self._backward)

    def _forward(self, ctx, x):
        tape = []

        def call(module, *inputs):
            output, context = module(*inputs)
            tape.append((module, context))
            return output

        normalized = call(self.ln1, x)
        qkv = call(self.qkv, normalized)
        attended = call(self.attention, qkv)
        residual = call(self.projection, attended, x)
        normalized = call(self.ln2, residual)
        activated = call(self.fc, normalized)
        output = call(self.fc_projection, activated, residual)
        ctx.metadata["tape"] = tape
        return output

    def _backward(self, ctx, gradient):
        tape = list(ctx.metadata["tape"])

        def reverse(dy):
            module, context = tape.pop()
            return module.backward(context, dy)[0]

        branch = reverse(gradient)  # MLP output projection.
        branch = reverse(branch)  # Fused linear + GELU.
        branch = reverse(branch)  # Second layer norm.
        self.run("add", dict(n=self.elements), gradient, branch, self.grad1)
        branch = reverse(self.grad1)  # Attention output projection.
        branch = reverse(branch)  # Causal attention.
        branch = reverse(branch)  # QKV projection.
        branch = reverse(branch)  # First layer norm.
        self.run("add", dict(n=self.elements), self.grad1, branch, self.grad2)
        assert not tape
        return self.grad2
