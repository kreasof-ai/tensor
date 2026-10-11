"""Token and position embeddings, with accumulation into a tied output weight."""

from tensor_nn.module import Module


class TokenPositionEmbedding(Module):
    def __init__(self, library, token, position, *, batch, sequence, width, vocab):
        super().__init__(library, name="embedding")
        self.token = self.register_parameter("token", token)
        self.position = self.register_parameter("position", position)
        self.geometry = dict(b=batch, s=sequence, c=width, v=vocab)
        self.output = self.empty(batch * sequence * width)
        self.bind(self._forward, self._backward)

    def _forward(self, ctx, tokens):
        self.run(
            "embedding",
            self.geometry,
            tokens,
            self.token.half,
            self.position.half,
            self.output,
        )
        ctx.save_for_backward(tokens)
        return self.output

    def _backward(self, ctx, gradient):
        # The output projection wrote token.grad first. The scatter adds to it,
        # so embedding and classifier share one parameter and one optimizer state.
        self.run("zero", dict(n=self.position.grad.shape[0]), self.position.grad)
        self.run(
            "embedding_backward",
            self.geometry,
            ctx.saved_tensors[0],
            gradient,
            self.token.grad,
            self.position.grad,
        )
        return None  # Integer token ids have no gradient.
