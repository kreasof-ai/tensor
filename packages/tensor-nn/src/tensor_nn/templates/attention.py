"""CausalSelfAttention module template with allocated outputs and explicit backward."""

from tensor_nn.module import Module


class CausalSelfAttention(Module):
    def __init__(
        self, library, *, batch, sequence, width, heads, name="causal_attention"
    ):
        super().__init__(library, name=name)
        b, s, c, h = batch, sequence, width, heads
        d = c // h
        n = b * s * c
        bh = b * h
        pack = dict(b=b, s=s, c=c, h=h)
        soft = dict(s=s, bh=bh, d=d)
        q, k, v, y, dq, dk, dv, dyh = [self.empty(n) for _ in range(8)]
        scores, prob, dp, ds = [self.empty(bh * s * s) for _ in range(4)]
        saved = self.empty(bh * s * s, "float32")
        merged = self.empty(n)
        dx = self.empty(3 * n)

        def forward(ctx, x):
            self.run("pack_qkv", pack, x, q, k, v)
            self.run("gemm", dict(batch=bh, m=s, k=d, cols=s, tb=True), q, k, scores)
            self.run("softmax", soft, scores, prob, saved)
            self.run("gemm", dict(batch=bh, m=s, k=s, cols=d), prob, v, y)
            self.run("merge_heads", pack, y, merged)
            ctx.save_for_backward(q, k, v, prob, saved)
            return merged

        def backward(ctx, dy):
            q, k, v, prob, saved = ctx.saved_tensors
            self.run("split_heads", pack, dy, dyh)
            self.run("gemm", dict(batch=bh, m=s, k=d, cols=s, tb=True), dyh, v, dp)
            self.run("gemm", dict(batch=bh, m=s, k=s, cols=d, ta=True), prob, dyh, dv)
            self.run("softmax_backward", soft, dp, saved, ds)
            self.run("gemm", dict(batch=bh, m=s, k=s, cols=d), ds, k, dq)
            self.run("gemm", dict(batch=bh, m=s, k=s, cols=d, ta=True), ds, q, dk)
            self.run("unpack_qkv", pack, dq, dk, dv, dx)
            return dx

        self.bind(forward, backward)
