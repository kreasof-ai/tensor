"""Explicit native window/shared-memory and gating kernels for research models.

Existing Operators entry points keep their original specializations. Window sizes
include the current token; shared_prefix keys are globally causal and the remaining
keys are a causal local bank, normalized together in one softmax.
"""

import torch
from torch.autograd.function import once_differentiable
from .llt import Operators, _dtype

WINDOW_MODULE = "tensor_torch.templates.llt_window"
POINTWISE_MODULE = "tensor_torch.templates.recurrent_ops"


class RecurrentOperators(Operators):
    def window_attention(
        self, q, k, v, *, window_size, shared_prefix=0, query_offset=0, scale=None
    ):
        if type(window_size) != int or window_size < 1:
            raise ValueError(
                "window_size must be a positive integer including the current token"
            )
        if (
            type(shared_prefix) != int
            or k.ndim != 4
            or not 0 <= shared_prefix < k.shape[2]
        ):
            raise ValueError("shared_prefix must leave a nonempty local bank")
        # Validate without compiling or executing the ordinary attention kernel.
        if any(
            t.ndim != 4 or not t.is_contiguous() or t.device.type != "cuda"
            for t in (q, k, v)
        ):
            raise ValueError("window attention requires contiguous CUDA BHSD tensors")
        b, h, m, d = q.shape
        bk, kh, n, dk = k.shape
        dv = v.shape[-1]
        if (
            bk != b
            or v.shape[:3] != (b, kh, n)
            or dk != d
            or kh < 1
            or any(size < 1 for size in (*q.shape, *k.shape, *v.shape))
            or h % kh
            or d % 16
            or dv % 16
            or d > 256
            or dv > 128
        ):
            raise ValueError("unsupported window attention geometry")
        if (
            q.dtype not in (torch.float16, torch.bfloat16)
            or len({t.dtype for t in (q, k, v)}) != 1
            or len({t.device for t in (q, k, v)}) != 1
        ):
            raise ValueError("window attention requires matching FP16/BF16 tensors")
        if type(query_offset) != int or query_offset < 0:
            raise ValueError("query_offset must be nonnegative")
        import math

        scale = d**-0.5 if scale is None else float(scale)
        if not math.isfinite(scale):
            raise ValueError("scale must be finite")
        p = (
            b,
            h,
            kh,
            m,
            n,
            d,
            dv,
            _dtype(q),
            True,
            query_offset,
            scale,
            window_size,
            shared_prefix,
        )
        return _Window.apply(q, k, v, self, p)

    def shared_decode(self, q, global_cache, local_cache, *, scale=None, partitions=32):
        if torch.is_grad_enabled() and q.requires_grad:
            raise ValueError("shared cached decode is inference-only")
        if partitions not in (8, 16, 32, 64, 128, 256):
            raise ValueError("unsupported partition count")
        g, l = global_cache, local_cache
        if q.ndim != 4 or q.shape[2] != 1 or not q.is_contiguous():
            raise ValueError("decode requires contiguous one-token queries")
        if (
            g.keys.shape[:2] != l.keys.shape[:2]
            or g.keys.shape[-1] != l.keys.shape[-1]
            or g.values.shape[-1] != l.values.shape[-1]
        ):
            raise ValueError("cache geometry mismatch")
        if (
            q.shape[0] != g.keys.shape[0]
            or q.shape[1] % g.keys.shape[1]
            or q.shape[-1] != g.keys.shape[-1]
        ):
            raise ValueError("query geometry mismatch")
        if any(
            t.dtype != q.dtype or t.device != q.device
            for t in (g.keys, g.values, l.keys, l.values)
        ):
            raise ValueError("cache dtype/device mismatch")
        b, h, _, d = q.shape
        kh = g.keys.shape[1]
        dv = g.values.shape[-1]
        scale = d**-0.5 if scale is None else float(scale)
        p = (b, h, kh, g.capacity, l.capacity, d, dv, _dtype(q), scale, partitions)
        partial, stats = self.call(
            "shared_decode_partial",
            p,
            ["partial", "stats"],
            q,
            g.keys,
            l.keys,
            g.values,
            l.values,
            g.lengths,
            l.lengths,
            module="tensor_torch.templates.shared_decode",
        )
        return self.call(
            "decode_merge",
            (b, h, dv, _dtype(q), partitions),
            ["out"],
            partial,
            stats,
            module="tensor_torch.templates.llt_decode",
        )

    def silu(self, x):
        return _Unary.apply(x, self, "silu")

    def sigmoid(self, x):
        return _Unary.apply(x, self, "sigmoid")

    def multiply(self, x, y):
        if x.shape != y.shape:
            raise ValueError("multiply requires equal shapes")
        return _Multiply.apply(x, y, self)

    def blend(self, gate, state, proposal):
        if gate.shape != state.shape or state.shape != proposal.shape:
            raise ValueError("blend requires equal shapes")
        return _Blend.apply(gate, state, proposal, self)


class _Window(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, ops, p):
        out, lse, exact = ops.call(
            "attention_forward",
            p,
            ["out", "lse", "exact"],
            q,
            k,
            v,
            module=WINDOW_MODULE,
        )
        ctx.ops, ctx.p = ops, p
        ctx.save_for_backward(q, k, v, lse, exact)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        q, k, v, lse, exact = ctx.saved_tensors
        ops, p = ctx.ops, ctx.p
        b, h, kh, m, n, d, dv, dt = p[:8]
        dy = dy.contiguous()
        delta = ops.call("attention_delta", (b, h, m, dv, dt), ["delta"], exact, dy)
        dq = ops.call(
            "attention_dq", p, ["dq"], q, k, v, dy, lse, delta, module=WINDOW_MODULE
        )
        split = m >= 256 and h // kh >= 4
        dk, dvout = ops.call(
            "attention_dkv",
            (*p, split),
            ["dk", "dvout"],
            q,
            k,
            v,
            dy,
            lse,
            delta,
            module=WINDOW_MODULE,
        )
        if split:
            dk, dvout = ops.call(
                "attention_sum_heads",
                (b, h, kh, n, d, dv, dt),
                ["dk", "dvout"],
                dk,
                dvout,
            )
        return dq, dk, dvout, None, None


class _Unary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ops, kind):
        ctx.ops, ctx.kind = ops, kind
        ctx.save_for_backward(x)
        return ops.call(
            "unary",
            (x.numel(), _dtype(x), _dtype(x), kind),
            ["out"],
            x.contiguous().reshape(-1),
            module=POINTWISE_MODULE,
        ).reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        dx = ctx.ops.call(
            "unary",
            (x.numel(), _dtype(x), _dtype(dy), ctx.kind + "_dx"),
            ["out"],
            x.contiguous().reshape(-1),
            dy.contiguous().reshape(-1),
            module=POINTWISE_MODULE,
        )
        return dx.reshape(x.shape), None, None


class _Multiply(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y, ops):
        ctx.ops = ops
        ctx.save_for_backward(x, y)
        out_dtype = torch.promote_types(x.dtype, y.dtype)
        return ops.call(
            "multiply",
            (x.numel(), _dtype(x), _dtype(y), str(out_dtype).removeprefix("torch.")),
            ["out"],
            x.contiguous().reshape(-1),
            y.contiguous().reshape(-1),
            module=POINTWISE_MODULE,
        ).reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        x, y = ctx.saved_tensors
        ops = ctx.ops

        def grad(other, target):
            return ops.call(
                "multiply",
                (dy.numel(), _dtype(dy), _dtype(other), _dtype(target)),
                ["out"],
                dy.contiguous().reshape(-1),
                other.contiguous().reshape(-1),
                module=POINTWISE_MODULE,
            ).reshape(target.shape)

        return grad(y, x), grad(x, y), None


class _Blend(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, x, z, ops):
        ctx.ops = ops
        ctx.save_for_backward(g, x, z)
        dt = str(
            torch.promote_types(torch.promote_types(g.dtype, x.dtype), z.dtype)
        ).removeprefix("torch.")
        ctx.p = (x.numel(), _dtype(g), _dtype(x), _dtype(z), dt)
        return ops.call(
            "blend",
            ctx.p,
            ["out"],
            *(t.contiguous().reshape(-1) for t in (g, x, z)),
            module=POINTWISE_MODULE,
        ).reshape(x.shape)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy):
        g, x, z = ctx.saved_tensors
        dg, dx, dz = ctx.ops.call(
            "blend_backward",
            ctx.p,
            ["dg", "dx", "dz"],
            *(t.contiguous().reshape(-1) for t in (g, x, z, dy)),
            module=POINTWISE_MODULE,
        )
        return dg.reshape(g.shape), dx.reshape(x.shape), dz.reshape(z.shape), None
