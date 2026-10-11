"""Enumerate the kernel specializations used by the nanoGPT training example."""

import math
from tensor_nn.kernels import identity
from tensor_nn_examples.nanogpt.config import parameter_shapes


def requirements(config):
    r, c = config.rows, config.width
    values = {}

    def add(kind, **p):
        values[identity(kind, p)] = (kind, p)

    for i, o in [(c, 3 * c), (c, c), (c, 4 * c), (4 * c, c), (c, config.vocab)]:
        kind = (
            "gemm_gelu"
            if (i, o) == (c, 4 * c)
            else "gemm_residual" if (i, o) in ((c, c), (4 * c, c)) else "gemm"
        )
        add(kind, batch=1, m=r, k=i, cols=o, tb=True)
        add("gemm", batch=1, m=r, k=o, cols=i)
        add("gemm", batch=1, m=o, k=r, cols=i, ta=True)
        add("cast", n=i * o)
    bh = config.batch * config.heads
    s = config.sequence
    d = c // config.heads
    add("gemm", batch=bh, m=s, k=d, cols=s, tb=True)
    add("gemm", batch=bh, m=s, k=s, cols=d)
    add("gemm", batch=bh, m=s, k=s, cols=d, ta=True)
    for kind in ("pack_qkv", "unpack_qkv", "merge_heads", "split_heads"):
        add(kind, b=config.batch, s=s, c=c, h=config.heads)
    for kind in ("softmax", "softmax_backward"):
        add(kind, s=s, bh=bh, d=d)
    for kind in ("embedding", "embedding_backward"):
        add(kind, b=config.batch, s=s, c=c, v=config.vocab)
    for kind in ("norm", "norm_backward", "column_sum"):
        add(kind, r=r, c=c)
    add("add", n=r * c)
    for kind in ("gelu", "gelu_backward"):
        add(kind, n=r * 4 * c)
    for kind in ("ce_parts", "ce_loss", "ce_backward"):
        add(kind, r=r, v=config.vocab)
    add("zero", n=s * c)
    shapes = parameter_shapes(config)
    total = sum(math.ceil(math.prod(shape) / 1024) for shape in shapes.values())
    for name, shape in shapes.items():
        n = math.prod(shape)
        add("sumsq", n=n, total=total, scale=config.loss_scale)
        add("adamw", n=n, scale=config.loss_scale, decay=0.1 if len(shape) > 1 else 0.0)
    add("sum_parts", n=total)
    add("clip", n=math.ceil(total / 1024), limit=config.clip)
    return values
