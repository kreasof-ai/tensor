"""Unroll implementation for WebGPU kernels."""

from __future__ import annotations


def lower_explicit_unroll(kernel):
    """Opt-in expansion of bounded T.unroll loops before WGSL code generation.

    TileLang's WGSL path may retain unrolled loops as ordinary shader loops.
    Expand only an explicitly requested schedule; serial reductions stay loops.
    """
    import tvm

    mode = str(kernel.attrs.get("tensor.webgpu.loop_unroll", "none"))
    if mode not in ("none", "explicit"):
        raise ValueError("invalid WebGPU loop unroll mode")
    if mode == "none":
        return kernel
    ir = tvm.tirx

    def check(node):
        if isinstance(node, ir.For) and node.kind == ir.ForKind.UNROLLED:
            if (
                not isinstance(node.extent, ir.IntImm)
                or not 0 <= int(node.extent) <= 16
            ):
                raise ValueError(
                    "explicit WebGPU unroll requires a static extent at most 16"
                )

    ir.stmt_functor.post_order_visit(kernel.body, check)
    name = str(kernel.attrs["global_symbol"])
    return ir.transform.UnrollLoop()(tvm.IRModule({name: kernel}))[name]
