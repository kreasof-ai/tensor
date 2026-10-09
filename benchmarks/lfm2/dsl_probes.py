"""Small producer-only numerical probes for the package's DSL rounding macros."""


def half_conversion_kernel(n, unpack=False):
    import tilelang.language as T
    from tensor_llm.lfm2.kernels.webgpu import half_bits, round_half

    convert = half_bits() if unpack else round_half()

    @T.prim_func
    def kernel(
        x: T.Tensor((n,), "uint32" if unpack else "float32"), out: T.Tensor((n,), "float32")
    ):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
            for lane in T.Parallel(128):
                i = block * 128 + lane
                if i < n:
                    out[i] = convert(x[i])

    return kernel


def kernel_with_subgroup_size(kind, parameters, size):
    """Exercise the actual small-subgroup branch, without editing Python source."""
    import tvm
    from tensor_llm.lfm2.kernels.webgpu import make_kernel

    ir = tvm.tirx
    kernel = make_kernel(kind, parameters)

    def subgroup_call(node):
        return (
            isinstance(node, ir.Call)
            and getattr(node.op, "name", None) == "tirx.call_extern"
            and node.args[0].value == "tensor_subgroup_size"
        )

    def replace(node):
        if kind == "attention":
            # Force the score-computation guard only. Workgroup-wide subgroup
            # sum/max reductions still need the actual physical subgroup size.
            if isinstance(node, ir.GE) and subgroup_call(node.a):
                return ir.const(False, "bool")
        elif subgroup_call(node):
            return ir.const(size, "uint32")

    body = ir.stmt_functor.ir_transform(kernel.body, None, replace, ["tirx.Call", "tirx.GE"])
    return kernel.with_body(body)


def subgroup_source(kind, parameters, size):
    from tensor.compiler.entry import export_source

    return export_source(
        __name__,
        "kernel_with_subgroup_size",
        kind,
        parameters,
        size,
        dependencies=(
            "tensor_llm.lfm2.kernels.baseline",
            "tensor_llm.common.gguf",
            "tensor_llm.lfm2.kernels.webgpu",
            "tensor.compiler.entry",
            "tensor.compiler.webgpu_templates",
        ),
    )
