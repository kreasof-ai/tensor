"""Barriers implementation for WebGPU kernels."""

from __future__ import annotations


def verify_uniform_barriers(function):
    """Reject synchronization in a branch/trip count varying across a workgroup.

    Native shader validation is not a substitute for this check: some native
    backends accept shaders with divergent barriers and produce wrong results.
    Read-only global loads with uniform indices are uniform under an explicit
    no-alias contract. Shared memory and possibly aliased loads remain varying.
    """
    import tvm

    ir = tvm.tirx
    varying, definitions, written = set(), [], set()
    noalias = bool(function.attrs and function.attrs.get("tirx.noalias", False))

    def collect(node):
        if isinstance(node, ir.BufferStore):
            written.add(node.buffer.data)
        if isinstance(node, ir.AttrStmt) and node.attr_key == "thread_extent":
            if str(node.node.thread_tag).startswith("threadIdx."):
                varying.add(node.node.var)
        if isinstance(node, ir.For) and node.thread_binding is not None:
            if str(node.thread_binding.thread_tag).startswith("threadIdx."):
                varying.add(node.loop_var)
        if type(node).__name__ in ("LetStmt", "Bind"):
            definitions.append((node.var, node.value))

    ir.stmt_functor.post_order_visit(function.body, collect)

    def nonuniform(expression):
        found = []
        ir.stmt_functor.post_order_visit(
            expression,
            lambda n: (
                found.append(True)
                if isinstance(n, ir.BufferLoad) and (
                    not noalias or n.buffer.scope() != "global" or n.buffer.data in written
                )
                or isinstance(n, ir.Var)
                and n in varying
                else None
            ),
        )
        return bool(found)

    changed = True
    while changed:
        changed = False
        for var, value in definitions:
            if var not in varying and nonuniform(value):
                varying.add(var)
                changed = True
    stack = []

    def enter(node):
        if isinstance(node, ir.IfThenElse):
            stack.append(nonuniform(node.condition))
        elif isinstance(node, ir.For):
            stack.append(nonuniform(node.min) or nonuniform(node.extent))
        elif isinstance(node, ir.Evaluate) and any(stack):

            def check(call):
                if (
                    isinstance(call, ir.Call)
                    and getattr(call.op, "name", "") == "tirx.tvm_storage_sync"
                ):
                    raise ValueError(
                        "WebGPU profile rejects nonuniform workgroup barriers"
                    )

            ir.stmt_functor.post_order_visit(node.value, check)
        return None

    def leave(node):
        if isinstance(node, (ir.IfThenElse, ir.For)):
            stack.pop()
        return None

    ir.stmt_functor.ir_transform(
        function.body, enter, leave, ["tirx.IfThenElse", "tirx.For", "tirx.Evaluate"]
    )
