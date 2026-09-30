"""Portable SIMT tile lowering for the bounded WebGPU inference profile.

This producer-only pass expands GEMM and sum/max reductions into TIRx parallel
output loops and serial FP32 accumulation. TileLang still owns layout inference, copies,
reductions, synchronization and WGSL code generation.
"""

from __future__ import annotations


def lower_simt_gemm(kernel):
    import tvm
    ir = tvm.tirx
    counter = 0
    buffers = {}

    def collect(node):
        if isinstance(node, ir.SBlock):
            for buffer in node.alloc_buffers:
                if buffer.scope() in ("local.fragment", "shared.dyn"):
                    buffers[buffer] = ir.decl_buffer(buffer.shape, buffer.dtype, buffer.name,
                                                    scope="shared", data_alignment=buffer.data_alignment)
    ir.stmt_functor.post_order_visit(kernel.body, collect)

    def materialize(node):
        if isinstance(node, ir.Call) and str(node.op.name) == "tl.infinity":
            if str(node.dtype) != "float32":
                raise ValueError("WebGPU infinity supports float32 only")
            return ir.reinterpret("float32", ir.const(0x7f800000, "uint32"))
        if isinstance(node, ir.BufferLoad) and node.buffer in buffers:
            return ir.BufferLoad(buffers[node.buffer], node.indices)
        if isinstance(node, ir.BufferStore) and node.buffer in buffers:
            return ir.BufferStore(buffers[node.buffer], node.value, node.indices)
        if isinstance(node, ir.SBlock):
            if len(node.reads) or len(node.writes) or len(node.match_buffers):
                raise ValueError("WebGPU SIMT profile needs opaque tile blocks without explicit region annotations")
            return ir.SBlock(node.iter_vars, node.reads, node.writes, node.name_hint, node.body,
                             node.init, [buffers.get(b, b) for b in node.alloc_buffers],
                             node.match_buffers, node.annotations)
        if isinstance(node, ir.For) and "num_stages" in node.annotations:
            annotations = {str(k): v for k, v in node.annotations.items() if str(k) != "num_stages"}
            return ir.For(node.loop_var, node.min, node.extent, node.kind, node.body,
                          node.thread_binding, annotations)
        return None

    kernel = kernel.with_body(ir.stmt_functor.ir_transform(kernel.body, None, materialize))

    def barrier():
        return ir.Evaluate(ir.call_intrin("int32", "tirx.tvm_storage_sync", "shared"))

    def local_accumulator(name, initial, loop_var, extent, value, destination, indices):
        local = ir.decl_buffer((1,), "float32", name, scope="local")
        accumulator = ir.BufferLoad(local, [0])
        update = value(accumulator)
        block = ir.SBlock([], [], [], name, ir.SeqStmt([
            ir.BufferStore(local, initial, [0]),
            ir.For(loop_var, 0, extent, ir.ForKind.SERIAL, ir.BufferStore(local, update, [0])),
            ir.BufferStore(destination, accumulator, indices)]), alloc_buffers=[local])
        return ir.SBlockRealize([], True, block)

    def rewrite(node):
        nonlocal counter
        if not isinstance(node, ir.Evaluate) or not isinstance(node.value, ir.Call):
            return None
        call = node.value
        if str(call.op.name) == "tl.tileop.reduce":
            source, destination, operation, dimension, clear = call.args
            a, c = source.args[0], destination.args[0]
            axis = int(dimension)
            if len(a.indices) != 2 or len(c.indices) != 1 or axis not in (0, 1) or str(operation.value) not in ("max", "sum"):
                raise ValueError("WebGPU reduction supports sum/max of two-dimensional tiles along one axis")
            counter += 1
            row, reduction = [ir.Var(f"wgpu_{name}_{counter}", "int32") for name in ("row", "reduce")]
            source_index = [row, reduction] if axis == 1 else [reduction, row]
            value = ir.BufferLoad(a.buffer, [x+y for x,y in zip(a.indices, source_index)])
            index = [c.indices[0]+row]
            is_max = str(operation.value) == "max"
            initial = ir.reinterpret("float32", ir.const(0xff800000, "uint32")) if is_max else ir.const(0, str(c.buffer.dtype))
            initial = ir.if_then_else(clear, initial, ir.BufferLoad(c.buffer, index))
            body = local_accumulator(f"wgpu_reduce_acc_{counter}", initial, reduction, source.args[axis+2],
                                     lambda acc: ir.max(acc, value) if is_max else acc+value, c.buffer, index)
            return ir.SeqStmt([barrier(), ir.For(row, 0, source.args[3-axis], ir.ForKind.PARALLEL, body), barrier()])
        if str(call.op.name) != "tl.tileop.gemm":
            return None
        args = call.args
        if len(args) != 19 or int(args[14]) != 1 or int(args[15]) or int(args[16]):
            raise ValueError("WebGPU GEMM supports ordinary, unscaled synchronous tiles only")
        loads = []
        for region in args[:3]:
            if not isinstance(region, ir.Call) or str(region.op.name) != "tl.region":
                raise ValueError("WebGPU GEMM needs explicit two-dimensional buffer regions")
            load = region.args[0]
            if not isinstance(load, ir.BufferLoad) or len(load.indices) != 2:
                raise ValueError("WebGPU GEMM needs two-dimensional buffer regions")
            loads.append(load)
        a, b, c = loads
        if str(c.buffer.dtype) != "float32" or any(str(x.buffer.dtype) not in ("float16", "float32") for x in (a, b)):
            raise ValueError("WebGPU GEMM requires FP16/FP32 inputs and FP32 accumulation")
        counter += 1
        i, j, k = [ir.Var(f"wgpu_{name}_{counter}", "int32") for name in ("i", "j", "k")]
        m, n, depth = (int(args[index]) for index in (5, 6, 7))
        ai = [k, i] if int(args[3]) else [i, k]
        bi = [j, k] if int(args[4]) else [k, j]
        av = ir.BufferLoad(a.buffer, [x + y for x, y in zip(a.indices, ai)])
        bv = ir.BufferLoad(b.buffer, [x + y for x, y in zip(b.indices, bi)])
        ci = [c.indices[0] + i, c.indices[1] + j]
        initial = ir.if_then_else(args[9], ir.const(0, "float32"), ir.BufferLoad(c.buffer, ci))
        dot = local_accumulator(f"wgpu_gemm_acc_{counter}", initial, k, depth,
                                lambda acc: acc+ir.Cast("float32", av)*ir.Cast("float32", bv), c.buffer, ci)
        return ir.SeqStmt([barrier(), ir.For(i, 0, m, ir.ForKind.PARALLEL,
                      ir.For(j, 0, n, ir.ForKind.PARALLEL, dot)), barrier()])

    body = ir.stmt_functor.ir_transform(kernel.body, None, rewrite, ["tirx.Evaluate"])
    blocks = {}
    ir.stmt_functor.post_order_visit(body, lambda n: blocks.update({str(n.thread_binding.thread_tag): n})
                                    if isinstance(n, ir.For) and n.thread_binding is not None else None)
    if "blockIdx.z" in blocks:
        z = blocks["blockIdx.z"]
        y = blocks.get("blockIdx.y")
        if y is None:
            raise ValueError("WebGPU needs a y launch axis to flatten batch z")
        combined = ir.Var("wgpu_batch_head", "int32")
        def flatten(node):
            if not isinstance(node, ir.For) or node.thread_binding is None:
                return None
            tag = str(node.thread_binding.thread_tag)
            if tag == "blockIdx.z":
                return node.body
            if tag == "blockIdx.y":
                inner = ir.stmt_functor.substitute(node.body,
                    {y.loop_var: combined % y.extent, z.loop_var: combined // y.extent})
                binding = ir.IterVar(None, combined, y.thread_binding.iter_type, "blockIdx.y")
                return ir.For(combined, 0, y.extent*z.extent, node.kind, inner, binding, node.annotations)
            return None
        body = ir.stmt_functor.ir_transform(body, None, flatten, ["tirx.For"])
    return kernel.with_body(body)


def verify_uniform_barriers(function):
    """Reject synchronization in a branch/trip count varying across a workgroup.

    Native shader validation is not a substitute for this check: some native
    backends accept shaders with divergent barriers and produce wrong results.
    Buffer-dependent control flow is conservatively considered nonuniform.
    """
    import tvm
    ir = tvm.tirx
    varying, definitions = set(), []

    def collect(node):
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
        ir.stmt_functor.post_order_visit(expression, lambda n: found.append(True)
            if isinstance(n, ir.BufferLoad) or isinstance(n, ir.Var) and n in varying else None)
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
                if isinstance(call, ir.Call) and getattr(call.op, "name", "") == "tirx.tvm_storage_sync":
                    raise ValueError("WebGPU profile rejects nonuniform workgroup barriers")
            ir.stmt_functor.post_order_visit(node.value, check)
        return None

    def leave(node):
        if isinstance(node, (ir.IfThenElse, ir.For)):
            stack.pop()
        return None
    ir.stmt_functor.ir_transform(function.body, enter, leave,
                                ["tirx.IfThenElse", "tirx.For", "tirx.Evaluate"])
