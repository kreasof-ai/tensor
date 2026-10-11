"""Matmul implementation for WebGPU kernels."""

from __future__ import annotations


def lower_simt_gemm(kernel):
    import tvm

    ir = tvm.tirx
    counter = 0
    buffers = {}
    reduction_mode = str(kernel.attrs.get("tensor.webgpu.reduction", "ordered"))
    if reduction_mode not in ("ordered", "tree"):
        raise ValueError("WebGPU reduction must be ordered or tree")

    def collect(node):
        if isinstance(node, ir.SBlock):
            for buffer in node.alloc_buffers:
                if buffer.scope() in ("local.fragment", "shared.dyn"):
                    buffers[buffer] = ir.decl_buffer(
                        buffer.shape,
                        buffer.dtype,
                        buffer.name,
                        scope="shared",
                        data_alignment=buffer.data_alignment,
                    )

    ir.stmt_functor.post_order_visit(kernel.body, collect)

    buffer_vars = {old.data: new.data for old, new in buffers.items()}

    def remap_annotations(annotations):
        """Re-key Var-keyed annotations onto the substituted buffers.

        Layout annotations address a block's buffers by Var, and substituting
        alloc_buffers hands TileLang fresh Vars for the same names. Carrying the
        annotations across unchanged would leave them pointing at buffers the
        block no longer allocates, which layout inference cannot resolve.
        """
        if not annotations:
            return annotations
        remapped = {}
        for key, value in annotations.items():
            if isinstance(value, ir.Var) or not hasattr(value, "items"):
                remapped[key] = value
                continue
            entries = {}
            for entry_key, entry_value in value.items():
                entries[buffer_vars.get(entry_key, entry_key)] = entry_value
            remapped[key] = type(value)(entries) if entries else value
        return type(annotations)(remapped) if remapped else annotations

    def materialize(node):
        if isinstance(node, ir.Call) and str(node.op.name) == "tl.infinity":
            if str(node.dtype) != "float32":
                raise ValueError("WebGPU infinity supports float32 only")
            return ir.reinterpret("float32", ir.const(0x7F800000, "uint32"))
        if isinstance(node, ir.BufferLoad) and node.buffer in buffers:
            return ir.BufferLoad(buffers[node.buffer], node.indices)
        if isinstance(node, ir.BufferStore) and node.buffer in buffers:
            return ir.BufferStore(buffers[node.buffer], node.value, node.indices)
        if isinstance(node, ir.SBlock):
            if len(node.reads) or len(node.writes) or len(node.match_buffers):
                raise ValueError(
                    "WebGPU SIMT profile needs opaque tile blocks without explicit region annotations"
                )
            return ir.SBlock(
                node.iter_vars,
                node.reads,
                node.writes,
                node.name_hint,
                node.body,
                node.init,
                [buffers.get(b, b) for b in node.alloc_buffers],
                node.match_buffers,
                remap_annotations(node.annotations),
            )
        if isinstance(node, ir.For) and "num_stages" in node.annotations:
            annotations = {
                str(k): v for k, v in node.annotations.items() if str(k) != "num_stages"
            }
            return ir.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                node.body,
                node.thread_binding,
                annotations,
            )
        return None

    kernel = kernel.with_body(
        ir.stmt_functor.ir_transform(kernel.body, None, materialize)
    )

    def barrier():
        return ir.Evaluate(ir.call_intrin("int32", "tirx.tvm_storage_sync", "shared"))

    # Give a thread stable ownership of its output microtiles across the entire
    # serial K loop. Intermediate fragment consumers (notably attention's score
    # reductions) must still use the ordinary materialized tile lowering.
    accumulation = str(kernel.attrs.get("tensor.webgpu.gemm_accumulation", "auto"))
    if accumulation not in ("auto", "register", "shared"):
        raise ValueError("WebGPU GEMM accumulation must be auto, register or shared")
    micro = int(kernel.attrs.get("tensor.webgpu.gemm_microtile", 2))
    if micro not in (1, 2, 4):
        raise ValueError("WebGPU GEMM microtile must be 1, 2 or 4")
    bindings = {}
    ir.stmt_functor.post_order_visit(
        kernel.body,
        lambda node: (
            bindings.update({str(node.thread_binding.thread_tag): node})
            if isinstance(node, ir.For) and node.thread_binding is not None
            else None
        ),
    )
    thread = bindings.get("threadIdx.x")
    eligible_threads = (
        thread is not None
        and isinstance(thread.extent, ir.IntImm)
        and int(thread.extent) > 0
        and all(
            tag not in bindings
            or isinstance(bindings[tag].extent, ir.IntImm)
            and int(bindings[tag].extent) == 1
            for tag in ("threadIdx.y", "threadIdx.z")
        )
    )
    fragments = set(buffers[b] for b in buffers if b.scope() == "local.fragment")

    def retain_accumulators(node):
        nonlocal counter
        if (
            accumulation == "shared"
            or not eligible_threads
            or not isinstance(node, ir.For)
        ):
            return None
        if node.kind != ir.ForKind.SERIAL or node.thread_binding is not None:
            return None
        statements = (
            list(node.body.seq) if isinstance(node.body, ir.SeqStmt) else [node.body]
        )
        last = statements[-1]
        if not isinstance(last, ir.Evaluate) or not isinstance(last.value, ir.Call):
            return None
        call = last.value
        if str(call.op.name) != "tl.tileop.gemm":
            return None
        args = call.args
        if len(args) != 19 or any(
            not isinstance(args[index], ir.IntImm)
            for index in (3, 4, 5, 6, 7, 9, 14, 15, 16)
        ):
            return None
        if int(args[9]) or int(args[14]) != 1 or int(args[15]) or int(args[16]):
            return None
        loads = []
        for region in args[:3]:
            if not isinstance(region, ir.Call) or str(region.op.name) != "tl.region":
                return None
            load = region.args[0]
            if not isinstance(load, ir.BufferLoad) or len(load.indices) != 2:
                return None
            loads.append(load)
        a, b, c = loads
        if c.buffer not in fragments or str(c.buffer.dtype) != "float32":
            return None
        if any(
            x.buffer.scope() != "shared"
            or str(x.buffer.dtype) not in ("float16", "float32")
            for x in (a, b)
        ):
            return None
        unsafe = []

        def inspect(part):
            if (
                isinstance(part, (ir.BufferLoad, ir.BufferStore))
                and part.buffer == c.buffer
            ):
                unsafe.append(True)
            if isinstance(part, ir.Var) and part.same_as(c.buffer.data):
                unsafe.append(True)
            if isinstance(part, ir.Call) and str(part.op.name) in (
                "tl.tileop.gemm",
                "tl.tileop.reduce",
            ):
                unsafe.append(True)
            if isinstance(part, ir.SBlock) and (
                part.alloc_buffers or part.init is not None
            ):
                unsafe.append(True)

        for statement in statements[:-1]:
            ir.stmt_functor.post_order_visit(statement, inspect)
        for index in (*a.indices, *b.indices, *c.indices):
            ir.stmt_functor.post_order_visit(
                index,
                lambda part: (
                    unsafe.append(True)
                    if isinstance(part, ir.Var) and part.same_as(node.loop_var)
                    else None
                ),
            )
        if unsafe:
            return None
        m, n, depth = (int(args[index]) for index in (5, 6, 7))
        if min(m, n, depth) <= 0:
            return None
        # Long reductions recover the cost of a larger private live set. The
        # RX 6700 XT ablation also found repeatable shorter-K regressions, so
        # leave those on the existing schedule unless a producer/tuner opts in.
        if accumulation == "auto" and (
            not isinstance(node.extent, ir.IntImm) or int(node.extent) * depth < 2048
        ):
            return None
        tm, tn = (micro if extent % micro == 0 else 1 for extent in (m, n))
        threads = int(thread.extent)
        count = (m // tm) * (n // tn)
        groups = (count + threads - 1) // threads
        # Keep the first implementation bounded; larger output tiles need a
        # separate resource-aware schedule rather than unbounded private arrays.
        if groups * tm * tn > 64:
            return None
        counter += 1
        local = ir.decl_buffer(
            (groups * tm * tn,),
            "float32",
            f"wgpu_gemm_loop_acc_{counter}",
            scope="local",
        )
        k = ir.Var(f"wgpu_loop_k_{counter}", "int32")
        initialize, updates, finish = [], [], []

        def group_statement(parts, tid):
            body = parts[0] if len(parts) == 1 else ir.SeqStmt(parts)
            return ir.IfThenElse(tid < count, body, None) if count % threads else body

        for group in range(groups):
            tid = thread.loop_var + group * threads
            row, col = tid // (n // tn), tid % (n // tn)
            starts, dots, stores = [], [], []
            for rr in range(tm):
                for cc in range(tn):
                    slot = group * tm * tn + rr * tn + cc
                    ii, jj = row * tm + rr, col * tn + cc
                    indices = [c.indices[0] + ii, c.indices[1] + jj]
                    aa = [k, ii] if int(args[3]) else [ii, k]
                    bb = [jj, k] if int(args[4]) else [k, jj]
                    lhs = ir.Cast(
                        "float32",
                        ir.BufferLoad(a.buffer, [x + y for x, y in zip(a.indices, aa)]),
                    )
                    rhs = ir.Cast(
                        "float32",
                        ir.BufferLoad(b.buffer, [x + y for x, y in zip(b.indices, bb)]),
                    )
                    acc = ir.BufferLoad(local, [slot])
                    starts.append(
                        ir.BufferStore(local, ir.BufferLoad(c.buffer, indices), [slot])
                    )
                    dots.append(ir.BufferStore(local, acc + lhs * rhs, [slot]))
                    stores.append(ir.BufferStore(c.buffer, acc, indices))
            initialize.append(group_statement(starts, tid))
            updates.append(group_statement(dots, tid))
            finish.append(group_statement(stores, tid))
        dot = ir.For(
            k,
            0,
            depth,
            ir.ForKind.SERIAL,
            updates[0] if len(updates) == 1 else ir.SeqStmt(updates),
        )
        loop_body = ir.SeqStmt([*statements[:-1], barrier(), dot, barrier()])
        loop = ir.For(
            node.loop_var,
            node.min,
            node.extent,
            node.kind,
            loop_body,
            node.thread_binding,
            node.annotations,
        )
        body = ir.SeqStmt([barrier(), *initialize, loop, *finish, barrier()])
        block = ir.SBlock(
            [], [], [], f"wgpu_gemm_loop_{counter}", body, alloc_buffers=[local]
        )
        return ir.SBlockRealize([], True, block)

    kernel = kernel.with_body(
        ir.stmt_functor.ir_transform(
            kernel.body, None, retain_accumulators, ["tirx.For"]
        )
    )

    def local_accumulator(name, initial, loop_var, extent, value, destination, indices):
        local = ir.decl_buffer((1,), "float32", name, scope="local")
        accumulator = ir.BufferLoad(local, [0])
        update = value(accumulator)
        block = ir.SBlock(
            [],
            [],
            [],
            name,
            ir.SeqStmt(
                [
                    ir.BufferStore(local, initial, [0]),
                    ir.For(
                        loop_var,
                        0,
                        extent,
                        ir.ForKind.SERIAL,
                        ir.BufferStore(local, update, [0]),
                    ),
                    ir.BufferStore(destination, accumulator, indices),
                ]
            ),
            alloc_buffers=[local],
        )
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
            rank = len(a.indices)
            if (
                rank not in (1, 2)
                or len(c.indices) != 1
                or not 0 <= axis < rank
                or str(operation.value) not in ("max", "sum")
            ):
                raise ValueError(
                    "WebGPU reduction supports sum/max of one/two-dimensional tiles along one axis"
                )
            counter += 1
            row, reduction = [
                ir.Var(f"wgpu_{name}_{counter}", "int32") for name in ("row", "reduce")
            ]
            source_index = (
                [reduction]
                if rank == 1
                else [row, reduction]
                if axis == 1
                else [reduction, row]
            )
            value = ir.BufferLoad(
                a.buffer, [x + y for x, y in zip(a.indices, source_index)]
            )
            index = [c.indices[0] + row]
            is_max = str(operation.value) == "max"
            initial = (
                ir.reinterpret("float32", ir.const(0xFF800000, "uint32"))
                if is_max
                else ir.const(0, str(c.buffer.dtype))
            )
            initial = ir.if_then_else(clear, initial, ir.BufferLoad(c.buffer, index))
            body = local_accumulator(
                f"wgpu_reduce_acc_{counter}",
                initial,
                reduction,
                source.args[axis + 2],
                lambda acc: ir.max(acc, value) if is_max else acc + value,
                c.buffer,
                index,
            )
            rows = 1 if rank == 1 else source.args[3 - axis]
            if reduction_mode == "tree":
                extent = int(source.args[axis + 2])
                count = int(rows)
                padded = 1 << (extent - 1).bit_length()
                scratch = ir.decl_buffer(
                    (count, padded), "float32", f"wgpu_tree_{counter}", scope="shared"
                )
                identity = (
                    ir.reinterpret("float32", ir.const(0xFF800000, "uint32"))
                    if is_max
                    else ir.const(0, "float32")
                )
                stage = ir.BufferStore(
                    scratch,
                    ir.if_then_else(reduction < extent, value, identity),
                    [row, reduction],
                )
                steps = [
                    barrier(),
                    ir.For(
                        row,
                        0,
                        count,
                        ir.ForKind.PARALLEL,
                        ir.For(reduction, 0, padded, ir.ForKind.PARALLEL, stage),
                    ),
                    barrier(),
                ]
                stride = padded // 2
                while stride:
                    lhs = ir.BufferLoad(scratch, [row, reduction])
                    rhs = ir.BufferLoad(scratch, [row, reduction + stride])
                    update = ir.BufferStore(
                        scratch,
                        ir.max(lhs, rhs) if is_max else lhs + rhs,
                        [row, reduction],
                    )
                    rr, kk = (
                        ir.Var(f"wgpu_tree_row_{counter}_{stride}", "int32"),
                        ir.Var(f"wgpu_tree_lane_{counter}_{stride}", "int32"),
                    )
                    update = ir.stmt_functor.substitute(
                        update, {row: rr, reduction: kk}
                    )
                    steps.extend(
                        [
                            ir.For(
                                rr,
                                0,
                                count,
                                ir.ForKind.PARALLEL,
                                ir.For(kk, 0, stride, ir.ForKind.PARALLEL, update),
                            ),
                            barrier(),
                        ]
                    )
                    stride //= 2
                result = ir.BufferLoad(scratch, [row, 0])
                previous = ir.BufferLoad(c.buffer, index)
                result = ir.if_then_else(
                    clear,
                    result,
                    ir.max(previous, result) if is_max else previous + result,
                )
                rr = ir.Var(f"wgpu_tree_output_{counter}", "int32")
                store = ir.stmt_functor.substitute(
                    ir.BufferStore(c.buffer, result, index), {row: rr}
                )
                steps.append(ir.For(rr, 0, count, ir.ForKind.PARALLEL, store))
                steps.append(barrier())
                block = ir.SBlock(
                    [],
                    [],
                    [],
                    f"wgpu_tree_reduce_{counter}",
                    ir.SeqStmt(steps),
                    alloc_buffers=[scratch],
                )
                return ir.SBlockRealize([], True, block)
            return ir.SeqStmt(
                [barrier(), ir.For(row, 0, rows, ir.ForKind.PARALLEL, body), barrier()]
            )
        if str(call.op.name) != "tl.tileop.gemm":
            return None
        args = call.args
        if len(args) != 19 or int(args[14]) != 1 or int(args[15]) or int(args[16]):
            raise ValueError(
                "WebGPU GEMM supports ordinary, unscaled synchronous tiles only"
            )
        loads = []
        for region in args[:3]:
            if not isinstance(region, ir.Call) or str(region.op.name) != "tl.region":
                raise ValueError(
                    "WebGPU GEMM needs explicit two-dimensional buffer regions"
                )
            load = region.args[0]
            if not isinstance(load, ir.BufferLoad) or len(load.indices) != 2:
                raise ValueError("WebGPU GEMM needs two-dimensional buffer regions")
            loads.append(load)
        a, b, c = loads
        if str(c.buffer.dtype) != "float32" or any(
            str(x.buffer.dtype) not in ("float16", "float32") for x in (a, b)
        ):
            raise ValueError(
                "WebGPU GEMM requires FP16/FP32 inputs and FP32 accumulation"
            )
        counter += 1
        i, j, k = [
            ir.Var(f"wgpu_{name}_{counter}", "int32") for name in ("i", "j", "k")
        ]
        m, n, depth = (int(args[index]) for index in (5, 6, 7))
        ai = [k, i] if int(args[3]) else [i, k]
        bi = [j, k] if int(args[4]) else [k, j]
        av = ir.BufferLoad(a.buffer, [x + y for x, y in zip(a.indices, ai)])
        bv = ir.BufferLoad(b.buffer, [x + y for x, y in zip(b.indices, bi)])
        ci = [c.indices[0] + i, c.indices[1] + j]
        initial = ir.if_then_else(
            args[9], ir.const(0, "float32"), ir.BufferLoad(c.buffer, ci)
        )
        micro = int(kernel.attrs.get("tensor.webgpu.gemm_microtile", 2))
        if micro not in (1, 2, 4):
            raise ValueError("WebGPU GEMM microtile must be 1, 2 or 4")
        tm, tn = (micro if extent % micro == 0 else 1 for extent in (m, n))
        local = ir.decl_buffer(
            (tm * tn,), "float32", f"wgpu_gemm_acc_{counter}", scope="local"
        )
        initialize, updates, finish = [], [], []
        for row_offset in range(tm):
            for col_offset in range(tn):
                slot = row_offset * tn + col_offset
                ii, jj = i * tm + row_offset, j * tn + col_offset
                indices = [c.indices[0] + ii, c.indices[1] + jj]
                initial = ir.if_then_else(
                    args[9], ir.const(0, "float32"), ir.BufferLoad(c.buffer, indices)
                )
                aa = [k, ii] if int(args[3]) else [ii, k]
                bb = [jj, k] if int(args[4]) else [k, jj]
                lhs = ir.Cast(
                    "float32",
                    ir.BufferLoad(a.buffer, [x + y for x, y in zip(a.indices, aa)]),
                )
                rhs = ir.Cast(
                    "float32",
                    ir.BufferLoad(b.buffer, [x + y for x, y in zip(b.indices, bb)]),
                )
                acc = ir.BufferLoad(local, [slot])
                initialize.append(ir.BufferStore(local, initial, [slot]))
                updates.append(ir.BufferStore(local, acc + lhs * rhs, [slot]))
                finish.append(ir.BufferStore(c.buffer, acc, indices))
        inner = updates[0] if len(updates) == 1 else ir.SeqStmt(updates)
        body = ir.SeqStmt(
            [*initialize, ir.For(k, 0, depth, ir.ForKind.SERIAL, inner), *finish]
        )
        block = ir.SBlock(
            [],
            [],
            [],
            f"wgpu_gemm_register_tile_{counter}",
            body,
            alloc_buffers=[local],
        )
        dot = ir.SBlockRealize([], True, block)
        return ir.SeqStmt(
            [
                barrier(),
                ir.For(
                    i,
                    0,
                    m // tm,
                    ir.ForKind.PARALLEL,
                    ir.For(j, 0, n // tn, ir.ForKind.PARALLEL, dot),
                ),
                barrier(),
            ]
        )

    body = ir.stmt_functor.ir_transform(kernel.body, None, rewrite, ["tirx.Evaluate"])
    blocks = {}
    ir.stmt_functor.post_order_visit(
        body,
        lambda n: (
            blocks.update({str(n.thread_binding.thread_tag): n})
            if isinstance(n, ir.For) and n.thread_binding is not None
            else None
        ),
    )
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
                inner = ir.stmt_functor.substitute(
                    node.body,
                    {y.loop_var: combined % y.extent, z.loop_var: combined // y.extent},
                )
                binding = ir.IterVar(
                    None, combined, y.thread_binding.iter_type, "blockIdx.y"
                )
                return ir.For(
                    combined,
                    0,
                    y.extent * z.extent,
                    node.kind,
                    inner,
                    binding,
                    node.annotations,
                )
            return None

        body = ir.stmt_functor.ir_transform(body, None, flatten, ["tirx.For"])
    return kernel.with_body(body)
