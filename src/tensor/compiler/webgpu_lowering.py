"""Portable SIMT tile lowering for the bounded WebGPU inference profile.

This producer-only pass retains eligible GEMM accumulators across K loops and offers ordered
or parallel-tree sum/max reductions. TileLang owns layout inference, copies,
synchronization and WGSL generation; typed helpers cover packed loads.
"""

from __future__ import annotations


def lower_wgsl_intrinsics(source):
    """Small typed WGSL operations absent from TileLang's scalar codegen.

    Keep packed byte alignment out of TIR's signed/unsigned index arithmetic.
    The WGSL backend then owns exact unsigned shifts and half unpacking.
    """
    helpers = {
        "tensor_byte_align_u32": '''fn tensor_byte_align_u32(lo:u32, hi:u32, byte:u32)->u32 {
  let shift=byte*8u;
  return (lo>>shift) | select(0u,hi<<((32u-shift)&31u),shift!=0u);
}''',
        "tensor_unpack_f16": '''fn tensor_unpack_f16(bits:u32)->f32 {
  return unpack2x16float(bits).x;
}''',
    }
    for name, helper in helpers.items():
        if name+'(' in source:
            source += '\n'+helper+'\n'
    if "dot4I8Packed(" in source:
        source = 'requires packed_4x8_integer_dot_product;\n'+source
    if "tensor_subgroup_size()" in source:
        source = source.replace("threadIdx : vec3<u32>", "threadIdx : vec3<u32>,\n  @builtin(subgroup_size) tensorSubgroupSize : u32")
        source = source.replace("tensor_subgroup_size()", "tensorSubgroupSize")
    return source


def register_matmul_schedule(rows, depth, columns, lhs_value, rhs_value, *,
                             tile_m=16, tile_n=32, tile_k=32, threads=128, pad=0,
                             lhs_pad=0, lhs_transpose=False, dot_width=1, unroll=False):
    """Reusable SIMT schedule with accumulators retained across all K tiles.

    Producers supply load/rounding expressions; this schedule owns distribution,
    shared layout, barriers and register microtiles. No GPU/compiler imports.
    """
    if any(type(v) is not int or v<=0 for v in (rows,depth,columns,tile_m,tile_n,tile_k,threads)) or type(pad) is not int or pad<0:
        raise ValueError('register matmul dimensions must be positive integers')
    if type(lhs_pad) is not int or lhs_pad<0 or type(lhs_transpose) is not bool or type(unroll) is not bool or dot_width not in (1,4) or tile_k%dot_width:
        raise ValueError('invalid register matmul shared layout or dot width')
    nr=tile_n//2
    if nr==0:raise ValueError('invalid register matmul tile')
    if tile_n%2 or threads%nr or tile_m%(threads//nr) or depth%tile_k:
        raise ValueError('invalid register matmul tile')
    row_stride=threads//nr
    micro_m=tile_m//row_stride
    initialize='\n'.join(f'    acc{i}{j} = T.alloc_var("float32")\n    acc{i}{j} = 0' for i in range(micro_m) for j in range(2))
    def lhs_index(row,k):return f'[{k}, {row}]' if lhs_transpose else f'[{row}, {k}]'
    if dot_width==1:
        loads='\n'.join(f'            left{i} = T.cast(lhs{lhs_index(f"mr + {i*row_stride}","kk")}, "float32")' for i in range(micro_m))
        loads+='\n'+'\n'.join(f'            right{j} = T.cast(rhs[kk, nr + {j*nr}], "float32")' for j in range(2))
        multiply='\n'.join(f'            acc{i}{j} = acc{i}{j} + left{i} * right{j}' for i in range(micro_m) for j in range(2))
    else:
        loads='\n'.join(f'            left{i} = T.call_extern("float32x4", "vec4<f32>", '+', '.join(f'T.cast(lhs{lhs_index(f"mr + {i*row_stride}",f"kk * 4 + {lane}")}, "float32")' for lane in range(4))+')' for i in range(micro_m))
        loads+='\n'+'\n'.join(f'            right{j} = T.call_extern("float32x4", "vec4<f32>", '+', '.join(f'T.cast(rhs[kk * 4 + {lane}, nr + {j*nr}], "float32")' for lane in range(4))+')' for j in range(2))
        multiply='\n'.join(f'            acc{i}{j} = acc{i}{j} + T.call_extern("float32", "dot", left{i}, right{j})' for i in range(micro_m) for j in range(2))
    lhs_shape=(tile_k,tile_m+lhs_pad) if lhs_transpose else (tile_m,tile_k+lhs_pad)
    stores='\n'.join(f'''    if by * {tile_m} + mr + {i*row_stride} < {rows}:
        if bx * {tile_n} + nr + {j*nr} < {columns}:
            out[(by * {tile_m} + mr + {i*row_stride}) * {columns} + bx * {tile_n} + nr + {j*nr}] = acc{i}{j}''' for i in range(micro_m) for j in range(2))
    return f'''with T.Kernel(T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n}), threads={threads}) as (by, bx):
    tx = T.get_thread_binding()
    mr = tx // {nr}
    nr = tx % {nr}
    lhs = T.alloc_shared({lhs_shape}, "float16")
    rhs = T.alloc_shared(({tile_k}, {tile_n+pad}), "float16")
    value = T.alloc_var("float32")
{initialize}
    for tile in T.serial({depth//tile_k}):
        for i, j in T.Parallel({tile_m}, {tile_k}):
            value = T.if_then_else(by * {tile_m} + i < {rows}, x[(by * {tile_m} + i) * {depth} + tile * {tile_k} + j], 0)
            lhs{lhs_index('i','j')} = {lhs_value}
        for i, j in T.Parallel({tile_n}, {tile_k}):
            value = {rhs_value}
            rhs[j, i] = value
        T.sync_threads()
        for kk in T.{'unroll' if unroll else 'serial'}({tile_k//dot_width}):
{loads}
{multiply}
        T.sync_threads()
{stores}'''


def lower_explicit_unroll(kernel):
    """Opt-in expansion of bounded T.unroll loops before WGSL code generation.

    TileLang's WGSL path may retain unrolled loops as ordinary shader loops.
    Expand only an explicitly requested schedule; serial reductions stay loops.
    """
    import tvm
    mode=str(kernel.attrs.get('tensor.webgpu.loop_unroll','none'))
    if mode not in ('none','explicit'):raise ValueError('invalid WebGPU loop unroll mode')
    if mode=='none':return kernel
    ir=tvm.tirx
    def check(node):
        if isinstance(node,ir.For) and node.kind==ir.ForKind.UNROLLED:
            if not isinstance(node.extent,ir.IntImm) or not 0<=int(node.extent)<=16:
                raise ValueError('explicit WebGPU unroll requires a static extent at most 16')
    ir.stmt_functor.post_order_visit(kernel.body,check)
    name=str(kernel.attrs['global_symbol'])
    return ir.transform.UnrollLoop()(tvm.IRModule({name:kernel}))[name]


def outer_product_matmul_schedule(rows, depth, columns, *, tile_m=64, tile_n=64,
                                  tile_k=16, micro_m=4, micro_n=4, threads=256,
                                  lhs_layout='km', lhs_pad=0, rhs_pad=0,
                                  owner_axis='column', unroll=4, fma=True,
                                  dtype='float32', epilogue='gemm', explicit_unroll=False,
                                  lhs_value=None):
    """Staged SIMT outer products with complete-K private FP32 accumulators.

    Cooperative loads preserve row-major A and transposed row-major B storage.
    Shared layouts and output ownership are independent scheduling choices.
    The generated source uses x[M*K], w[N*K], optional bias[N], and out[M*N].
    """
    dims=(rows,depth,columns,tile_m,tile_n,tile_k,micro_m,micro_n,threads,unroll)
    if any(type(v) is not int or v<=0 for v in dims):raise ValueError('invalid outer-product dimensions')
    if threads not in (64,128,256,512) or tile_m%micro_m or tile_n%micro_n or (tile_m//micro_m)*(tile_n//micro_n)!=threads:
        raise ValueError('invalid outer-product ownership')
    if micro_m*micro_n>64 or tile_k%unroll or unroll>16:
        raise ValueError('invalid outer-product register footprint or unroll')
    if lhs_layout not in ('mk','km') or owner_axis not in ('row','column') or dtype not in ('float32','float16') or epilogue not in ('gemm','linear') or type(fma) is not bool or type(explicit_unroll) is not bool:
        raise ValueError('invalid outer-product layout or arithmetic')
    if any(type(v) is not int or v<0 for v in (lhs_pad,rhs_pad)):
        raise ValueError('invalid outer-product padding')
    if lhs_value is not None and (type(lhs_value) is not str or not lhs_value.strip()):
        raise ValueError('invalid outer-product activation expression')
    activation=(lhs_value or f'x[({{row}}) * {depth} + ({{k}})]').format(
        row=f'by * {tile_m} + ar',k=f'tile * {tile_k} + ak')
    lhs_shape=(tile_k,tile_m+lhs_pad) if lhs_layout=='km' else (tile_m,tile_k+lhs_pad)
    rhs_shape=(tile_k,tile_n+rhs_pad)
    size=(lhs_shape[0]*lhs_shape[1]+rhs_shape[0]*rhs_shape[1])*(4 if dtype=='float32' else 2)
    if size>32768:raise ValueError('outer-product shared storage exceeds 32 KiB')
    rm,rn=tile_m//micro_m,tile_n//micro_n
    mr,nr=('tx // '+str(rn),'tx % '+str(rn)) if owner_axis=='column' else ('tx % '+str(rm),'tx // '+str(rm))
    idx=lambda row,k:f'[{k}, {row}]' if lhs_layout=='km' else f'[{row}, {k}]'
    lines=(['T.func_attr({"tensor.webgpu.loop_unroll":"explicit"})'] if explicit_unroll else [])
    lines += [f'with T.Kernel(T.ceildiv({columns}, {tile_n}), T.ceildiv({rows}, {tile_m}), threads={threads}) as (bx, by):',
           '    tx = T.get_thread_binding()',f'    mr = {mr}',f'    nr = {nr}',
           f'    lhs = T.alloc_shared({lhs_shape}, "{dtype}")',f'    rhs = T.alloc_shared({rhs_shape}, "{dtype}")']
    for i in range(micro_m):
        for j in range(micro_n):lines += [f'    acc{i}_{j} = T.alloc_var("float32")',f'    acc{i}_{j} = 0']
    lines += [f'    for tile in T.serial(T.ceildiv({depth}, {tile_k})):',
              f'        for load_a in T.serial(T.ceildiv({tile_m*tile_k}, {threads})):',
              f'            flat_a = load_a * {threads} + tx',
              f'            ar = flat_a // {tile_k}',f'            ak = flat_a % {tile_k}',
              f'            if ar < {tile_m}:',
              f'                lhs{idx("ar","ak")} = T.if_then_else((by * {tile_m} + ar < {rows}) & (tile * {tile_k} + ak < {depth}), T.cast({activation}, "{dtype}"), T.cast(0, "{dtype}"))',
              f'        for load_b in T.serial(T.ceildiv({tile_n*tile_k}, {threads})):',
              f'            flat_b = load_b * {threads} + tx',
              f'            br = flat_b // {tile_k}',f'            bk = flat_b % {tile_k}',
              f'            if br < {tile_n}:',
              f'                rhs[bk, br] = T.if_then_else((bx * {tile_n} + br < {columns}) & (tile * {tile_k} + bk < {depth}), w[(bx * {tile_n} + br) * {depth} + tile * {tile_k} + bk], T.cast(0, "{dtype}"))',
              '        T.sync_threads()',f'        for chunk in T.serial({tile_k//unroll}):',f'            for u in T.unroll({unroll}):',
              f'                kk = chunk * {unroll} + u']
    for i in range(micro_m):lines += [f'                left{i} = T.cast(lhs{idx(f"mr + {i*rm}","kk")}, "float32")']
    for j in range(micro_n):lines += [f'                right{j} = T.cast(rhs[kk, nr + {j*rn}], "float32")']
    for i in range(micro_m):
        for j in range(micro_n):
            value=f'T.call_extern("float32", "fma", left{i}, right{j}, acc{i}_{j})' if fma else f'acc{i}_{j} + left{i} * right{j}'
            lines += [f'                acc{i}_{j} = {value}']
    lines += ['        T.sync_threads()']
    for i in range(micro_m):
        for j in range(micro_n):
            row=f'by * {tile_m} + mr + {i*rm}';col=f'bx * {tile_n} + nr + {j*rn}'
            value=f'T.max(acc{i}_{j} + T.cast(bias[{col}], "float32"), 0)' if epilogue=='linear' else f'acc{i}_{j}'
            lines += [f'    if ({row} < {rows}) & ({col} < {columns}):',f'        out[({row}) * {columns} + {col}] = {value}']
    return '\n'.join(lines)


def partitioned_matmul_schedule(rows, depth, columns, lhs_value, rhs_value, *,
                                tile_m=8, tile_n=16, threads=128, partitions=8,
                                unroll=4, dot_width=1, owner_axis='column', k_layout='blocked'):
    """Direct-load SIMT microtiles with one workgroup-local K reduction.

    Load expressions use {row}, {column}, and {k} placeholders. The producer
    owns operand precision. Partitions never require a second dispatch or atomics.
    """
    dimensions=(rows,depth,columns,tile_m,tile_n,threads,partitions,unroll)
    if any(type(v) is not int or v<=0 for v in dimensions):
        raise ValueError('partitioned matmul dimensions must be positive integers')
    if partitions & (partitions-1) or threads%partitions or depth%(partitions*unroll):
        raise ValueError('invalid K partition or unroll')
    if dot_width not in (1,2,4) or unroll%dot_width:
        raise ValueError('invalid partitioned matmul dot width')
    if owner_axis not in ('row','column') or k_layout not in ('blocked','striped'):
        raise ValueError('invalid partitioned matmul distribution')
    owners=threads//partitions
    nc=min(tile_n,owners) if owner_axis=='column' else owners//min(tile_m,owners)
    if owners%nc or tile_m%(owners//nc) or tile_n%nc:
        raise ValueError('invalid partitioned output tile')
    rm=owners//nc; mm=tile_m//rm; nn=tile_n//nc
    if mm*nn>64 or tile_m*tile_n*partitions*4>32768:
        raise ValueError('partitioned matmul resource limit')
    lines=[f'with T.Kernel(T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n}), threads={threads}) as (by, bx):',
           '    tx = T.get_thread_binding()',f'    lane = tx % {partitions}',
           f'    mr = tx // {partitions*nc}',f'    nr = (tx // {partitions}) % {nc}',
           f'    scratch = T.alloc_shared(({tile_m*tile_n*partitions},), "float32")']
    for i in range(mm):
        for j in range(nn):lines += [f'    acc{i}_{j} = T.alloc_var("float32")',f'    acc{i}_{j} = 0']
    lines += [f'    for tile in T.serial({depth//(partitions*unroll)}):',
              f'        for u in T.unroll({unroll//dot_width}):']
    kval=lambda v:(f'tile * {partitions*unroll} + lane * {unroll} + u * {dot_width} + {v}' if k_layout=='blocked'
                   else f'tile * {partitions*unroll} + lane * {dot_width} + u * {partitions*dot_width} + {v}')
    for i in range(mm):
        row=f'by * {tile_m} + mr + {i*rm}'
        exprs=[f'T.if_then_else({row} < {rows}, {lhs_value.format(row=row,column="0",k=kval(v))}, 0)' for v in range(dot_width)]
        expr=exprs[0] if dot_width==1 else f'T.call_extern("float32x{dot_width}", "vec{dot_width}<f32>", '+', '.join(exprs)+')'
        lines += [f'            left{i} = {expr}']
    for j in range(nn):
        col=f'bx * {tile_n} + nr + {j*nc}'
        exprs=[f'T.if_then_else({col} < {columns}, {rhs_value.format(row="0",column=col,k=kval(v))}, 0)' for v in range(dot_width)]
        expr=exprs[0] if dot_width==1 else f'T.call_extern("float32x{dot_width}", "vec{dot_width}<f32>", '+', '.join(exprs)+')'
        lines += [f'            right{j} = {expr}']
    for i in range(mm):
        for j in range(nn):
            product=f'left{i} * right{j}' if dot_width==1 else f'T.call_extern("float32", "dot", left{i}, right{j})'
            lines += [f'            acc{i}_{j} = acc{i}_{j} + {product}']
    for i in range(mm):
        for j in range(nn):lines += [f'    scratch[((mr + {i*rm}) * {tile_n} + nr + {j*nc}) * {partitions} + lane] = acc{i}_{j}']
    lines += ['    T.sync_threads()', '    total = T.alloc_var("float32")',
              f'    for index in T.serial(T.ceildiv({tile_m*tile_n}, {threads})):',
              f'        slot = index * {threads} + tx',f'        if slot < {tile_m*tile_n}:',
              '            total = 0',f'            for part in T.unroll({partitions}):',
              f'                total = total + scratch[slot * {partitions} + part]',
              f'            if (by * {tile_m} + slot // {tile_n} < {rows}) & (bx * {tile_n} + slot % {tile_n} < {columns}):',
              f'                out[(by * {tile_m} + slot // {tile_n}) * {columns} + bx * {tile_n} + slot % {tile_n}] = total']
    return '\n'.join(lines)


def streamed_gemv_schedule(depth, columns, *, lanes=32, threads=128,
                          micro_rows=1, dot_width=4, unroll=1, accumulators=1,
                          k_layout='striped', shared_input=False):
    """F16-weight/F32-input GEMV with reusable input and independent K chains.

    Keep activation precision intact. Output tails are guarded and subgroup
    reductions have a shared-memory fallback for smaller physical subgroups.
    """
    if any(type(v) is not int or v<=0 for v in
           (depth,columns,lanes,threads,micro_rows,dot_width,unroll,accumulators)):
        raise ValueError('invalid GEMV dimensions')
    if lanes not in (8,16,32,64,128) or threads not in (64,128,256,512) or threads%lanes:
        raise ValueError('invalid GEMV distribution')
    if dot_width not in (1,2,4) or accumulators not in (1,2,4,8) or unroll%accumulators:
        raise ValueError('invalid GEMV arithmetic')
    if depth%(lanes*dot_width*unroll) or micro_rows>8 or micro_rows*accumulators>32:
        raise ValueError('invalid GEMV tile')
    if k_layout not in ('blocked','striped') or type(shared_input) is not bool:
        raise ValueError('invalid GEMV input layout')
    if (depth*4 if shared_input else 0)+micro_rows*threads*4>32768:
        raise ValueError('GEMV scratch exceeds portable limit')
    rows=threads//lanes;tile_rows=rows*micro_rows;step=lanes*dot_width*unroll
    lines=[f'with T.Kernel(T.ceildiv({columns}, {tile_rows}), threads={threads}) as bx:',
           '    tx = T.get_thread_binding()',f'    lane = tx % {lanes}',f'    row = tx // {lanes}',
           f'    scratch = T.alloc_shared(({micro_rows*threads},), "float32")']
    if shared_input:
        lines += [f'    lhs = T.alloc_shared(({depth},), "float32")',f'    for i in T.Parallel({depth}):',
                  '        lhs[i] = x[i]','    T.sync_threads()']
    for i in range(micro_rows):
        for j in range(accumulators):lines += [f'    acc{i}_{j} = T.alloc_var("float32")',f'    acc{i}_{j} = 0']
    lines += [f'    for tile in T.serial({depth//step}):']
    # Generate unroll slots explicitly to distribute independent accumulators.
    for u in range(unroll):
        kval=lambda v:(f'tile * {step} + lane * {unroll*dot_width} + {u*dot_width+v}' if k_layout=='blocked'
                       else f'tile * {step} + {u*lanes*dot_width} + lane * {dot_width} + {v}')
        lhs=[f'{"lhs" if shared_input else "x"}[{kval(v)}]' for v in range(dot_width)]
        vector=lambda terms:terms[0] if dot_width==1 else f'T.call_extern("float32x{dot_width}", "vec{dot_width}<f32>", '+', '.join(terms)+')'
        lines += [f'        left{u} = {vector(lhs)}']
        for i in range(micro_rows):
            output_row=f'bx * {tile_rows} + row + {i*rows}'
            rhs=[f'T.cast(w[({output_row}) * {depth} + {kval(v)}], "float32")' for v in range(dot_width)]
            lines += [f'        if {output_row} < {columns}:',f'            right{u} = {vector(rhs)}']
            prod=f'left{u} * right{u}' if dot_width==1 else f'T.call_extern("float32", "dot", left{u}, right{u})'
            lines += [f'            acc{i}_{u%accumulators} = acc{i}_{u%accumulators} + {prod}']
    for i in range(micro_rows):
        for j in range(1,accumulators):lines += [f'    acc{i}_0 = acc{i}_0 + acc{i}_{j}']
    lines += [f'    if T.call_extern("uint32", "tensor_subgroup_size") >= {lanes}:']
    for stride in (lanes>>i for i in range(1,lanes.bit_length())):
        for i in range(micro_rows):
            lines += [f'        acc{i}_0 = acc{i}_0 + T.call_extern("float32", "subgroupShuffleXor", acc{i}_0, T.uint32({stride}))']
    lines += ['    else:']
    for i in range(micro_rows):lines += [f'        scratch[{i*threads} + tx] = acc{i}_0']
    lines += ['        T.sync_threads()']
    for stride in (lanes>>i for i in range(1,lanes.bit_length())):
        lines += [f'        if lane < {stride}:']
        for i in range(micro_rows):
            lines += [f'            scratch[{i*threads} + tx] = scratch[{i*threads} + tx] + scratch[{i*threads+stride} + tx]']
        lines += ['        T.sync_threads()']
    for i in range(micro_rows):lines += [f'        acc{i}_0 = scratch[{i*threads} + tx]']
    for i in range(micro_rows):
        output_row=f'bx * {tile_rows} + row + {i*rows}'
        lines += [f'    if (lane == 0) & ({output_row} < {columns}):',f'        out[{output_row}] = acc{i}_0']
    return '\n'.join(lines)


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
                    buffers[buffer] = ir.decl_buffer(buffer.shape, buffer.dtype, buffer.name,
                                                    scope="shared", data_alignment=buffer.data_alignment)
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
                             node.match_buffers, remap_annotations(node.annotations))
        if isinstance(node, ir.For) and "num_stages" in node.annotations:
            annotations = {str(k): v for k, v in node.annotations.items() if str(k) != "num_stages"}
            return ir.For(node.loop_var, node.min, node.extent, node.kind, node.body,
                          node.thread_binding, annotations)
        return None

    kernel = kernel.with_body(ir.stmt_functor.ir_transform(kernel.body, None, materialize))

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
    ir.stmt_functor.post_order_visit(kernel.body, lambda node: bindings.update(
        {str(node.thread_binding.thread_tag): node})
        if isinstance(node, ir.For) and node.thread_binding is not None else None)
    thread = bindings.get("threadIdx.x")
    eligible_threads = thread is not None and isinstance(thread.extent, ir.IntImm) and int(thread.extent) > 0 and all(
        tag not in bindings or isinstance(bindings[tag].extent, ir.IntImm) and int(bindings[tag].extent) == 1
        for tag in ("threadIdx.y", "threadIdx.z"))
    fragments = set(buffers[b] for b in buffers if b.scope() == "local.fragment")

    def retain_accumulators(node):
        nonlocal counter
        if accumulation == "shared" or not eligible_threads or not isinstance(node, ir.For):
            return None
        if node.kind != ir.ForKind.SERIAL or node.thread_binding is not None:
            return None
        statements = list(node.body.seq) if isinstance(node.body, ir.SeqStmt) else [node.body]
        last = statements[-1]
        if not isinstance(last, ir.Evaluate) or not isinstance(last.value, ir.Call):
            return None
        call = last.value
        if str(call.op.name) != "tl.tileop.gemm":
            return None
        args = call.args
        if len(args) != 19 or any(not isinstance(args[index], ir.IntImm) for index in (3, 4, 5, 6, 7, 9, 14, 15, 16)):
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
        if any(x.buffer.scope() != "shared" or str(x.buffer.dtype) not in ("float16", "float32") for x in (a, b)):
            return None
        unsafe = []
        def inspect(part):
            if isinstance(part, (ir.BufferLoad, ir.BufferStore)) and part.buffer == c.buffer:
                unsafe.append(True)
            if isinstance(part, ir.Var) and part.same_as(c.buffer.data):
                unsafe.append(True)
            if isinstance(part, ir.Call) and str(part.op.name) in ("tl.tileop.gemm", "tl.tileop.reduce"):
                unsafe.append(True)
            if isinstance(part, ir.SBlock) and (part.alloc_buffers or part.init is not None):
                unsafe.append(True)
        for statement in statements[:-1]:
            ir.stmt_functor.post_order_visit(statement, inspect)
        for index in (*a.indices, *b.indices, *c.indices):
            ir.stmt_functor.post_order_visit(index, lambda part: unsafe.append(True)
                if isinstance(part, ir.Var) and part.same_as(node.loop_var) else None)
        if unsafe:
            return None
        m, n, depth = (int(args[index]) for index in (5, 6, 7))
        if min(m, n, depth) <= 0:
            return None
        # Long reductions recover the cost of a larger private live set. The
        # RX 6700 XT ablation also found repeatable shorter-K regressions, so
        # leave those on the existing schedule unless a producer/tuner opts in.
        if accumulation == "auto" and (not isinstance(node.extent, ir.IntImm) or int(node.extent) * depth < 2048):
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
        local = ir.decl_buffer((groups * tm * tn,), "float32", f"wgpu_gemm_loop_acc_{counter}", scope="local")
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
                    lhs = ir.Cast("float32", ir.BufferLoad(a.buffer, [x+y for x, y in zip(a.indices, aa)]))
                    rhs = ir.Cast("float32", ir.BufferLoad(b.buffer, [x+y for x, y in zip(b.indices, bb)]))
                    acc = ir.BufferLoad(local, [slot])
                    starts.append(ir.BufferStore(local, ir.BufferLoad(c.buffer, indices), [slot]))
                    dots.append(ir.BufferStore(local, acc + lhs * rhs, [slot]))
                    stores.append(ir.BufferStore(c.buffer, acc, indices))
            initialize.append(group_statement(starts, tid))
            updates.append(group_statement(dots, tid))
            finish.append(group_statement(stores, tid))
        dot = ir.For(k, 0, depth, ir.ForKind.SERIAL, updates[0] if len(updates) == 1 else ir.SeqStmt(updates))
        loop_body = ir.SeqStmt([*statements[:-1], barrier(), dot, barrier()])
        loop = ir.For(node.loop_var, node.min, node.extent, node.kind, loop_body,
                      node.thread_binding, node.annotations)
        body = ir.SeqStmt([barrier(), *initialize, loop, *finish, barrier()])
        block = ir.SBlock([], [], [], f"wgpu_gemm_loop_{counter}", body, alloc_buffers=[local])
        return ir.SBlockRealize([], True, block)

    kernel = kernel.with_body(ir.stmt_functor.ir_transform(kernel.body, None, retain_accumulators, ["tirx.For"]))

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
            rank = len(a.indices)
            if rank not in (1, 2) or len(c.indices) != 1 or not 0 <= axis < rank or str(operation.value) not in ("max", "sum"):
                raise ValueError("WebGPU reduction supports sum/max of one/two-dimensional tiles along one axis")
            counter += 1
            row, reduction = [ir.Var(f"wgpu_{name}_{counter}", "int32") for name in ("row", "reduce")]
            source_index = [reduction] if rank == 1 else [row, reduction] if axis == 1 else [reduction, row]
            value = ir.BufferLoad(a.buffer, [x+y for x,y in zip(a.indices, source_index)])
            index = [c.indices[0]+row]
            is_max = str(operation.value) == "max"
            initial = ir.reinterpret("float32", ir.const(0xff800000, "uint32")) if is_max else ir.const(0, str(c.buffer.dtype))
            initial = ir.if_then_else(clear, initial, ir.BufferLoad(c.buffer, index))
            body = local_accumulator(f"wgpu_reduce_acc_{counter}", initial, reduction, source.args[axis+2],
                                     lambda acc: ir.max(acc, value) if is_max else acc+value, c.buffer, index)
            rows = 1 if rank == 1 else source.args[3-axis]
            if reduction_mode == "tree":
                extent = int(source.args[axis+2])
                count = int(rows)
                padded = 1 << (extent-1).bit_length()
                scratch = ir.decl_buffer((count, padded), "float32", f"wgpu_tree_{counter}", scope="shared")
                identity = ir.reinterpret("float32", ir.const(0xff800000, "uint32")) if is_max else ir.const(0, "float32")
                stage = ir.BufferStore(scratch, ir.if_then_else(reduction < extent, value, identity), [row, reduction])
                steps = [barrier(), ir.For(row, 0, count, ir.ForKind.PARALLEL,
                         ir.For(reduction, 0, padded, ir.ForKind.PARALLEL, stage)), barrier()]
                stride = padded//2
                while stride:
                    lhs = ir.BufferLoad(scratch, [row, reduction])
                    rhs = ir.BufferLoad(scratch, [row, reduction+stride])
                    update = ir.BufferStore(scratch, ir.max(lhs,rhs) if is_max else lhs+rhs, [row,reduction])
                    rr,kk=ir.Var(f"wgpu_tree_row_{counter}_{stride}","int32"),ir.Var(f"wgpu_tree_lane_{counter}_{stride}","int32")
                    update=ir.stmt_functor.substitute(update,{row:rr,reduction:kk})
                    steps.extend([ir.For(rr, 0, count, ir.ForKind.PARALLEL,
                                  ir.For(kk, 0, stride, ir.ForKind.PARALLEL, update)), barrier()])
                    stride //= 2
                result = ir.BufferLoad(scratch,[row,0])
                previous = ir.BufferLoad(c.buffer,index)
                result = ir.if_then_else(clear,result,ir.max(previous,result) if is_max else previous+result)
                rr=ir.Var(f"wgpu_tree_output_{counter}","int32")
                store=ir.stmt_functor.substitute(ir.BufferStore(c.buffer,result,index),{row:rr})
                steps.append(ir.For(rr,0,count,ir.ForKind.PARALLEL,store))
                steps.append(barrier())
                block = ir.SBlock([],[],[],f"wgpu_tree_reduce_{counter}",ir.SeqStmt(steps),alloc_buffers=[scratch])
                return ir.SBlockRealize([],True,block)
            return ir.SeqStmt([barrier(), ir.For(row, 0, rows, ir.ForKind.PARALLEL, body), barrier()])
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
        micro = int(kernel.attrs.get("tensor.webgpu.gemm_microtile", 2))
        if micro not in (1, 2, 4):
            raise ValueError("WebGPU GEMM microtile must be 1, 2 or 4")
        tm, tn = (micro if extent % micro == 0 else 1 for extent in (m, n))
        local = ir.decl_buffer((tm*tn,), "float32", f"wgpu_gemm_acc_{counter}", scope="local")
        initialize, updates, finish = [], [], []
        for row_offset in range(tm):
            for col_offset in range(tn):
                slot = row_offset*tn+col_offset
                ii, jj = i*tm+row_offset, j*tn+col_offset
                indices = [c.indices[0]+ii,c.indices[1]+jj]
                initial = ir.if_then_else(args[9],ir.const(0,"float32"),ir.BufferLoad(c.buffer,indices))
                aa = [k,ii] if int(args[3]) else [ii,k]
                bb = [jj,k] if int(args[4]) else [k,jj]
                lhs = ir.Cast("float32",ir.BufferLoad(a.buffer,[x+y for x,y in zip(a.indices,aa)]))
                rhs = ir.Cast("float32",ir.BufferLoad(b.buffer,[x+y for x,y in zip(b.indices,bb)]))
                acc = ir.BufferLoad(local,[slot])
                initialize.append(ir.BufferStore(local,initial,[slot]))
                updates.append(ir.BufferStore(local,acc+lhs*rhs,[slot]))
                finish.append(ir.BufferStore(c.buffer,acc,indices))
        inner=updates[0] if len(updates)==1 else ir.SeqStmt(updates)
        body = ir.SeqStmt([*initialize,ir.For(k,0,depth,ir.ForKind.SERIAL,inner),*finish])
        block = ir.SBlock([],[],[],f"wgpu_gemm_register_tile_{counter}",body,alloc_buffers=[local])
        dot = ir.SBlockRealize([],True,block)
        return ir.SeqStmt([barrier(), ir.For(i, 0, m//tm, ir.ForKind.PARALLEL,
                      ir.For(j, 0, n//tn, ir.ForKind.PARALLEL, dot)), barrier()])

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
