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
        "tensor_dot2_f16_add": '''fn tensor_dot2_f16_add(a:u32, b:u32, accumulator:f32)->f32 {
  return accumulator + dot(unpack2x16float(a), unpack2x16float(b));
}''',
        "tensor_fma2_f16_vec": '''fn tensor_fma2_f16_vec(a:u32, b:u32, accumulator:vec2<f16>)->vec2<f16> {
  return fma(vec2<f16>(unpack2x16float(a)), vec2<f16>(unpack2x16float(b)), accumulator);
}''',
        "tensor_sum2_f16_vec": '''fn tensor_sum2_f16_vec(a:vec2<f16>)->f32 {
  return f32(a.x) + f32(a.y);
}''',
    }
    for name, helper in helpers.items():
        if name+'(' in source:
            source += '\n'+helper+'\n'
    if 'tensor_fma2_f16_vec(' in source and 'enable f16;' not in source:
        source='enable f16;\n'+source
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
                                  lhs_value=None, rhs_value=None, rhs_transform=None,
                                  dot_width=1,packed_pairs=False,half_accum=False,group_order='column'):
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
    if type(dot_width) is not int or dot_width not in (1,2,4) or tile_k%(unroll*dot_width):
        raise ValueError('invalid outer-product dot width')
    if type(packed_pairs) is not bool or (packed_pairs and (dtype!='float16' or dot_width!=2 or lhs_layout!='km')):
        raise ValueError('packed outer-product pairs require F16, K-major layout and dot width two')
    if type(half_accum) is not bool or (half_accum and not packed_pairs):
        raise ValueError('half accumulation requires packed F16 pairs')
    if group_order not in ('row','column'):raise ValueError('invalid outer-product group order')
    if lhs_layout not in ('mk','km') or owner_axis not in ('row','column') or dtype not in ('float32','float16') or epilogue not in ('gemm','linear') or type(fma) is not bool or type(explicit_unroll) is not bool:
        raise ValueError('invalid outer-product layout or arithmetic')
    if any(type(v) is not int or v<0 for v in (lhs_pad,rhs_pad)):
        raise ValueError('invalid outer-product padding')
    if lhs_value is not None and (type(lhs_value) is not str or not lhs_value.strip()):
        raise ValueError('invalid outer-product activation expression')
    if any(v is not None and (type(v) is not str or not v.strip()) for v in (rhs_value,rhs_transform)):
        raise ValueError('invalid outer-product weight expression')
    if packed_pairs:
        return packed_outer_product_schedule(rows,depth,columns,tile_m=tile_m,tile_n=tile_n,
            tile_k=tile_k,micro_m=micro_m,micro_n=micro_n,threads=threads,lhs_pad=lhs_pad,
            rhs_pad=rhs_pad,owner_axis=owner_axis,unroll=unroll,explicit_unroll=explicit_unroll,
            epilogue=epilogue,lhs_value=lhs_value,rhs_value=rhs_value,rhs_transform=rhs_transform,half_accum=half_accum,group_order=group_order)
    activation=(lhs_value or f'x[({{row}}) * {depth} + ({{k}})]').format(
        row=f'by * {tile_m} + ar',k=f'tile * {tile_k} + ak')
    weight=(rhs_value or f'w[({{column}}) * {depth} + ({{k}})]').format(
        column=f'bx * {tile_n} + br',row='0',k=f'tile * {tile_k} + bk')
    lhs_shape=(tile_k,tile_m+lhs_pad) if lhs_layout=='km' else (tile_m,tile_k+lhs_pad)
    rhs_shape=(tile_k,tile_n+rhs_pad)
    size=(lhs_shape[0]*lhs_shape[1]+rhs_shape[0]*rhs_shape[1])*(4 if dtype=='float32' else 2)
    if size>32768:raise ValueError('outer-product shared storage exceeds 32 KiB')
    rm,rn=tile_m//micro_m,tile_n//micro_n
    mr,nr=('tx // '+str(rn),'tx % '+str(rn)) if owner_axis=='column' else ('tx % '+str(rm),'tx // '+str(rm))
    idx=lambda row,k:f'[{k}, {row}]' if lhs_layout=='km' else f'[{row}, {k}]'
    lines=(['T.func_attr({"tensor.webgpu.loop_unroll":"explicit"})'] if explicit_unroll else [])
    grid=(f'T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n})' if group_order=='row' else
          f'T.ceildiv({columns}, {tile_n}), T.ceildiv({rows}, {tile_m})')
    axes='by, bx' if group_order=='row' else 'bx, by'
    lines += [f'with T.Kernel({grid}, threads={threads}) as ({axes}):',
           '    tx = T.get_thread_binding()',f'    mr = {mr}',f'    nr = {nr}',
           f'    lhs = T.alloc_shared({lhs_shape}, "{dtype}")',f'    rhs = T.alloc_shared({rhs_shape}, "{dtype}")']
    if rhs_transform is not None:lines += ['    value = T.alloc_var("float32")']
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
              f'            if br < {tile_n}:']
    if rhs_transform is None:
        operand=weight if rhs_value is None else f'T.cast({weight}, "{dtype}")'
        lines += [f'                rhs[bk, br] = T.if_then_else((bx * {tile_n} + br < {columns}) & (tile * {tile_k} + bk < {depth}), {operand}, T.cast(0, "{dtype}"))']
    else:
        lines += [f'                value = T.if_then_else(bx * {tile_n} + br < {columns}, T.if_then_else(tile * {tile_k} + bk < {depth}, {weight}, 0), 0)',
                  f'                rhs[bk, br] = T.cast({rhs_transform.format(value="value")}, "{dtype}")']
    lines += [
              '        T.sync_threads()',f'        for chunk in T.serial({tile_k//(unroll*dot_width)}):',f'            for u in T.unroll({unroll}):',
              f'                kk = chunk * {unroll} + u' if dot_width==1 else f'                kk = (chunk * {unroll} + u) * {dot_width}']
    def vector(values):
        return values[0] if dot_width==1 else f'T.call_extern("float32x{dot_width}", "vec{dot_width}<f32>", '+', '.join(values)+')'
    for i in range(micro_m):
        values=[f'T.cast(lhs{idx(f"mr + {i*rm}","kk" if dot_width==1 else f"kk + {lane}")}, "float32")' for lane in range(dot_width)]
        lines += [f'                left{i} = {vector(values)}']
    for j in range(micro_n):
        values=[f'T.cast(rhs[{"kk" if dot_width==1 else f"kk + {lane}"}, nr + {j*rn}], "float32")' for lane in range(dot_width)]
        lines += [f'                right{j} = {vector(values)}']
    for i in range(micro_m):
        for j in range(micro_n):
            value=(f'acc{i}_{j} + T.call_extern("float32", "dot", left{i}, right{j})' if dot_width>1 else
                   f'T.call_extern("float32", "fma", left{i}, right{j}, acc{i}_{j})' if fma else f'acc{i}_{j} + left{i} * right{j}')
            lines += [f'                acc{i}_{j} = {value}']
    lines += ['        T.sync_threads()']
    for i in range(micro_m):
        for j in range(micro_n):
            row=f'by * {tile_m} + mr + {i*rm}';col=f'bx * {tile_n} + nr + {j*rn}'
            value=f'T.max(acc{i}_{j} + T.cast(bias[{col}], "float32"), 0)' if epilogue=='linear' else f'acc{i}_{j}'
            lines += [f'    if ({row} < {rows}) & ({col} < {columns}):',f'        out[({row}) * {columns} + {col}] = {value}']
    return '\n'.join(lines)


def packed_outer_product_schedule(rows,depth,columns,*,tile_m,tile_n,tile_k,
                                  micro_m,micro_n,threads,lhs_pad,rhs_pad,owner_axis,
                                  unroll,explicit_unroll,epilogue,lhs_value,rhs_value,rhs_transform,half_accum=False,group_order='column'):
    """Adjacent K halves in shared u32; optional short private F16 FMA chains.

    Even/odd K lanes round independently in F16 for ``unroll`` terms before
    contributing to the complete-K F32 total. This changes accumulation
    precision and requires a producer's model-level acceptance, not just timing.
    """
    pk=tile_k//2;rm,rn=tile_m//micro_m,tile_n//micro_n
    if (pk*(tile_m+lhs_pad+tile_n+rhs_pad)*4)>32768:
        raise ValueError('packed outer-product storage exceeds 32 KiB')
    mr,nr=(f'tx // {rn}',f'tx % {rn}') if owner_axis=='column' else (f'tx % {rm}',f'tx // {rm}')
    lines=(['T.func_attr({"tensor.webgpu.loop_unroll":"explicit"})'] if explicit_unroll else [])
    grid=(f'T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n})' if group_order=='row' else
          f'T.ceildiv({columns}, {tile_n}), T.ceildiv({rows}, {tile_m})')
    axes='by, bx' if group_order=='row' else 'bx, by'
    lines += [f'with T.Kernel({grid}, threads={threads}) as ({axes}):',
              '    tx = T.get_thread_binding()',f'    mr = {mr}',f'    nr = {nr}']
    for name,width in (('lhs',tile_m+lhs_pad),('rhs',tile_n+rhs_pad)):
        lines += [f'    {name} = T.alloc_shared(({pk}, {width}), "uint32")']
    lines += ['    value0 = T.alloc_var("float32")','    value1 = T.alloc_var("float32")']
    for i in range(micro_m):
        for j in range(micro_n):
            lines += [f'    acc{i}_{j} = T.alloc_var("float32")',f'    acc{i}_{j} = 0']
            if half_accum:lines += [f'    partial{i}_{j} = T.alloc_var("float16x2")']
    lines += [f'    for tile in T.serial(T.ceildiv({depth}, {tile_k})):',
              f'        for load_a in T.serial(T.ceildiv({tile_m*pk}, {threads})):',
              f'            flat = load_a * {threads} + tx',f'            ar = flat // {pk}',f'            pair = flat % {pk}',
              f'            if ar < {tile_m}:']
    av=[]
    for lane in range(2):
        row=f'by * {tile_m} + ar';k=f'tile * {tile_k} + pair * 2 + {lane}'
        expr=(lhs_value or f'x[({{row}}) * {depth} + ({{k}})]').format(row=row,k=k)
        av.append(f'T.if_then_else(({row} < {rows}) & ({k} < {depth}), T.cast({expr}, "float32"), 0)')
    lines += [f'                lhs[pair, ar] = T.call_extern("uint32", "pack2x16float", T.call_extern("float32x2", "vec2<f32>", {av[0]}, {av[1]}))']
    lines += [f'        for load_b in T.serial(T.ceildiv({tile_n*pk}, {threads})):',
              f'            flat = load_b * {threads} + tx',f'            br = flat // {pk}',f'            pair = flat % {pk}',
              f'            if br < {tile_n}:']
    for lane in range(2):
        column=f'bx * {tile_n} + br';k=f'tile * {tile_k} + pair * 2 + {lane}'
        expr=(rhs_value or f'w[({{column}}) * {depth} + ({{k}})]').format(column=column,k=k,row='0')
        lines += [f'                value{lane} = T.if_then_else(bx * {tile_n} + br < {columns}, T.if_then_else({k} < {depth}, T.cast({expr}, "float32"), 0), 0)']
    bv=[rhs_transform.format(value=f'value{lane}') if rhs_transform else f'value{lane}' for lane in range(2)]
    lines += [f'                rhs[pair, br] = T.call_extern("uint32", "pack2x16float", T.call_extern("float32x2", "vec2<f32>", {bv[0]}, {bv[1]}))']
    lines += ['        T.sync_threads()',f'        for chunk in T.serial({pk//unroll}):']
    if half_accum:
        for i in range(micro_m):
            for j in range(micro_n):lines += [f'            partial{i}_{j} = T.call_extern("float16x2", "vec2<f16>", T.cast(0, "float16"))']
    lines += [f'            for u in T.unroll({unroll}):',f'                kk = chunk * {unroll} + u']
    for name,width,stride,count in (('left','lhs',rm,micro_m),('right','rhs',rn,micro_n)):
        index='mr' if name=='left' else 'nr'
        for i in range(count):
            expr=f'{width}[kk, {index} + {i*stride}]'
            lines += [f'                {name}{i} = {expr}']
    for i in range(micro_m):
        for j in range(micro_n):
            if half_accum:lines += [f'                partial{i}_{j} = T.call_extern("float16x2", "tensor_fma2_f16_vec", left{i}, right{j}, partial{i}_{j})']
            else:lines += [f'                acc{i}_{j} = T.call_extern("float32", "tensor_dot2_f16_add", left{i}, right{j}, acc{i}_{j})']
    if half_accum:
        for i in range(micro_m):
            for j in range(micro_n):lines += [f'            acc{i}_{j} = acc{i}_{j} + T.call_extern("float32", "tensor_sum2_f16_vec", partial{i}_{j})']
    lines += ['        T.sync_threads()']
    for i in range(micro_m):
        for j in range(micro_n):
            row=f'by * {tile_m} + mr + {i*rm}';col=f'bx * {tile_n} + nr + {j*rn}'
            value=f'T.max(acc{i}_{j} + T.cast(bias[{col}], "float32"), 0)' if epilogue=='linear' else f'acc{i}_{j}'
            lines += [f'    if ({row} < {rows}) & ({col} < {columns}):',f'        out[({row}) * {columns} + {col}] = {value}']
    return '\n'.join(lines)


def packed_integer_matmul_schedule(rows, depth, columns, rhs_words, rhs_scales, *,
                                  tile_m=16, tile_n=32, micro_m=2, micro_n=2,
                                  threads=128, owner_axis='column', group_order='row',tile_k=32,
                                  fixed_residual=False,signed_rhs=False):
    """Two-component signed-byte activations, unsigned nibble RHS, F32 sums.

    Producers provide packed RHS word/scale expressions with {column}, {block}
    and {word} placeholders. One or two RHS matrices select linear/SwiGLU output.
    Input planes use packed[2*M*K/4], scales/sums[2*M*K/32]. No GPU imports.
    """
    dims=(rows,depth,columns,tile_m,tile_n,micro_m,micro_n,threads,tile_k)
    if any(type(v) is not int or v<=0 for v in dims) or tile_k%32 or depth%tile_k:
        raise ValueError('invalid packed integer matmul dimensions')
    if threads not in (64,128,256,512) or tile_m%micro_m or tile_n%micro_n or (tile_m//micro_m)*(tile_n//micro_n)!=threads:
        raise ValueError('invalid packed integer matmul ownership')
    parts=len(rhs_words)
    if parts not in (1,2) or len(rhs_scales)!=parts or any(type(v) is not str or not v.strip() for v in (*rhs_words,*rhs_scales)):
        raise ValueError('invalid packed integer matmul RHS expressions')
    if micro_m*micro_n>32 or owner_axis not in ('column','row') or group_order not in ('column','row'):
        raise ValueError('invalid packed integer matmul layout')
    if type(fixed_residual) is not bool:raise ValueError('invalid fixed residual mode')
    if type(signed_rhs) is not bool:raise ValueError('invalid signed RHS mode')
    blocks=tile_k//32
    storage=(tile_m*80+parts*tile_n*36)*blocks
    if storage>32768:raise ValueError('packed integer matmul exceeds 32 KiB')
    rm,rn=tile_m//micro_m,tile_n//micro_n
    mr,nr=(f'tx // {rn}',f'tx % {rn}') if owner_axis=='column' else (f'tx % {rm}',f'tx // {rm}')
    grid=(f'T.ceildiv({rows}, {tile_m}), T.ceildiv({columns}, {tile_n})', '(by, bx)') if group_order=='row' else (f'T.ceildiv({columns}, {tile_n}), T.ceildiv({rows}, {tile_m})','(bx, by)')
    lines=['T.func_attr({"tensor.webgpu.loop_unroll":"explicit"})',
           f'with T.Kernel({grid[0]}, threads={threads}) as {grid[1]}:',
           '    tx = T.get_thread_binding()',f'    mr = {mr}',f'    nr = {nr}',
           f'    lhs = T.alloc_shared((2, {tile_m}, {8*blocks}), "uint32")',
           f'    factors = T.alloc_shared((2, {tile_m}, {blocks}), "float32")',
           f'    offsets = T.alloc_shared((2, {tile_m}, {blocks}), "int32")',
           f'    rhs = T.alloc_shared(({parts}, {8*blocks}, {tile_n}), "uint32")',
           f'    weight_scale = T.alloc_shared(({parts}, {blocks}, {tile_n}), "float32")']
    for part in range(parts):
        for i in range(micro_m):
            for j in range(micro_n):
                lines += [f'    acc{part}_{i}_{j} = T.alloc_var("float32")',f'    acc{part}_{i}_{j} = 0']
                for component in range(2):lines += [f'    dot{part}_{component}_{i}_{j} = T.alloc_var("int32")']
    lines += [f'    for tile in T.serial({depth//tile_k}):',
              f'        for load_a in T.serial(T.ceildiv({tile_m*8*blocks}, {threads})):',
              f'            flat = load_a * {threads} + tx',f'            ar = flat // {8*blocks}',f'            word = flat % {8*blocks}',
              f'            if ar < {tile_m}:']
    for component in range(2):
        row=f'by * {tile_m} + ar';index=f'({row}) * {depth//4} + tile * {8*blocks} + word + {component*rows*depth//4}'
        scaleindex=f'({row}) * {depth//32} + tile * {blocks} + word // 8 + {component*rows*depth//32}'
        lines += [f'                lhs[{component}, ar, word] = T.if_then_else({row} < {rows}, packed[{index}], T.uint32(0))',
                  '                if word % 8 == 0:',f'                    factors[{component}, ar, word // 8] = T.if_then_else({row} < {rows}, scales[{scaleindex}], 0)',
                  f'                    offsets[{component}, ar, word // 8] = T.if_then_else({row} < {rows}, sums[{scaleindex}], 0)']
    rhs_words_per_block=8 if signed_rhs else 4
    lines += [f'        for load_b in T.serial(T.ceildiv({tile_n*rhs_words_per_block*blocks}, {threads})):',
              f'            flat = load_b * {threads} + tx',f'            br = flat // {rhs_words_per_block*blocks}',f'            word = flat % {rhs_words_per_block*blocks}',
              f'            if br < {tile_n}:']
    for part in range(parts):
        column=f'bx * {tile_n} + br'
        values=dict(column=column,block=f'tile * {blocks} + word // {rhs_words_per_block}',word=f'word % {rhs_words_per_block}')
        lines += [f'                bits{part} = T.if_then_else({column} < {columns}, {rhs_words[part].format(**values)}, T.uint32(0))']
        if signed_rhs:lines += [f'                rhs[{part}, word, br] = bits{part}']
        else:lines += [f'                rhs[{part}, (word // 4) * 8 + word % 4, br] = bits{part} & T.uint32(252645135)',
                      f'                rhs[{part}, (word // 4) * 8 + word % 4 + 4, br] = (bits{part} >> 4) & T.uint32(252645135)']
        lines += [f'                if word % {rhs_words_per_block} == 0:',f'                    weight_scale[{part}, word // {rhs_words_per_block}, br] = T.if_then_else({column} < {columns}, {rhs_scales[part].format(**values)}, 0)']
    lines += ['        T.sync_threads()',f'        for block in T.serial({blocks}):']
    for part in range(parts):
        for component in range(2):
            for i in range(micro_m):
                for j in range(micro_n):lines += [f'            dot{part}_{component}_{i}_{j} = 0']
    lines += ['            for word in T.unroll(8):']
    for component in range(2):
        for i in range(micro_m):lines += [f'                left{component}_{i} = lhs[{component}, mr + {i*rm}, block * 8 + word]']
    for part in range(parts):
        for j in range(micro_n):lines += [f'                right{part}_{j} = rhs[{part}, block * 8 + word, nr + {j*rn}]']
        for component in range(2):
            for i in range(micro_m):
                for j in range(micro_n):lines += [f'                dot{part}_{component}_{i}_{j} = dot{part}_{component}_{i}_{j} + T.call_extern("int32", "dot4I8Packed", left{component}_{i}, right{part}_{j})']
    for part in range(parts):
        for i in range(micro_m):
            for j in range(micro_n):
                row=f'mr + {i*rm}';col=f'nr + {j*rn}'
                values=[f'T.cast(dot{part}_{component}_{i}_{j}'+('' if signed_rhs else f' - 8 * offsets[{component}, {row}, block]')+f', "float32") * factors[{component}, {row}, block]' for component in range(2)]
                if fixed_residual:
                    combined=(f'dot{part}_0_{i}_{j} * 254 + dot{part}_1_{i}_{j}' if signed_rhs else
                              f'(dot{part}_0_{i}_{j} - 8 * offsets[0, {row}, block]) * 254 + dot{part}_1_{i}_{j} - 8 * offsets[1, {row}, block]')
                    lines += [f'            acc{part}_{i}_{j} = acc{part}_{i}_{j} + T.cast({combined}, "float32") * factors[1, {row}, block] * weight_scale[{part}, block, {col}]']
                else:
                    lines += [f'            acc{part}_{i}_{j} = acc{part}_{i}_{j} + ({values[0]} + {values[1]}) * weight_scale[{part}, block, {col}]']
    lines += ['        T.sync_threads()']
    for i in range(micro_m):
        for j in range(micro_n):
            row=f'by * {tile_m} + mr + {i*rm}';col=f'bx * {tile_n} + nr + {j*rn}';value=f'acc0_{i}_{j}'
            if parts==2:value=f'({value}) / (1 + T.exp(-({value}))) * acc1_{i}_{j}'
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
