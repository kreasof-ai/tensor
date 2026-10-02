"""Packed LFM2 kernels with portable and optional standard subgroup schedules."""
import math
import re
from .kernels import emit, source as cuda_source, weight
from .gguf import TYPES


def half_bits(bits):
    return f'T.call_extern("float32", "tensor_unpack_f16", T.cast({bits}, "uint32"))'


def packed_byte(address):
    return f'((w[({address}) // 4] >> (({address}) % 4 * 8)) & T.uint32(255))'


def packed_half(address):
    return half_bits(f'((w[({address}) // 4] >> (({address}) % 4 * 8)) & T.uint32(65535))')


def packed_word(address):
    # GGML's 18-byte Q4 blocks alternate between two- and four-byte alignment.
    return f'T.call_extern("uint32", "tensor_byte_align_u32", w[({address}) // 4], w[(({address}) + 3) // 4], T.cast(({address}) % 4, "uint32"))'


def segmented_reduce(lanes):
    shuffle='\n'.join(f'        accum = accum + T.call_extern("float32", "subgroupShuffleXor", accum, T.uint32({stride}))' for stride in (lanes>>i for i in range(1,lanes.bit_length())))
    fallback='\n'.join(f'''        if lane < {stride}:
            scratch[tx] = scratch[tx] + scratch[tx + {stride}]
        T.sync_threads()''' for stride in (lanes>>i for i in range(1,lanes.bit_length())))
    return f'''    if T.call_extern("uint32", "tensor_subgroup_size") >= {lanes}:
{shuffle}
        scratch[tx] = accum
    else:
        scratch[tx] = accum
        T.sync_threads()
{fallback}'''


def tree_reduce(size,operation='sum'):
    lines=[]
    for stride in (size>>i for i in range(1,size.bit_length())):
        left,right='scratch[tx]',f'scratch[tx + {stride}]'
        expression=f'T.max({left}, {right})' if operation=='max' else f'{left} + {right}'
        lines.extend((f'    if tx < {stride}:',f'        scratch[tx] = {expression}','    T.sync_threads()'))
    return '\n'.join(lines)


def subgroup_reduce(size,value,operation='sum'):
    builtin='subgroupMax' if operation=='max' else 'subgroupAdd'
    initial='-T.infinity("float32")' if operation=='max' else '0'
    update=f'T.max({value}, scratch[i])' if operation=='max' else f'{value} + scratch[i]'
    return f'''    scratch[tx] = T.call_extern("float32", "{builtin}", {value})
    T.sync_threads()
    if tx == 0:
        {value} = {initial}
        for group in T.serial(T.ceildiv({size}, T.cast(T.call_extern("uint32", "tensor_subgroup_size"), "int32"))):
            i = group * T.cast(T.call_extern("uint32", "tensor_subgroup_size"), "int32")
            {value} = {update}
        scratch[0] = {value}
    T.sync_threads()'''


def rms_width(c):
    """Threads for the single-workgroup decode reduction over c elements.

    Decode normalisation reduces a whole row inside one workgroup, so a
    64-thread wave occupies one of forty compute units and walks c/64
    dependent iterations twice, once to accumulate and once to rescale. A full
    256-thread CU quarter-wave cuts that chain by four at the same dispatch
    count. c must still divide evenly, exactly as the previous fixed width
    required, and add_rms rewrites this kernel's source so it shares the value.
    """
    for width in (256, 128, 64):
        if c % width == 0:
            return width
    return 64


def round_half(value):
    """Round FP32 to an exactly representable FP16 value before native casting.

    Native f16 conversion need not match NumPy's ties-to-even rounding. Integer
    rounding also handles FP16 subnormals, retaining the stated prefill contract.

    The exponent range tests compare a signed cast. Lowering the equivalent
    unsigned compare emits `exponent / 113u < 1u`, and an unsigned divide costs
    far more than the branch it replaces. Every staged prefill operand passes
    through this routine, so those two divisions dominated the staging loop.
    """
    bits=f'T.reinterpret("uint32", {value})'
    exponent=f'(({bits} >> 23) & 255)'
    signed=f'T.cast({exponent}, "int32")'
    shift=f'T.min(T.max(126 - {signed}, 1), 24)'
    mantissa=f'(({bits} & 8388607) | 8388608)'
    rounded=f'(({mantissa} + (T.uint32(1) << ({shift} - 1)) - 1 + (({mantissa} >> {shift}) & 1)) >> {shift})'
    quantum=f'T.reinterpret("float32", T.uint32({103<<23}))'
    small=f'T.if_then_else({signed} < 102, 0.0, T.cast({rounded}, "float32") * {quantum}) * T.if_then_else(({bits} >> 31) != 0, -1.0, 1.0)'
    normal=f'T.reinterpret("float32", ({bits} + 4095 + (({bits} >> 13) & 1)) & T.uint32(4294959104))'
    return f'T.if_then_else({signed} < 113, {small}, {normal})'


def source(kind,p):
    r=p.get('r',1)
    a=lambda name,n,dtype='float32':(name,n,dtype)
    if kind=='quantize_q8':
        k=p['k']
        packed=' | '.join(f'((T.reinterpret("uint32", quants[tx * 4 + {i}]) & 255) << {i*8})' for i in range(4))
        return emit([a('x',k),a('out',k//4,'uint32'),a('scales',k//32),a('sums',k//32,'int32')],f'''with T.Kernel({k//32},threads=32) as block:
    tx=T.get_thread_binding()
    scratch=T.alloc_shared((32,),"float32")
    quants=T.alloc_shared((32,),"int32")
    scale=T.alloc_var("float32")
    scratch[tx]=T.abs(x[block * 32 + tx])
    T.sync_threads()
{tree_reduce(32,'max')}
    scale=T.max(scratch[0] / 127, 1.0e-20)
    quants[tx]=T.cast(T.round(x[block * 32 + tx] / scale),"int32")
    T.sync_threads()
    if tx < 8:
        out[block * 8 + tx]={packed}
    scratch[tx]=T.cast(quants[tx],"float32")
    T.sync_threads()
{tree_reduce(32)}
    if tx == 0:
        scales[block]=scale
        sums[block]=T.cast(scratch[0],"int32")''')
    if kind=='linear_q8':
        k,o=p['k'],p['o'];lanes=16;row_count=8
        base=f'((bx * 8 + row) * {k//32} + tile * 4 + lane // 4) * 18'
        block='tile * 4 + lane // 4'
        return emit([a('x',k//4,'uint32'),a('scales',k//32),a('sums',k//32,'int32'),a('w',k*o//32*18//4,'uint32'),a('out',o)],f'''with T.Kernel(T.ceildiv({o},8),threads=128) as bx:
    tx=T.get_thread_binding()
    row=tx // 16
    lane=tx % 16
    accum=T.alloc_var("float32")
    scratch=T.alloc_shared((128,),"float32")
    accum=0
    if bx * 8 + row < {o}:
        for tile in T.serial({k//128}):
            packed={packed_word(base+' + 2 + lane % 4 * 4')}
            low=T.call_extern("int32","dot4I8Packed",packed & T.uint32(252645135),x[({block}) * 8 + lane % 4])
            high=T.call_extern("int32","dot4I8Packed",(packed >> 4) & T.uint32(252645135),x[({block}) * 8 + lane % 4 + 4])
            accum=accum + T.cast(low + high - 2 * sums[{block}],"float32") * scales[{block}] * ({packed_half(base)})
{segmented_reduce(lanes)}
    if (lane == 0) & (bx * 8 + row < {o}):
        out[bx * 8 + row]=scratch[tx]''')
    if kind=='ffn':
        # Reuse the projection layouts/decoders for two matrices. The pair
        # shares activation loads, synchronization and the SwiGLU epilogue.
        text=source('linear',p);q=p['type'];k,o=p['k'],p['o'];_,block,size=TYPES[q]
        count=k*o if q in (0,1) else k*o//block*size//4
        dtype='float16' if q==1 else 'float32' if q==0 else 'uint32'
        text=text.replace(', out: T.Tensor',f', w2: T.Tensor(({count},), "{dtype}"), out: T.Tensor')
        def paired(line):
            def replace(match):
                name=match[0]
                if name.startswith('acc'):return name.replace('acc','up',1)
                return name+'2'
            return re.sub(r'\b(?:accum|acc\d+|scale|packed|low\d+|high\d+|scratch|rhs|right\d+|value|w)\b',replace,line)
        lines=[]
        for line in text.splitlines():
            if 'out[' in line and ' = ' in line:
                destination,gate=line.split(' = ',1);up=paired(gate)
                lines.append(destination+f' = ({gate}) / (1 + T.exp(-({gate}))) * ({up})')
                continue
            lines.append(line)
            stripped=line.strip()
            if (' = ' in line and (re.match(r'(accum|acc\d+|scratch(?:\[.*?\])?|rhs(?:\[.*?\])?|right\d+|scale|packed|low\d+|high\d+) = ',stripped)
                                      or stripped.startswith('value = T.if_then_else(bx'))):
                lines.append(paired(line))
            elif stripped=='value = T.alloc_var("float32")':lines.append(paired(line))
        return '\n'.join(lines)+'\n'
    if kind=='linear_add':
        if r!=1:raise ValueError('residual projection fusion requires decode rows')
        text=source('linear',p)
        text=text.replace(', out: T.Tensor',f', residual: T.Tensor(({p["o"]},), "float32"), out: T.Tensor')
        lines=[]
        for line in text.splitlines():
            if 'out[' in line and ' = ' in line:
                destination,value=line.split(' = ',1)
                index=destination.strip()[4:-1]
                line=destination+f' = residual[{index}] + ({value})'
            lines.append(line)
        return '\n'.join(lines)+'\n'
    if kind=='add_rms':
        c=p['c'];text=source('rms',p)
        text=text.replace(', w: T.Tensor',f', mixed: T.Tensor(({r*c},), "float32"), residual: T.Tensor(({r*c},), "float32"), w: T.Tensor')
        text=re.sub(r'x\[([^\]]+)\]',lambda m:f'(x[{m[1]}] + mixed[{m[1]}])',text)
        index=f'row * {c} + i * {rms_width(c)} + tx'
        text=text.replace(f'            out[{index}]',f'            residual[{index}] = x[{index}] + mixed[{index}]\n            out[{index}]')
        return text
    if kind=='argmax':
        n=p['n'];threads=256
        reduce='\n'.join(f'''    if tx < {stride}:
        if (values[tx + {stride}] > values[tx]) | ((values[tx + {stride}] == values[tx]) & (indices[tx + {stride}] < indices[tx])):
            values[tx] = values[tx + {stride}]
            indices[tx] = indices[tx + {stride}]
    T.sync_threads()''' for stride in (128,64,32,16,8,4,2,1))
        return emit([a('logits',n),a('token',1,'int32'),a('pos',2,'int32')],f'''with T.Kernel(1, threads={threads}):
    tx = T.get_thread_binding()
    best = T.alloc_var("float32")
    index = T.alloc_var("int32")
    values = T.alloc_shared(({threads},), "float32")
    indices = T.alloc_shared(({threads},), "int32")
    best = -T.infinity("float32")
    index = {n}
    for tile in T.serial(T.ceildiv({n}, {threads})):
        i = tile * {threads} + tx
        if i < {n}:
            if (logits[i] > best) | ((logits[i] == best) & (i < index)):
                best = logits[i]
                index = i
    values[tx] = best
    indices[tx] = index
    T.sync_threads()
{reduce}
    if tx == 0:
        token[0] = indices[0]
        pos[1] = 1''')
    if kind=='rms' and r==1:
        c=p['c'];threads=rms_width(c)
        reduce=subgroup_reduce(threads,'total') if p.get('sg') else '    scratch[tx] = total\n    T.sync_threads()\n'+tree_reduce(threads)
        return emit([a('x',r*c),a('w',c),a('out',r*c)],f'''with T.Kernel({r}, threads={threads}) as row:
    tx = T.get_thread_binding()
    total = T.alloc_var("float32")
    scratch = T.alloc_shared(({threads},), "float32")
    total = 0
    for i in T.serial({c//threads}):
        total = total + x[row * {c} + i * {threads} + tx] * x[row * {c} + i * {threads} + tx]
{reduce}
    for i in T.serial({c//threads}):
        out[row * {c} + i * {threads} + tx] = x[row * {c} + i * {threads} + tx] * T.rsqrt(scratch[0] / {c} + {p['eps']}) * w[i * {threads} + tx]''')
    if kind in ('linear','embedding'):
        text=cuda_source(kind,p)
        q=p['type']
        if kind=='linear' and r==1:
            k,o=p['k'],p['o'];_,block,size=TYPES[q]
            count=k*o if q in (0,1) else k*o//block*size//4
            dtype='float16' if q==1 else 'float32' if q==0 else 'uint32'
            threads=128;lanes=16 if q==2 else 32
            accumulators=1
            if q==2 and k%128==0:
                # The larger streamed matrices benefit from shorter K chains.
                # Narrow shapes retain their measured 16-lane schedule.
                wide=p.get('sg') and k%256==0 and min(k,o)>=2048
                lanes=p.get('gemv_lanes',32 if wide else 16);threads=p.get('gemv_threads',128)
                accumulators=p.get('gemv_accumulators',1)
                if lanes not in (8,16,32) or threads not in (64,128,256) or accumulators not in (1,4) or k%(lanes*8):
                    raise ValueError('unsupported packed Q4 decode schedule')
            row_count=threads//lanes
            expr=weight(q,f'bx * {row_count} + row','tile * 32 + lane',k,packed_words=True)
            if q==2 and k%128==0:
                tile_width=lanes*8
                base=f'((bx * {row_count} + row) * {k//32} + tile * {lanes//4} + lane // 4) * 18'
                terms=[]
                for byte in range(4):
                    low=f'((packed >> {byte*8}) & 15)'
                    high=f'((packed >> {byte*8+4}) & 15)'
                    index=f'tile * {tile_width} + lane // 4 * 32 + lane % 4 * 4 + {byte}'
                    acc='accum' if accumulators==1 else f'acc{byte}'
                    terms.extend((f'            {acc} = {acc} + x[{index}] * (scale * (T.cast({low}, "float32") - 8))',
                                  f'            {acc} = {acc} + x[{index} + 16] * (scale * (T.cast({high}, "float32") - 8))'))
                loop=f'''        for tile in T.serial({k//tile_width}):
            scale = {packed_half(base)}
            packed = {packed_word(base+' + 2 + lane % 4 * 4')}
'''+ '\n'.join(terms)
                if p.get('gemv_dot',wide):
                    if accumulators!=1:raise ValueError('packed dot schedule uses one accumulator')
                    loads=[]
                    index=f'tile * {tile_width} + lane // 4 * 32 + lane % 4 * 4'
                    for half in range(2):
                        lhs=', '.join(f'x[{index} + {half*16+byte}]' for byte in range(4))
                        rhs=', '.join(f'T.cast((packed >> {byte*8+half*4}) & 15, "float32") - 8' for byte in range(4))
                        loads.extend((f'            left{half} = T.call_extern("float32x4", "vec4<f32>", {lhs})',
                                      f'            right{half} = T.call_extern("float32x4", "vec4<f32>", {rhs})',
                                      f'            accum = accum + scale * T.call_extern("float32", "dot", left{half}, right{half})'))
                    loop=f'''        for tile in T.serial({k//tile_width}):
            scale = {packed_half(base)}
            packed = {packed_word(base+' + 2 + lane % 4 * 4')}
'''+ '\n'.join(loads)
            elif q==14:
                base=f'((bx * 4 + row) * {k//256} + tile) * 210'
                reads='\n'.join(f'            low{i} = {packed_byte(base+f" + {i*32} + lane")}' for i in range(4))
                reads+='\n'+'\n'.join(f'            high{i} = {packed_byte(base+f" + {128+i*32} + lane")}' for i in range(2))
                terms=[]
                for group in range(8):
                    byte=f'low{group//4*2+group%2}'
                    value=f'(({byte} >> {4*(group%4//2)}) & 15) | ((high{group//4} >> {2*(group%4)} & 3) << 4)'
                    scale=packed_byte(base+f' + {192+group*2} + lane // 16')
                    signed=f'T.cast(T.cast({scale}, "int32") - T.if_then_else({scale} >= 128, 256, 0), "float32")'
                    terms.append(f'            accum = accum + x[tile * 256 + {group*32} + lane] * (scale * {signed} * (T.cast({value}, "float32") - 32))')
                loop=f'''        for tile in T.serial({k//256}):
            scale = {packed_half(base+' + 208')}
{reads}
'''+ '\n'.join(terms)
            elif q in (0,1) and k%128==0:
                index='tile * 128 + lane * 4'
                lhs=', '.join(f'x[{index} + {i}]' for i in range(4))
                rhs=', '.join(f'T.cast(w[(bx * 4 + row) * {k} + {index} + {i}], "float32")' for i in range(4))
                loop=f'''        for tile in T.serial({k//128}):
            accum = accum + T.call_extern("float32", "dot", T.call_extern("float32x4", "vec4<f32>", {lhs}), T.call_extern("float32x4", "vec4<f32>", {rhs}))'''
            else:
                if q==2:expr=weight(q,f'bx * {row_count} + row','tile * 32 + lane',k,packed_words=True)
                if q==2:
                    base=f'((bx * {row_count} + row) * {k//32} + tile) * 18'
                    loop=f'''        for tile in T.serial({k//32}):
            scale = {packed_half(base)}
            packed = {packed_byte(base+' + 2 + lane')}
            accum = accum + x[tile * 32 + lane] * (scale * (T.cast(packed & 15, "float32") - 8))
            accum = accum + x[tile * 32 + lane + 16] * (scale * (T.cast(packed >> 4, "float32") - 8))'''
                else:
                    loop=f'''        for tile in T.serial({k//32}):
            accum = accum + x[tile * 32 + lane] * ({expr})'''
            reduction='\n'.join(f'''    if lane < {stride}:
        scratch[tx] = scratch[tx] + scratch[tx + {stride}]
    T.sync_threads()''' for stride in (lanes>>i for i in range(1,lanes.bit_length())))
            reduction = segmented_reduce(lanes) if p.get('sg') else '    scratch[tx] = accum\n    T.sync_threads()\n'+reduction
            extra='' if accumulators==1 else '\n'+'\n'.join(f'    acc{i} = T.alloc_var("float32")\n    acc{i} = 0' for i in range(4))
            combine='' if accumulators==1 else '\n    accum = (acc0 + acc1) + (acc2 + acc3)'
            return emit([a('x',k),a('w',count,dtype),a('out',o)],f'''with T.Kernel(T.ceildiv({o}, {row_count}), threads={threads}) as bx:
    tx = T.get_thread_binding()
    row = tx // {lanes}
    lane = tx % {lanes}
    accum = T.alloc_var("float32")
    scratch = T.alloc_shared(({threads},), "float32")
    accum = 0{extra}
    if bx * {row_count} + row < {o}:
{loop}{combine}
{reduction}
    if (lane == 0) & (bx * {row_count} + row < {o}):
        out[bx * {row_count} + row] = scratch[tx]''')
        if kind=='linear' and r>1:
            k,o=p['k'],p['o'];_,block,size=TYPES[q]
            count=k*o if q in (0,1) else k*o//block*size//4
            dtype='float16' if q==1 else 'float32' if q==0 else 'uint32'
            args=[a('x',r*k),a('w',count,dtype),a('out',r*o)]
            from tensor.compiler.webgpu_lowering import register_matmul_schedule
            tm,tn,bk=p.get('tile',(16,32,64))
            expr=weight(q,f'bx * {tn} + i',f'tile * {bk} + j',k,packed_words=True)
            value=round_half('value') if q!=1 else 'value'
            rhs=f'T.if_then_else(bx * {tn} + i < {o}, {expr}, 0)'
            body=register_matmul_schedule(r,k,o,round_half('value'),rhs,tile_m=tm,tile_n=tn,tile_k=bk,pad=p.get('pad',0),
                lhs_pad=p.get('lhs_pad',0),lhs_transpose=p.get('lhs_transpose',False),
                dot_width=p.get('dot_width',4 if q==1 or k>=2048 else 1),unroll=p.get('unroll',False))
            body=body.replace('rhs[j, i] = value',f'rhs[j, i] = {value}')
            return emit(args,body)
        if q in (0,1):return text
        _,block,size=TYPES[q]
        k=p['k'] if kind=='linear' else p['c']
        o=p['o'] if kind=='linear' else p['v']
        count=k*o//block*size
        if count%4:raise ValueError('WebGPU packed matrices must align to four bytes')
        text=text.replace(f'w: T.Tensor(({count},), "uint8")',f'w: T.Tensor(({count//4},), "uint32")')
        coordinates=[('bx * 4 + row','tile * 32 + lane'),('bx * 64 + i','tile * 32 + j')] if kind=='linear' else [('tokens[row]','col')]
        for row,col in coordinates:
            text=text.replace(weight(q,row,col,k),weight(q,row,col,k,packed_words=True))
        return text
    if kind in ('qnorm','kvnorm'):
        h,kh,d,cap=p['h'],p['kh'],p['d'],p['cap']
        query=kind=='qnorm';heads=h if query else kh
        args=([a('q',r*h*d),a('qw',d),a('qo',r*h*d),a('pos',2,'int32')] if query else
              [a('k',r*kh*d),a('v',r*kh*d),a('kw',d),a('kc',cap*kh*d,'float16'),a('vc',cap*kh*d,'float16'),a('pos',2,'int32')])
        x,w,out=('q','qw','qo') if query else ('k','kw','kc')
        target=f'(row * {heads} + head) * {d}' if query else f'((pos[0] + row) * {heads} + head) * {d}'
        store=f'''for i in T.Parallel({d//2}):
    angle = T.cast(pos[0] + row, "float32") * T.exp(-{math.log(p['theta'])*2/d} * i)
    {out}[{target} + i] = norm[i] * T.cos(angle) - norm[i + {d//2}] * T.sin(angle)
    {out}[{target} + i + {d//2}] = norm[i] * T.sin(angle) + norm[i + {d//2}] * T.cos(angle)'''
        if not query:
            for expression in (f'norm[i] * T.cos(angle) - norm[i + {d//2}] * T.sin(angle)',
                               f'norm[i] * T.sin(angle) + norm[i + {d//2}] * T.cos(angle)'):
                store=store.replace(expression,round_half('('+expression+')'))
            store+='\n'+f'''for i in T.Parallel({d}):
    vc[{target} + i] = {round_half(f'v[(row * {heads} + head) * {d} + i]')}'''
            store='if row < pos[1]:\n'+'\n'.join('    '+line for line in store.splitlines())
        return emit(args,f'''with T.Kernel({heads}, {r}, threads=64) as (head, row):
    square = T.alloc_fragment(({d},), "float32")
    total = T.alloc_fragment((1,), "float32")
    norm = T.alloc_shared(({d},), "float32")
    for i in T.Parallel({d}):
        square[i] = {x}[(row * {heads} + head) * {d} + i] * {x}[(row * {heads} + head) * {d} + i]
    T.reduce_sum(square, total, dim=0)
    for i in T.Parallel({d}):
        norm[i] = {x}[(row * {heads} + head) * {d} + i] * T.rsqrt(total[0] / {d} + {p['eps']}) * {w}[i]
'''+ '\n'.join('    '+line for line in store.splitlines()))
    if kind=='attention_scores':
        h,kh,d,cap=p['h'],p['kh'],p['d'],p['cap']
        shuffles='\n'.join(f'            dot = dot + T.call_extern("float32", "subgroupShuffleXor", dot, T.uint32({stride}))' for stride in (16,8,4,2,1))
        return emit([a('q',h*d),a('kc',cap*kh*d,'float16'),a('out',h*cap),a('pos',2,'int32')],f'''with T.Kernel({h}, T.ceildiv({cap},32), threads=128) as (head,block):
    tx = T.get_thread_binding()
    dot = T.alloc_var("float32")
    if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
        lane = tx % 32
        for tile in T.serial(8):
            token = block * 32 + tile * 4 + tx // 32
            dot = 0
            if token <= pos[0]:
                for j in T.serial({d//32}):
                    channel = j * 32 + lane
                    dot = dot + q[head * {d} + channel] * T.cast(kc[(token * {kh} + head // {h//kh}) * {d} + channel], "float32")
{shuffles}
            if (lane == 0) & (token < {cap}):
                out[head * {cap} + token] = T.if_then_else(token <= pos[0],dot * {d**-0.5},-T.infinity("float32"))
    else:
        if tx < 32:
            token = block * 32 + tx
            if token < {cap}:
                dot = -T.infinity("float32")
                if token <= pos[0]:
                    dot = 0
                    for j in T.serial({d}):
                        dot = dot + q[head * {d} + j] * T.cast(kc[(token * {kh} + head // {h//kh}) * {d} + j], "float32")
                    dot = dot * {d**-0.5}
                out[head * {cap} + token] = dot''')
    if kind=='attention':
        h,kh,d,cap=p['h'],p['kh'],p['d'],p['cap'];threads=128
        maximum_reduce=subgroup_reduce(threads,'maximum','max') if p.get('sg') else '    scratch[tx] = maximum\n    T.sync_threads()\n'+tree_reduce(threads,'max')
        sum_reduce=subgroup_reduce(threads,'total') if p.get('sg') else '    scratch[tx] = total\n    T.sync_threads()\n'+tree_reduce(threads)
        score_loop=f'''    for tile in T.serial(T.ceildiv({cap}, {threads})):
        token = tile * {threads} + tx
        if token < {cap}:
            dot = -T.infinity("float32")
            if token <= pos[0] + row:
                dot = 0
                for j in T.serial({d}):
                    dot = dot + q[(row * {h} + head) * {d} + j] * T.cast(kc[(token * {kh} + head // {h//kh}) * {d} + j], "float32")
                dot = dot * {d**-0.5}
            scores[token] = dot
            maximum = T.max(maximum, dot)'''
        if p.get('sg'):
            shuffles='\n'.join(f'            dot = dot + T.call_extern("float32", "subgroupShuffleXor", dot, T.uint32({stride}))' for stride in (16,8,4,2,1))
            fallback='\n'.join('    '+line for line in score_loop.splitlines())
            score_loop=f'''    for token in T.Parallel({cap}):
        scores[token] = -T.infinity("float32")
    T.sync_threads()
    if T.call_extern("uint32", "tensor_subgroup_size") >= 32:
        lane = tx % 32
        for tile in T.serial(T.ceildiv(pos[0] + row + 1, 4)):
            token = tile * 4 + tx // 32
            dot = 0
            if token <= pos[0] + row:
                for j in T.serial({d//32}):
                    channel = j * 32 + lane
                    dot = dot + q[(row * {h} + head) * {d} + channel] * T.cast(kc[(token * {kh} + head // {h//kh}) * {d} + channel], "float32")
{shuffles}
            if (lane == 0) & (token <= pos[0] + row):
                scores[token] = dot * {d**-0.5}
    else:
{fallback}
    T.sync_threads()
    for tile in T.serial(T.ceildiv({cap}, {threads})):
        token = tile * {threads} + tx
        if token < {cap}:
            maximum = T.max(maximum, scores[token])'''
        split = p.get('sg') and r==1
        if split:
            score_loop=f'''    for tile in T.serial(T.ceildiv({cap}, {threads})):
        token = tile * {threads} + tx
        if token < {cap}:
            scores[token] = q[head * {cap} + token]
            maximum = T.max(maximum,scores[token])'''
        arguments=[a('q',h*cap if split else r*h*d),*([] if split else [a('kc',cap*kh*d,'float16')]),a('vc',cap*kh*d,'float16'),a('out',r*h*d),a('pos',2,'int32')]
        text=emit(arguments,f'''with T.Kernel({h}, {r}, threads={threads}) as (head, row):
    tx = T.get_thread_binding()
    dot = T.alloc_var("float32")
    maximum = T.alloc_var("float32")
    total = T.alloc_var("float32")
    result = T.alloc_var("float32")
    scores = T.alloc_shared(({cap},), "float32")
    scratch = T.alloc_shared(({threads},), "float32")
    maximum = -T.infinity("float32")
{score_loop}
{maximum_reduce}
    total = 0
    for tile in T.serial(T.ceildiv({cap}, {threads})):
        token = tile * {threads} + tx
        if token < {cap}:
            scores[token] = T.exp(scores[token] - scratch[0])
            total = total + scores[token]
    T.sync_threads()
{sum_reduce}
    result = 0
    if tx < {d}:
        for token in T.serial({cap}):
            if token <= pos[0] + row:
                result = result + scores[token] * T.cast(vc[(token * {kh} + head // {h//kh}) * {d} + tx], "float32")
        out[(row * {h} + head) * {d} + tx] = result / scratch[0]''')
        if p.get('sg'):
            text=text.replace(f'T.ceildiv({cap}, {threads})','T.ceildiv(pos[0] + row + 1, '+str(threads)+')')
            text=text.replace(f'for token in T.serial({cap}):','for token in T.serial(pos[0] + row + 1):')
            text=text.replace('            if token <= pos[0] + row:\n                result =','            result =')
        return text
    if kind=='qkv':raise ValueError('WebGPU uses separate query/cache normalization kernels')
    return cuda_source(kind,p)
