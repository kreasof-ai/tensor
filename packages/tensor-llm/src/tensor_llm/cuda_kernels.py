"""Inspectable TileLang/TIRx CUDA algorithms for packed LFM2 inference.

Measured schedules belong to producer profiles. All model arithmetic and
control flow stay in IR; the backend lowers only typed scalar/vector operations.
Runtime imports remain compiler-free.
"""
import re
import textwrap
from .kernels import emit, weight, source as baseline_source
from .gguf import TYPES

CUDA_PROFILES = ('default', 'optimized')
CUDA_GROUPED_THRESHOLD = 4096

def _u16(buffer, offset):
    return f'T.call_extern("uint32", "tensor_load_u16", T.address_of({buffer}[{offset}]))'


def _word(buffer, offset):
    return f'({_u16(buffer,offset)} | ({_u16(buffer,f"({offset}) + 2")} << 16))'


def _half(buffer, offset):
    return f'T.cast(T.reinterpret("float16", T.cast({_u16(buffer,offset)}, "uint16")), "float32")'


def _vector(buffer, index, name, dtype='float32', width=4):
    return f'''{name} = T.alloc_local(({width},), "{dtype}")
for component in T.vectorized({width}):
    {name}[component] = {buffer}[({index}) + component]'''


def _dot(q,k,row_bytes,buffer,width=4):
    """Packed GGML arithmetic expressed completely in frontend scalar/vector IR."""
    if q in (0,1):
        return f'''j = tile * {32*width} + lane * {width}
{_vector('x','j','activation',width=width)}
{_vector(buffer,f'row * {k} + j','coefficient','float16' if q==1 else 'float32',width)}
dot = T.alloc_var("float32")
dot = 0
for component in T.unroll({width}):
    dot = T.ieee_fmaf(activation[component], T.cast(coefficient[component], "float32"), dot)'''
    if q==2:
        body=f'''base = row * {row_bytes} + (tile * 8 + lane // 4) * 18
scale = {_half(buffer,'base')}
packed = {_word(buffer,'base + 2 + lane % 4 * 4')}
j = tile * 256 + lane // 4 * 32 + lane % 4 * 4
{_vector('x','j','activation')}
{_vector('x','j + 16','activation_hi')}
dot = T.alloc_var("float32")
dot = 0
for component in T.unroll(4):
    dot = T.ieee_fmaf(activation[component], T.cast((packed >> (component * 8)) & 15, "float32") - 8, dot)
for component in T.unroll(4):
    dot = T.ieee_fmaf(activation_hi[component], T.cast((packed >> (component * 8 + 4)) & 15, "float32") - 8, dot)
dot = scale * dot'''
        return body
    if q==12:
        return f'''base = row * {row_bytes} + tile * 144
group = lane // 8 * 2
packed = {_word(buffer,'base + 16 + lane // 8 * 32 + lane % 8 * 4')}
j = tile * 256 + lane // 8 * 64 + lane % 8 * 4
{_vector('x','j','activation')}
{_vector('x','j + 32','activation_hi')}
lo = T.alloc_var("float32")
hi = T.alloc_var("float32")
lo = 0
hi = 0
for component in T.unroll(4):
    lo = T.ieee_fmaf(activation[component], T.cast((packed >> (component * 8)) & 15, "float32"), lo)
    hi = T.ieee_fmaf(activation_hi[component], T.cast((packed >> (component * 8 + 4)) & 15, "float32"), hi)
s0 = T.if_then_else(group < 4, {buffer}[base + 4 + group] & 63, ({buffer}[base + 8 + group] & 15) | (({buffer}[base + group] >> 6) << 4))
s1 = T.if_then_else(group < 4, {buffer}[base + 5 + group] & 63, ({buffer}[base + 9 + group] & 15) | (({buffer}[base + 1 + group] >> 6) << 4))
m0 = T.if_then_else(group < 4, {buffer}[base + 8 + group] & 63, ({buffer}[base + 8 + group] >> 4) | (({buffer}[base + 4 + group] >> 6) << 4))
m1 = T.if_then_else(group < 4, {buffer}[base + 9 + group] & 63, ({buffer}[base + 9 + group] >> 4) | (({buffer}[base + 5 + group] >> 6) << 4))
sum0 = (activation[0] + activation[1]) + (activation[2] + activation[3])
sum1 = (activation_hi[0] + activation_hi[1]) + (activation_hi[2] + activation_hi[3])
dot = {_half(buffer,'base')} * (T.cast(s0, "float32") * lo + T.cast(s1, "float32") * hi) - {_half(buffer,'base + 2')} * (T.cast(m0, "float32") * sum0 + T.cast(m1, "float32") * sum1)'''
    if q==14:
        return f'''base = row * {row_bytes} + tile * 210
dot = T.alloc_var("float32")
dot = 0
for half in T.unroll(2):
    low = {_word(buffer,'base + half * 64 + (lane // 8 % 2) * 32 + lane % 8 * 4')}
    high = {_word(buffer,'base + 128 + half * 32 + lane % 8 * 4')}
    {_vector('x','tile * 256 + half * 128 + lane * 4','activation').replace(chr(10),chr(10)+'    ')}
    value = T.alloc_var("float32")
    value = 0
    for component in T.unroll(4):
        quant = T.cast((low >> (component * 8 + lane // 16 * 4)) & 15, "int32") | T.cast(((high >> (component * 8 + lane // 8 * 2)) & 3) << 4, "int32")
        value = T.ieee_fmaf(activation[component], T.cast(quant - 32, "float32"), value)
    scale = T.cast(T.reinterpret("int8", {buffer}[base + 192 + half * 8 + lane // 4]), "float32")
    dot = T.ieee_fmaf(scale, value, dot)
dot = {_half(buffer,'base + 208')} * dot'''
    raise ValueError('unsupported packed dot encoding')


def gemv_source(kind,p):
    k,o,q=p['k'],p['o'],p['type']
    if q not in (0,1,2,12,14) or k%256:return fused_source(kind,p)
    threads,unroll,width=p.get('threads',128),p.get('unroll',4),p.get('f16_values',4)
    if threads not in (64,128,256) or unroll not in (1,2,4,8) or width not in (4,8,16) or k%(32*width):
        raise ValueError('unsupported CUDA packed GEMV schedule')
    _,block,size=TYPES[q]
    row_bytes=k*(4 if q==0 else 2) if q in (0,1) else k//block*size
    tiles=k//(32*width) if q in (0,1) else k//256
    paired,residual=kind=='ffn',kind=='linear_add'
    args=[('x',k,'float32'),('w',k*o if q in (0,1) else row_bytes*o,'float16' if q==1 else 'float32' if q==0 else 'uint8')]
    if paired:args.append(('w2',args[1][1],args[1][2]))
    if residual:args.append(('residual',o,'float32'))
    args.append(('out',o,'float32'))
    declarations='\n    ups = T.alloc_local(('+str(unroll)+',), "float32")' if paired else ''
    initialize='\n        ups[slot] = 0' if paired else ''
    compute='\n'.join('                    '+line for line in _dot(q,k,row_bytes,'w',width).splitlines())+'\n                    sums[slot] = sums[slot] + dot'
    if paired:compute+='\n'+'\n'.join('                    '+line for line in _dot(q,k,row_bytes,'w2',width).splitlines())+'\n                    ups[slot] = ups[slot] + dot'
    total='\n    up = T.alloc_var("float32")\n    up = 0' if paired else ''
    add='\n        up = up + ups[slot]' if paired else ''
    reduce='\n'.join(f'    total = total + T.shfl_down(total, {delta})'+(f'\n    up = up + T.shfl_down(up, {delta})' if paired else '') for delta in (16,8,4,2,1))
    value='total / (1 + T.exp(-total)) * up' if paired else 'residual[row] + total' if residual else 'total'
    return emit(args,f'''with T.Kernel(T.ceildiv({o}, {threads//32}), threads={threads}) as bx:
    tx = T.get_thread_binding()
    lane = tx % 32
    row = bx * {threads//32} + tx // 32
    sums = T.alloc_local(({unroll},), "float32"){declarations}
    for slot in T.unroll({unroll}):
        sums[slot] = 0{initialize}
    if row < {o}:
        for chunk in T.serial(T.ceildiv({tiles}, {unroll})):
            for slot in T.unroll({unroll}):
                tile = chunk * {unroll} + slot
                if tile < {tiles}:
{compute}
    total = T.alloc_var("float32")
    total = 0{total}
    for slot in T.unroll({unroll}):
        total = total + sums[slot]{add}
{reduce}
    if (lane == 0) & (row < {o}):
        out[row] = {value}''')


def warp_partial_source(p,splits):
    h,kh,d,cap=(p[n] for n in ('h','kh','d','cap'))
    if d!=64 or type(splits) is not int or splits<1:raise ValueError('unsupported warp attention schedule')
    shuffle='\n'.join(f'        score = score + T.shfl_down(score, {delta})' for delta in (16,8,4,2,1))
    return emit([('q',h*d,'float32'),('kc',cap*kh*d,'float16'),('vc',cap*kh*d,'float16'),('parts',h*splits*66,'float32'),('pos',2,'int32')],f'''with T.Kernel({h}, {splits}, threads=128) as (head, split):
    tx = T.get_thread_binding()
    lane = tx % 32
    warp = tx // 32
    chunk = T.ceildiv(pos[0] + 1, {splits})
    begin = split * chunk
    end = T.min(begin + chunk, pos[0] + 1)
    kvhead = head // {h//kh}
    q0 = q[head * 64 + lane]
    q1 = q[head * 64 + lane + 32]
    maxima = T.alloc_shared((4,), "float32")
    sums = T.alloc_shared((4,), "float32")
    values = T.alloc_shared((4, 64), "float32")
    maximum = T.alloc_var("float32")
    normalizer = T.alloc_var("float32")
    a0 = T.alloc_var("float32")
    a1 = T.alloc_var("float32")
    maximum = -T.infinity("float32")
    normalizer = 0
    a0 = 0
    a1 = 0
    for offset in T.serial(T.ceildiv(T.max(end - begin - warp, 0), 4)):
        token = begin + warp + offset * 4
        base = (token * {kh} + kvhead) * 64
        score = q0 * T.cast(kc[base + lane], "float32") + q1 * T.cast(kc[base + lane + 32], "float32")
{shuffle}
        score = T.shfl_sync(score, 0) * 0.125
        next_max = T.max(maximum, score)
        correction = T.exp(maximum - next_max)
        probability = T.exp(score - next_max)
        a0 = a0 * correction + probability * T.cast(vc[base + lane], "float32")
        a1 = a1 * correction + probability * T.cast(vc[base + lane + 32], "float32")
        normalizer = normalizer * correction + probability
        maximum = next_max
    values[warp, lane] = a0
    values[warp, lane + 32] = a1
    if lane == 0:
        maxima[warp] = maximum
        sums[warp] = normalizer
    T.sync_threads()
    if warp == 0:
        merged_max = T.max(T.max(maxima[0], maxima[1]), T.max(maxima[2], maxima[3]))
        total = T.alloc_var("float32")
        result0 = T.alloc_var("float32")
        result1 = T.alloc_var("float32")
        total = 0
        result0 = 0
        result1 = 0
        for other in T.unroll(4):
            factor = T.if_then_else(T.isfinite(merged_max), T.exp(maxima[other] - merged_max), 0)
            total = total + sums[other] * factor
            result0 = result0 + values[other, lane] * factor
            result1 = result1 + values[other, lane + 32] * factor
        base = (head * {splits} + split) * 66
        parts[base + lane] = result0
        parts[base + lane + 32] = result1
        if lane == 0:
            parts[base + 64] = merged_max
            parts[base + 65] = total''')


def grouped_source(p,stage=32,warps=2):
    h,kh,d,cap,splits=(p[n] for n in ('h','kh','d','cap','splits'))
    group=h//kh;threads=group*warps*32
    if d!=64 or h%kh or group not in (2,4) or stage not in (16,32,64) or warps not in (2,4):
        raise ValueError('unsupported grouped attention schedule')
    shuffle='\n'.join(f'            score = score + T.shfl_down(score, {delta})' for delta in (16,8,4,2,1))
    return emit([('q',h*d,'float32'),('kc',cap*kh*d,'float16'),('vc',cap*kh*d,'float16'),('parts',h*splits*66,'float32'),('pos',2,'int32')],f'''with T.Kernel({kh}, {splits}, threads={threads}) as (kvhead, split):
    tx = T.get_thread_binding()
    lane = tx % 32
    warp = tx // 32
    local = warp // {warps}
    worker = warp % {warps}
    head = kvhead * {group} + local
    chunk = T.ceildiv(pos[0] + 1, {splits})
    begin = split * chunk
    end = T.min(begin + chunk, pos[0] + 1)
    keys = T.alloc_shared(({stage}, 64), "float16")
    vals = T.alloc_shared(({stage}, 64), "float16")
    maxima = T.alloc_shared(({group*warps},), "float32")
    sums = T.alloc_shared(({group*warps},), "float32")
    values = T.alloc_shared(({group*warps}, 64), "float32")
    q0 = q[head * 64 + lane]
    q1 = q[head * 64 + lane + 32]
    maximum = T.alloc_var("float32")
    normalizer = T.alloc_var("float32")
    a0 = T.alloc_var("float32")
    a1 = T.alloc_var("float32")
    maximum = -T.infinity("float32")
    normalizer = 0
    a0 = 0
    a1 = 0
    for block in T.serial(T.ceildiv(T.max(end - begin, 0), {stage})):
        base = begin + block * {stage}
        for token, col in T.Parallel({stage}, 64):
            keys[token, col] = T.if_then_else(base + token < end, kc[((base + token) * {kh} + kvhead) * 64 + col], 0)
            vals[token, col] = T.if_then_else(base + token < end, vc[((base + token) * {kh} + kvhead) * 64 + col], 0)
        T.sync_threads()
        for offset in T.serial(T.ceildiv(T.max(T.min({stage}, end - base) - worker, 0), {warps})):
            token = worker + offset * {warps}
            score = q0 * T.cast(keys[token, lane], "float32") + q1 * T.cast(keys[token, lane + 32], "float32")
{shuffle}
            score = T.shfl_sync(score, 0) * 0.125
            next_max = T.max(maximum, score)
            correction = T.exp(maximum - next_max)
            probability = T.exp(score - next_max)
            a0 = a0 * correction + probability * T.cast(vals[token, lane], "float32")
            a1 = a1 * correction + probability * T.cast(vals[token, lane + 32], "float32")
            normalizer = normalizer * correction + probability
            maximum = next_max
        T.sync_threads()
    values[warp, lane] = a0
    values[warp, lane + 32] = a1
    if lane == 0:
        maxima[warp] = maximum
        sums[warp] = normalizer
    T.sync_threads()
    if worker == 0:
        merged_max = T.alloc_var("float32")
        total = T.alloc_var("float32")
        result0 = T.alloc_var("float32")
        result1 = T.alloc_var("float32")
        merged_max = -T.infinity("float32")
        total = 0
        result0 = 0
        result1 = 0
        for other in T.unroll({warps}):
            merged_max = T.max(merged_max, maxima[local * {warps} + other])
        for other in T.unroll({warps}):
            index = local * {warps} + other
            factor = T.if_then_else(T.isfinite(merged_max), T.exp(maxima[index] - merged_max), 0)
            total = total + sums[index] * factor
            result0 = result0 + values[index, lane] * factor
            result1 = result1 + values[index, lane + 32] * factor
        base = (head * {splits} + split) * 66
        parts[base + lane] = result0
        parts[base + lane + 32] = result1
        if lane == 0:
            parts[base + 64] = merged_max
            parts[base + 65] = total''')


def fused_source(kind, p):
    """Retain baseline arithmetic for unsupported packed encodings and prefill."""
    text = baseline_source('linear', p)
    if kind == 'linear': return text
    if kind == 'linear_add':
        text = text.replace(', out: T.Tensor', f', residual: T.Tensor(({p["r"] * p["o"]},), "float32"), out: T.Tensor')
        return re.sub(r'out\[([^\]]+)\] = (.+)', r'out[\1] = residual[\1] + (\2)', text)
    k, o, q = p['k'], p['o'], p['type']
    _, block, size = TYPES[q]
    count = k * o if q in (0, 1) else k * o // block * size
    dtype = 'float16' if q == 1 else 'float32' if q == 0 else 'uint8'
    text = text.replace(', out: T.Tensor', f', w2: T.Tensor(({count},), "{dtype}"), out: T.Tensor')
    pair = lambda line: re.sub(r'\b(rhs|accum|total|w)\b', lambda m: {'w': 'w2'}.get(m[0], m[0] + '2'), line)
    lines = []
    for line in text.splitlines():
        if 'out[' in line and ' = ' in line:
            destination, gate = line.split(' = ', 1)
            lines.append(destination + f' = ({gate}) / (1 + T.exp(-({gate}))) * ({pair(gate)})')
        else:
            lines.append(line)
            if re.search(r'\b(rhs|accum|total)\b', line) and not line.startswith(('def ', '    return')):
                lines.append(pair(line))
    return '\n'.join(lines) + '\n'

def source(kind, p):
    a = lambda name, n, dtype='float32': (name, n, dtype)
    if kind in ('linear', 'linear_add', 'ffn'):
        if p['r']==1:
            return gemv_source(kind,p)
        return prefill_source(kind,p) if 'block_m' in p else fused_source(kind,p)
    if kind == 'attention_partial': return warp_partial_source(p, p['splits'])
    if kind == 'attention_grouped': return grouped_source(p,p.get('stage',32),p.get('warps',2))
    if kind == 'attention_merge': return merge_source(p, p['splits'])
    if kind in ('qnorm', 'kvnorm'):
        r, h, kh, d, cap = (p[name] for name in ('r', 'h', 'kh', 'd', 'cap'))
        original = baseline_source('qkv', p)
        body = original.split(' as (head, row):\n', 1)[1].split('\ndef tensor_export', 1)[0]
        query, keys = body.split(f'        if head < {kh}:\n', 1)
        declarations = '\n'.join(query.splitlines()[:3]) + '\n'
        if kind == 'qnorm':
            args = [a('q', r*h*d), a('qw', d), a('qo', r*h*d), a('pos', 2, 'int32')]
            body = textwrap.dedent(query).rstrip()
            heads = h
        else:
            args = [a('k', r*kh*d), a('v', r*kh*d), a('kw', d),
                    a('kc', cap*kh*d, 'float16'), a('vc', cap*kh*d, 'float16'), a('pos', 2, 'int32')]
            body = textwrap.dedent(declarations).rstrip() + '\n' + textwrap.dedent(keys).rstrip()
            heads = kh
        return emit(args, f'with T.Kernel({heads}, {r}, threads=64) as (head, row):\n' +
                    '\n'.join('    ' + line for line in body.splitlines()))
    if kind == 'prefill_tail':
        r, c, t = p['r'], p['c'], p['t']
        return emit([a('x', r*c), a('out', t*c), a('pos', 2, 'int32'), a('tail_pos', 2, 'int32')], f'''with T.Kernel(T.ceildiv({t*c}, 256), threads=256) as bx:
    for lane in T.Parallel(256):
        i = bx * 256 + lane
        if i < {t*c}:
            out[i] = T.if_then_else(i // {c} < T.min(pos[1], {t}), x[(T.max(pos[1] - {t}, 0) + i // {c}) * {c} + i % {c}], 0)
        if (bx == 0) & (lane == 0):
            tail_pos[0] = pos[0] + T.max(pos[1] - {t}, 0)
            tail_pos[1] = T.min(pos[1], {t})''')
    if kind == 'add_rms':
        text = baseline_source('rms', p)
        n = p['r'] * p['c']
        text = text.replace(', w: T.Tensor', f', mixed: T.Tensor(({n},), "float32"), residual: T.Tensor(({n},), "float32"), w: T.Tensor')
        text = re.sub(r'x\[([^\]]+)\]', lambda m: f'(x[{m[1]}] + mixed[{m[1]}])', text)
        index = f'row * {p["c"]} + i'
        return text.replace(f'            out[{index}]', f'            residual[{index}] = x[{index}] + mixed[{index}]\n            out[{index}]')
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
    return baseline_source(kind, p)

def merge_source(p,splits):
    h,d=p['h'],p['d'];stride=d+2
    return emit([('parts',h*splits*stride,'float32'),('out',h*d,'float32')],f'''with T.Kernel({h}, threads=128) as head:
    maxima = T.alloc_fragment(({splits},), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    sums = T.alloc_fragment(({splits},), "float32")
    normalizer = T.alloc_fragment((1,), "float32")
    products = T.alloc_fragment(({splits}, {d}), "float32")
    result = T.alloc_fragment(({d},), "float32")
    for i in T.Parallel({splits}):
        maxima[i] = parts[(head * {splits} + i) * {stride} + {d}]
    T.reduce_max(maxima, maximum, dim=0)
    for i in T.Parallel({splits}):
        sums[i] = parts[(head * {splits} + i) * {stride} + {d+1}] * T.exp(maxima[i] - maximum[0])
    T.reduce_sum(sums, normalizer, dim=0)
    for i, j in T.Parallel({splits}, {d}):
        products[i, j] = parts[(head * {splits} + i) * {stride} + j] * T.exp(maxima[i] - maximum[0])
    T.reduce_sum(products, result, dim=0)
    for j in T.Parallel({d}):
        out[head * {d} + j] = result[j] / normalizer[0]''')

def pair_load(q,k,buffer,row,col):
    _,block,size=TYPES[q]
    row_bytes=k*2 if q==1 else k//block*size
    if q==1:
        return f'bits = T.ldg32({buffer}[({row}) * {k} + ({col})])'
    if q==2:
        return f'''base = ({row}) * {row_bytes} + ({col}) // 32 * 18
scale = {_half(buffer,'base')}
packed = {_u16(buffer,f'base + 2 + ({col}) % 16')}
shift = ({col}) % 32 // 16 * 4
lo = scale * (T.cast((packed >> shift) & 15, "float32") - 8)
hi = scale * (T.cast((packed >> (shift + 8)) & 15, "float32") - 8)
bits = T.call_extern("uint32", "tensor_pack_f16x2", lo, hi)'''
    if q==12:
        return f'''base = ({row}) * {row_bytes} + ({col}) // 256 * 144
weight_col = ({col}) % 256
group = weight_col // 32
scale = T.if_then_else(group < 4, {buffer}[base + 4 + group] & 63, ({buffer}[base + 8 + group] & 15) | (({buffer}[base + group] >> 6) << 4))
minimum = T.if_then_else(group < 4, {buffer}[base + 8 + group] & 63, ({buffer}[base + 8 + group] >> 4) | (({buffer}[base + 4 + group] >> 6) << 4))
ds = {_half(buffer,'base')} * T.cast(scale, "float32")
dm = {_half(buffer,'base + 2')} * T.cast(minimum, "float32")
packed = {_u16(buffer,'base + 16 + weight_col // 64 * 32 + weight_col % 32')}
shift = group % 2 * 4
lo = ds * T.cast((packed >> shift) & 15, "float32") - dm
hi = ds * T.cast((packed >> (shift + 8)) & 15, "float32") - dm
bits = T.call_extern("uint32", "tensor_pack_f16x2", lo, hi)'''
    if q==14:
        return f'''base = ({row}) * {row_bytes} + ({col}) // 256 * 210
weight_col = ({col}) % 256
group = weight_col % 128 // 32
low = {_u16(buffer,'base + weight_col // 128 * 64 + group % 2 * 32 + weight_col % 32')}
high = {_u16(buffer,'base + 128 + weight_col // 128 * 32 + weight_col % 32')}
a = T.cast((low >> (group // 2 * 4)) & 15, "int32") | T.cast(((high >> (group * 2)) & 3) << 4, "int32")
b = T.cast((low >> (group // 2 * 4 + 8)) & 15, "int32") | T.cast(((high >> (group * 2 + 8)) & 3) << 4, "int32")
scale = {_half(buffer,'base + 208')} * T.cast(T.reinterpret("int8", {buffer}[base + 192 + weight_col // 16]), "float32")
bits = T.call_extern("uint32", "tensor_pack_f16x2", scale * T.cast(a - 32, "float32"), scale * T.cast(b - 32, "float32"))'''
    raise ValueError('unsupported packed pair encoding')


def prefill_source(kind, p):
    """FP16 tensor-core operands, FP32 accumulation, larger staged tiles."""
    r,k,o,q=(p[n] for n in ('r','k','o','type'))
    bm,bn,bk=p.get('block_m',32),p.get('block_n',64),p.get('block_k',32)
    stages,threads=p.get('stages',2),p.get('threads',128)
    paired=kind=='ffn'
    _,block,size=TYPES[q];count=k*o if q in (0,1) else k*o//block*size
    dtype='float16' if q==1 else 'float32' if q==0 else 'uint8'
    args=[('x',r*k,'float32'),('w',count,dtype)]
    if paired:args.append(('w2',count,dtype))
    if kind=='linear_add':args.append(('residual',r*o,'float32'))
    args.append(('out',r*o,'float32'))
    read=lambda w:weight(q,f'bx * {bn} + i',f'tile * {bk} + j',k,buffer=w)
    declarations=''
    loads=''
    gemm=''
    if paired:
        declarations=f'\n    rhs2 = T.alloc_shared(({bn}, {bk}), "float16")\n    accum2 = T.alloc_fragment(({bm}, {bn}), "float32")\n    T.clear(accum2)'
        loads=f'\n            rhs2[i, j] = T.if_then_else(bx * {bn} + i < {o}, {read("w2")}, 0)'
        gemm='\n        T.gemm(lhs, rhs2, accum2, transpose_B=True)'
    value='accum[i, j]'
    if paired:value=f'({value}) / (1 + T.exp(-({value}))) * accum2[i, j]'
    if kind=='linear_add':value=f'residual[(by * {bm} + i) * {o} + bx * {bn} + j] + ({value})'
    result=emit(args,f'''with T.Kernel(T.ceildiv({r}, {bm}), T.ceildiv({o}, {bn}), threads={threads}) as (by, bx):
    lhs = T.alloc_shared(({bm}, {bk}), "float16")
    rhs = T.alloc_shared(({bn}, {bk}), "float16")
    accum = T.alloc_fragment(({bm}, {bn}), "float32")
    T.clear(accum){declarations}
    for tile in T.Pipelined({k//bk}, num_stages={stages}):
        for i, j in T.Parallel({bm}, {bk}):
            lhs[i, j] = T.if_then_else(by * {bm} + i < {r}, x[(by * {bm} + i) * {k} + tile * {bk} + j], 0)
        for i, j in T.Parallel({bn}, {bk}):
            rhs[i, j] = T.if_then_else(bx * {bn} + i < {o}, {read("w")}, 0){loads}
        T.gemm(lhs, rhs, accum, transpose_B=True){gemm}
    for i, j in T.Parallel({bm}, {bn}):
        if (by * {bm} + i < {r}) & (bx * {bn} + j < {o}):
            out[(by * {bm} + i) * {o} + bx * {bn} + j] = {value}''')
    if p.get('packed_pairs'):
        old=f'        for i, j in T.Parallel({bn}, {bk}):\n            rhs[i, j] = T.if_then_else(bx * {bn} + i < {o}, {read("w")}, 0){loads}'
        replacement=f'        for i, j in T.Parallel({bn}, {bk//2}):'
        for rhs,w in [('rhs','w'),*([('rhs2','w2')] if paired else [])]:
            body=pair_load(q,k,w,f'bx * {bn} + i',f'tile * {bk} + j * 2')
            replacement+=f'\n            if bx * {bn} + i < {o}:\n'+ '\n'.join('                '+line for line in body.splitlines())
            replacement+=f'\n                {rhs}[i, j * 2] = T.reinterpret("float16", T.cast(bits & T.uint32(65535), "uint16"))\n                {rhs}[i, j * 2 + 1] = T.reinterpret("float16", T.cast(bits >> 16, "uint16"))\n            else:\n                {rhs}[i, j * 2] = 0\n                {rhs}[i, j * 2 + 1] = 0'
        old='\n'.join('    '+line for line in old.splitlines())
        replacement='\n'.join('    '+line for line in replacement.splitlines())
        assert old in result
        result=result.replace(old,replacement)
    return result
