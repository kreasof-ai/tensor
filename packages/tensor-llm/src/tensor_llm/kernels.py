"""TileLang/TIRx templates for packed GGML weights and LFM2 forward execution."""
import hashlib
import json
from .gguf import TYPES


def identity(kind,p):return hashlib.sha256(json.dumps([kind,p],sort_keys=True).encode()).hexdigest()[:24]


def emit(args,body):
    declarations=[f'{name}: T.Tensor(({n},), "{dtype}")' for name,n,dtype in args]
    return ('import tilelang.language as T\n\n@T.prim_func\ndef kernel('+', '.join(declarations)+'):\n'
            +'\n'.join('    '+line for line in body.splitlines())+'\n\ndef tensor_export():\n    return {"kernel": kernel}\n')


def weight(kind,row,col,k,buffer='w',*,packed_words=False):
    """Expression reading one exact GGML value; packed buffers are unsigned bytes."""
    if kind in (0,1):return f'T.cast({buffer}[({row}) * {k} + ({col})], "float32")'
    _,block,size=TYPES[kind]
    base=f'((({row}) * {k} + ({col})) // {block} * {size})';j=f'(({col}) % {block})'
    u=lambda offset:f'T.cast({buffer}[{base} + ({offset})], "uint32")'
    half=lambda offset:f'T.cast(T.reinterpret("float16", T.cast({u(offset)} | ({u(str(offset)+" + 1")} << 8), "uint16")), "float32")'
    if packed_words:
        u=lambda offset:f'(({buffer}[({base} + ({offset})) // 4] >> (({base} + ({offset})) % 4 * 8)) & T.uint32(255))'
        def half(offset):
            bits=f'({u(offset)} | ({u(str(offset)+" + 1")} << 8))'
            return f'T.call_extern("float32", "tensor_unpack_f16", {bits})'
    signed=lambda expr:f'T.cast(T.reinterpret("int8", T.cast({expr}, "uint8")), "float32")' if not packed_words else f'T.cast(T.cast({expr}, "int32") - T.if_then_else({expr} >= 128, 256, 0), "float32")'
    if kind==2:return f'{half(0)} * (T.cast(({u("2 + "+j+" % 16")} >> (4 * ({j} // 16))) & 15, "float32") - 8)'
    if kind==8:return f'{half(0)} * {signed(u("2 + "+j))}'
    if kind==12:
        group=f'({j} // 32)'
        scale=f'T.if_then_else({group} < 4, {u("4 + "+group)} & 63, ({u("8 + "+group)} & 15) | (({u(group)} >> 6) << 4))'
        minimum=f'T.if_then_else({group} < 4, {u("8 + "+group)} & 63, ({u("8 + "+group)} >> 4) | (({u("4 + "+group)} >> 6) << 4))'
        q=f'(({u("16 + "+j+" // 64 * 32 + "+j+" % 32")} >> (4 * ({group} % 2))) & 15)'
        return f'{half(0)} * T.cast({scale}, "float32") * T.cast({q}, "float32") - {half(2)} * T.cast({minimum}, "float32")'
    if kind==14:
        group=f'({j} % 128 // 32)'
        low=u(j+' // 128 * 64 + '+group+' % 2 * 32 + '+j+' % 32')
        high=u('128 + '+j+' // 128 * 32 + '+j+' % 32')
        scale=u('192 + '+j+' // 16')
        q=f'((({low} >> (4 * ({group} // 2))) & 15) | ((({high} >> (2 * {group})) & 3) << 4))'
        return f'{half(208)} * {signed(scale)} * (T.cast({q}, "float32") - 32)'
    raise ValueError('unsupported GGML encoding')


def source(kind,p):
    c=p.get('c');r=p.get('r',1)
    a=lambda name,n,dtype='float32':(name,n,dtype)
    if kind=='linear':
        k,o,q=p['k'],p['o'],p['type'];_,block,size=TYPES[q]
        w=a('w',k*o if q in (0,1) else k*o//block*size,'float16' if q==1 else 'float32' if q==0 else 'uint8')
        args=[a('x',r*k),w,a('out',r*o)]
        if r==1:
            return emit(args,f'''with T.Kernel(T.ceildiv({o}, 4), threads=128) as bx:
    accum = T.alloc_fragment((4, 32), "float32")
    total = T.alloc_fragment((4,), "float32")
    T.clear(accum)
    for tile in T.serial({k//32}):
        for row, lane in T.Parallel(4, 32):
            accum[row, lane] += x[tile * 32 + lane] * ({weight(q,'bx * 4 + row','tile * 32 + lane',k)})
    T.reduce_sum(accum, total, dim=1)
    for row in T.Parallel(4):
        if bx * 4 + row < {o}:
            out[bx * 4 + row] = total[row]''')
        return emit(args,f'''with T.Kernel(T.ceildiv({r}, 32), T.ceildiv({o}, 64), threads=128) as (by, bx):
    lhs = T.alloc_shared((32, 32), "float16")
    rhs = T.alloc_shared((64, 32), "float16")
    accum = T.alloc_fragment((32, 64), "float32")
    T.clear(accum)
    for tile in T.Pipelined({k//32}, num_stages=2):
        for i, j in T.Parallel(32, 32):
            lhs[i, j] = T.if_then_else(by * 32 + i < {r}, x[(by * 32 + i) * {k} + tile * 32 + j], 0)
        for i, j in T.Parallel(64, 32):
            rhs[i, j] = T.if_then_else(bx * 64 + i < {o}, {weight(q,'bx * 64 + i','tile * 32 + j',k)}, 0)
        T.gemm(lhs, rhs, accum, transpose_B=True)
    for i, j in T.Parallel(32, 64):
        if (by * 32 + i < {r}) & (bx * 64 + j < {o}):
            out[(by * 32 + i) * {o} + bx * 64 + j] = accum[i, j]''')
    if kind=='rms':
        return emit([a('x',r*c),a('w',c),a('out',r*c)],f'''with T.Kernel({r}, threads=256) as row:
    square = T.alloc_fragment(({c},), "float32")
    total = T.alloc_fragment((1,), "float32")
    for i in T.Parallel({c}):
        square[i] = x[row * {c} + i] * x[row * {c} + i]
    T.reduce_sum(square, total, dim=0)
    for i in T.Parallel({c}):
        out[row * {c} + i] = x[row * {c} + i] * T.rsqrt(total[0] / {c} + {p['eps']}) * w[i]''')
    if kind in ('add','swiglu'):
        n=r*c
        expression='x[i] + y[i]' if kind=='add' else 'x[i] / (1 + T.exp(-x[i])) * y[i]'
        return emit([a('x',n),a('y',n),a('out',n)],f'''with T.Kernel(T.ceildiv({n}, 256), threads=256) as block:
    for lane in T.Parallel(256):
        i = block * 256 + lane
        if i < {n}:
            out[i] = {expression}''')
    if kind=='embedding':
        q,v=p['type'],p['v'];_,block,size=TYPES[q]
        return emit([a('tokens',r,'int32'),a('w',v*c if q in (0,1) else v*c//block*size,'float16' if q==1 else 'float32' if q==0 else 'uint8'),a('out',r*c)],f'''with T.Kernel(T.ceildiv({c}, 256), {r}, threads=256) as (bx, row):
    for lane in T.Parallel(256):
        col = bx * 256 + lane
        if col < {c}:
            out[row * {c} + col] = {weight(q,'tokens[row]','col',c)}''')
    if kind=='conv':
        return emit([a('x',r*3*c),a('w',c*3),a('state',2*c),a('out',r*c),a('pos',2,'int32')],f'''with T.Kernel(T.ceildiv({c}, 256), threads=256) as block:
    for lane in T.Parallel(256):
        col = block * 256 + lane
        if col < {c}:
            left = T.alloc_var("float32")
            right = T.alloc_var("float32")
            current = T.alloc_var("float32")
            left = state[col]
            right = state[{c} + col]
            for row in T.serial({r}):
                if row < pos[1]:
                    current = x[row * {3*c} + col] * x[row * {3*c} + {2*c} + col]
                    out[row * {c} + col] = x[row * {3*c} + {c} + col] * (left * w[col * 3] + right * w[col * 3 + 1] + current * w[col * 3 + 2])
                    left = right
                    right = current
                else:
                    out[row * {c} + col] = 0
            state[col] = left
            state[{c} + col] = right''')
    if kind=='qkv':
        h,kh,d,cap=p['h'],p['kh'],p['d'],p['cap'];eps,theta=p['eps'],p['theta']
        args=[a('q',r*h*d),a('k',r*kh*d),a('v',r*kh*d),a('qw',d),a('kw',d),
              a('qo',r*h*d),a('kc',cap*kh*d,'float16'),a('vc',cap*kh*d,'float16'),a('pos',2,'int32')]
        # LFM2 uses NeoX RoPE: pair the two halves of each head.
        return emit(args,f'''with T.Kernel({h}, {r}, threads=64) as (head, row):
    square = T.alloc_fragment(({d},), "float32")
    total = T.alloc_fragment((1,), "float32")
    norm = T.alloc_shared(({d},), "float32")
    for i in T.Parallel({d}):
        square[i] = q[(row * {h} + head) * {d} + i] * q[(row * {h} + head) * {d} + i]
    T.reduce_sum(square, total, dim=0)
    for i in T.Parallel({d}):
        norm[i] = q[(row * {h} + head) * {d} + i] * T.rsqrt(total[0] / {d} + {eps}) * qw[i]
    for i in T.Parallel({d//2}):
        angle = T.cast(pos[0] + row, "float32") * T.exp(-{__import__('math').log(theta)*2/d} * i)
        qo[(row * {h} + head) * {d} + i] = norm[i] * T.cos(angle) - norm[i + {d//2}] * T.sin(angle)
        qo[(row * {h} + head) * {d} + i + {d//2}] = norm[i] * T.sin(angle) + norm[i + {d//2}] * T.cos(angle)
    if head < {kh}:
        for i in T.Parallel({d}):
            square[i] = k[(row * {kh} + head) * {d} + i] * k[(row * {kh} + head) * {d} + i]
        T.reduce_sum(square, total, dim=0)
        for i in T.Parallel({d}):
            norm[i] = k[(row * {kh} + head) * {d} + i] * T.rsqrt(total[0] / {d} + {eps}) * kw[i]
            if row < pos[1]:
                vc[((pos[0] + row) * {kh} + head) * {d} + i] = v[(row * {kh} + head) * {d} + i]
        for i in T.Parallel({d//2}):
            angle = T.cast(pos[0] + row, "float32") * T.exp(-{__import__('math').log(theta)*2/d} * i)
            if row < pos[1]:
                kc[((pos[0] + row) * {kh} + head) * {d} + i] = norm[i] * T.cos(angle) - norm[i + {d//2}] * T.sin(angle)
                kc[((pos[0] + row) * {kh} + head) * {d} + i + {d//2}] = norm[i] * T.sin(angle) + norm[i + {d//2}] * T.cos(angle)''')
    if kind=='attention':
        h,kh,d,cap=p['h'],p['kh'],p['d'],p['cap']
        args=[a('q',r*h*d),a('kc',cap*kh*d,'float16'),a('vc',cap*kh*d,'float16'),a('out',r*h*d),a('pos',2,'int32')]
        if r==1:
            return emit(args,f'''with T.Kernel({h}, threads=128) as head:
    dots = T.alloc_fragment((64, {d}), "float32")
    scores = T.alloc_fragment((64,), "float32")
    products = T.alloc_fragment((64, {d}), "float32")
    partial = T.alloc_fragment(({d},), "float32")
    result = T.alloc_fragment(({d},), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    previous = T.alloc_fragment((1,), "float32")
    normalizer = T.alloc_fragment((1,), "float32")
    total = T.alloc_fragment((1,), "float32")
    T.fill(maximum, -T.infinity("float32"))
    T.clear(result)
    T.clear(normalizer)
    for tile in T.serial(T.ceildiv(pos[0] + 1, 64)):
        for i, j in T.Parallel(64, {d}):
            dots[i, j] = q[head * {d} + j] * T.cast(kc[((tile * 64 + i) * {kh} + head // {h//kh}) * {d} + j], "float32")
        T.reduce_sum(dots, scores, dim=1)
        T.copy(maximum, previous)
        for i in T.Parallel(64):
            scores[i] = T.if_then_else(tile * 64 + i <= pos[0], scores[i] * {d**-0.5}, -T.infinity("float32"))
        T.reduce_max(scores, maximum, dim=0, clear=False)
        for i in T.Parallel(64):
            scores[i] = T.exp(scores[i] - maximum[0])
        T.reduce_sum(scores, total, dim=0)
        normalizer[0] = normalizer[0] * T.exp(previous[0] - maximum[0]) + total[0]
        for i, j in T.Parallel(64, {d}):
            products[i, j] = scores[i] * T.cast(vc[((tile * 64 + i) * {kh} + head // {h//kh}) * {d} + j], "float32")
        T.reduce_sum(products, partial, dim=0)
        for j in T.Parallel({d}):
            result[j] = result[j] * T.exp(previous[0] - maximum[0]) + partial[j]
    for j in T.Parallel({d}):
        out[head * {d} + j] = result[j] / normalizer[0]''')
        return emit(args,f'''with T.Kernel(T.ceildiv({r}, 32), {h}, threads=128) as (bx, head):
    query = T.alloc_shared((32, {d}), "float16")
    key = T.alloc_shared((64, {d}), "float16")
    value = T.alloc_shared((64, {d}), "float16")
    prob = T.alloc_shared((32, 64), "float16")
    scores = T.alloc_fragment((32, 64), "float32")
    result = T.alloc_fragment((32, {d}), "float32")
    maximum = T.alloc_fragment((32,), "float32")
    previous = T.alloc_fragment((32,), "float32")
    factor = T.alloc_fragment((32,), "float32")
    normalizer = T.alloc_fragment((32,), "float32")
    total = T.alloc_fragment((32,), "float32")
    for i, j in T.Parallel(32, {d}):
        query[i, j] = T.if_then_else(bx * 32 + i < {r}, q[((bx * 32 + i) * {h} + head) * {d} + j], 0)
    T.clear(result)
    T.clear(normalizer)
    T.fill(maximum, -T.infinity("float32"))
    for tile in T.serial(T.ceildiv(pos[0] + T.min({r}, (bx + 1) * 32), 64)):
        for i, j in T.Parallel(64, {d}):
            key[i, j] = kc[((tile * 64 + i) * {kh} + head // {h//kh}) * {d} + j]
            value[i, j] = vc[((tile * 64 + i) * {kh} + head // {h//kh}) * {d} + j]
        T.gemm(query, key, scores, transpose_B=True, clear_accum=True)
        T.copy(maximum, previous)
        for i, j in T.Parallel(32, 64):
            scores[i, j] = T.if_then_else(tile * 64 + j <= pos[0] + bx * 32 + i, scores[i, j] * {d**-0.5}, -T.infinity("float32"))
        T.reduce_max(scores, maximum, dim=1, clear=False)
        for i in T.Parallel(32):
            factor[i] = T.exp(previous[i] - maximum[i])
        for i, j in T.Parallel(32, 64):
            scores[i, j] = T.exp(scores[i, j] - maximum[i])
        T.reduce_sum(scores, total, dim=1)
        for i in T.Parallel(32):
            normalizer[i] = normalizer[i] * factor[i] + total[i]
        for i, j in T.Parallel(32, {d}):
            result[i, j] *= factor[i]
        T.copy(scores, prob)
        T.gemm(prob, value, result)
    for i, j in T.Parallel(32, {d}):
        if bx * 32 + i < {r}:
            out[((bx * 32 + i) * {h} + head) * {d} + j] = T.if_then_else(bx * 32 + i < pos[1], result[i, j] / normalizer[i], 0)''')
    if kind=='last':
        return emit([a('x',r*c),a('out',c),a('pos',2,'int32')],f'''with T.Kernel(T.ceildiv({c}, 256), threads=256) as block:
    for lane in T.Parallel(256):
        col = block * 256 + lane
        if col < {c}:
            out[col] = x[(pos[1] - 1) * {c} + col]''')
    if kind=='advance':
        return emit([a('pos',2,'int32')],f'''with T.Kernel(1, threads=32):
    for i in T.Parallel(1):
        pos[0] += pos[1]''')
    raise ValueError(f'unknown LFM2 kernel {kind}')
