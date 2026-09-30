"""TileLang source templates for the bounded manual-training operator library.

This module constructs source text only. Compiler imports stay inside build.
Every buffer is contiguous and flattened; logical dimensions are specialization
metadata. No vendor math library or Torch is used by the runtime kernels.
"""
from __future__ import annotations
import hashlib
import json
import math


def identity(kind, parameters):
    return hashlib.sha256(json.dumps([kind,parameters],sort_keys=True).encode()).hexdigest()[:24]


def emit(arguments, body):
    declarations = []
    for name, count, dtype in arguments:
        declarations.append(f'{name}: T.Tensor(({count},), "{dtype}")' if count else f'{name}: T.{dtype}')
    return ('import tilelang.language as T\n\n@T.prim_func\ndef kernel('+', '.join(declarations)+'):\n'
            +'\n'.join('    '+line for line in body.splitlines())
            +'\n\ndef tensor_export():\n    return {"kernel": kernel}\n')


def source(kind, p, schedule=None):
    n=p.get('n'); c=p.get('c'); r=p.get('r')
    f16='float16'; f32='float32'
    a=lambda name,count,dtype=f16:(name,count,dtype)
    scalar=lambda name:(name,None,'float32')
    simple='with T.Kernel(T.ceildiv(N, 256), threads=256) as block:\n    for lane in T.Parallel(256):\n        i = block * 256 + lane\n        if i < N:\n'
    if kind in ('zero','cast','add','gelu','gelu_backward'):
        if kind=='zero': args=[a('out',n,p.get('dtype',f32))]; expr='0'
        elif kind=='cast':
            args=[a('x',n,p.get('input',f16)),a('out',n,p.get('output',f32))]
            expr='T.cast(x[i], "float32")'+(' + out[i]' if p.get('add') else '')
        elif kind=='add': args=[a('x',n),a('y',n),a('out',n)];expr='T.cast(x[i], "float32") + T.cast(y[i], "float32")'
        elif kind=='gelu': args=[a('x',n),a('out',n)];expr='0.5 * T.cast(x[i], "float32") * (1 + T.erf(T.cast(x[i], "float32") * 0.7071067811865476))'
        else:
            args=[a('x',n),a('dy',n),a('out',n)]
            expr='T.cast(dy[i], "float32") * (0.5 * (1 + T.erf(T.cast(x[i], "float32") * 0.7071067811865476)) + T.cast(x[i], "float32") * 0.3989422804014327 * T.exp(-0.5 * T.cast(x[i], "float32") * T.cast(x[i], "float32")))'
        return emit(args,simple.replace('N',str(n))+'            out[i] = '+expr)
    if kind in ('gemm','gemm_gelu','gemm_residual'):
        b,m,k,cols=p['batch'],p['m'],p['k'],p['cols'];ta=p.get('ta',False);tb=p.get('tb',False)
        bm,bn,bk,stages=(schedule or (32,64,32,2))
        aa=(bk,bm) if ta else (bm,bk);bb=(bn,bk) if tb else (bk,bn)
        ai=f'(batch * {k} + tile * {bk} + i) * {m} + by * {bm} + j' if ta else f'(batch * {m} + by * {bm} + i) * {k} + tile * {bk} + j'
        bi=f'(batch * {cols} + bx * {bn} + i) * {k} + tile * {bk} + j' if tb else f'(batch * {k} + tile * {bk} + i) * {cols} + bx * {bn} + j'
        ag=f'(tile * {bk} + i < {k}) & (by * {bm} + j < {m})' if ta else f'(by * {bm} + i < {m}) & (tile * {bk} + j < {k})'
        bg=f'(bx * {bn} + i < {cols}) & (tile * {bk} + j < {k})' if tb else f'(tile * {bk} + i < {k}) & (bx * {bn} + j < {cols})'
        args=[a('x',b*m*k),a('w',b*k*cols)]
        pos=f'(batch * {m} + by * {bm} + i) * {cols} + bx * {bn} + j'
        store=f'out[{pos}] = accum[i, j]'
        if kind=='gemm_gelu':
            args.append(a('pre',b*m*cols))
            store=f'''value = T.cast(T.cast(accum[i, j], "float16"), "float32")
            pre[{pos}] = value
            out[{pos}] = 0.5 * value * (1 + T.erf(value * 0.7071067811865476))'''
        elif kind=='gemm_residual':
            args.append(a('residual',b*m*cols))
            store=f'out[{pos}] = T.cast(T.cast(accum[i, j], "float16"), "float32") + T.cast(residual[{pos}], "float32")'
        args.append(a('out',b*m*cols))
        return emit(args,f'''with T.Kernel(T.ceildiv({cols}, {bn}), T.ceildiv({m}, {bm}), {b}, threads=128) as (bx, by, batch):
    lhs = T.alloc_shared({aa!r}, "float16")
    rhs = T.alloc_shared({bb!r}, "float16")
    accum = T.alloc_fragment(({bm}, {bn}), "float32")
    T.clear(accum)
    for tile in T.Pipelined(T.ceildiv({k}, {bk}), num_stages={stages}):
        for i, j in T.Parallel({aa[0]}, {aa[1]}):
            lhs[i, j] = T.if_then_else({ag}, x[{ai}], 0)
        for i, j in T.Parallel({bb[0]}, {bb[1]}):
            rhs[i, j] = T.if_then_else({bg}, w[{bi}], 0)
        T.gemm(lhs, rhs, accum, transpose_A={ta!r}, transpose_B={tb!r})
    for i, j in T.Parallel({bm}, {bn}):
        if (by * {bm} + i < {m}) & (bx * {bn} + j < {cols}):
            {store}''')
    if kind=='embedding':
        b,s,c,v=p['b'],p['s'],p['c'],p['v']; n=b*s*c
        return emit([a('tokens',b*s,'int32'),a('w',v*c),a('position',s*c),a('out',n)],simple.replace('N',str(n))+f'''            out[i] = T.cast(w[tokens[i // {c}] * {c} + i % {c}], "float32") + T.cast(position[(i // {c} % {s}) * {c} + i % {c}], "float32")''')
    if kind=='embedding_backward':
        b,s,c,v=p['b'],p['s'],p['c'],p['v'];n=b*s*c
        return emit([a('tokens',b*s,'int32'),a('dy',n),a('dw',v*c,f32),a('dp',s*c,f32)],simple.replace('N',str(n))+f'''            T.atomic_add(dw[tokens[i // {c}] * {c} + i % {c}], T.cast(dy[i], "float32"))
            T.atomic_add(dp[(i // {c} % {s}) * {c} + i % {c}], T.cast(dy[i], "float32"))''')
    if kind=='norm':
        tile=1<<(c-1).bit_length()
        return emit([a('x',r*c),a('weight',c,f32),a('out',r*c),a('normal',r*c,f32),a('inverse',r,f32)],f'''with T.Kernel({r}, threads=256) as row:
    values = T.alloc_fragment(({tile},), "float32")
    square = T.alloc_fragment(({tile},), "float32")
    total = T.alloc_fragment((1,), "float32")
    variance = T.alloc_fragment((1,), "float32")
    for i in T.Parallel({tile}):
        values[i] = T.if_then_else(i < {c}, T.cast(x[row * {c} + i], "float32"), 0)
    T.reduce_sum(values, total, dim=0)
    for i in T.Parallel({tile}):
        square[i] = T.if_then_else(i < {c}, (values[i] - total[0] / {c}) * (values[i] - total[0] / {c}), 0)
    T.reduce_sum(square, variance, dim=0)
    for i in T.Parallel({tile}):
        if i < {c}:
            z = (values[i] - total[0] / {c}) * T.rsqrt(variance[0] / {c} + 0.00001)
            normal[row * {c} + i] = z
            out[row * {c} + i] = z * weight[i]
    inverse[row] = T.rsqrt(variance[0] / {c} + 0.00001)''')
    if kind=='norm_backward':
        tile=1<<(c-1).bit_length()
        return emit([a('dy',r*c),a('weight',c,f32),a('normal',r*c,f32),a('inverse',r,f32),a('dx',r*c),a('parts',r*c,f32)],f'''with T.Kernel({r}, threads=256) as row:
    values = T.alloc_fragment(({tile},), "float32")
    product = T.alloc_fragment(({tile},), "float32")
    total = T.alloc_fragment((1,), "float32")
    dot = T.alloc_fragment((1,), "float32")
    for i in T.Parallel({tile}):
        values[i] = T.if_then_else(i < {c}, T.cast(dy[row * {c} + i], "float32") * weight[i], 0)
        product[i] = T.if_then_else(i < {c}, values[i] * normal[row * {c} + i], 0)
    T.reduce_sum(values, total, dim=0)
    T.reduce_sum(product, dot, dim=0)
    for i in T.Parallel({tile}):
        if i < {c}:
            dx[row * {c} + i] = inverse[row] * (values[i] - (total[0] + normal[row * {c} + i] * dot[0]) / {c})
            parts[row * {c} + i] = T.cast(dy[row * {c} + i], "float32") * normal[row * {c} + i]''')
    if kind=='column_sum':
        return emit([a('x',r*c,f32),a('out',c,f32)],f'''with T.Kernel(T.ceildiv({c}, 256), threads=256) as block:
    sums = T.alloc_fragment((256,), "float32")
    T.clear(sums)
    for row in T.serial({r}):
        for i in T.Parallel(256):
            if block * 256 + i < {c}:
                sums[i] += x[row * {c} + block * 256 + i]
    for i in T.Parallel(256):
        if block * 256 + i < {c}:
            out[block * 256 + i] = sums[i]''')
    if kind in ('pack_qkv','unpack_qkv','merge_heads','split_heads'):
        b,s,c,h=p['b'],p['s'],p['c'],p['h'];d=c//h;n=b*s*c
        flat=f'(i // {s*d} // {h} * {s} + i // {d} % {s}) * {c} + (i // {s*d} % {h}) * {d} + i % {d}'
        if kind=='pack_qkv':
            args=[a('x',3*n),a('q',n),a('k',n),a('v',n)]
            expression=f'''j = {flat}
            q[i] = x[j // {c} * {3*c} + j % {c}]
            k[i] = x[j // {c} * {3*c} + {c} + j % {c}]
            v[i] = x[j // {c} * {3*c} + {2*c} + j % {c}]'''
        elif kind=='unpack_qkv':
            args=[a('q',n),a('k',n),a('v',n),a('out',3*n)]
            expression=f'''j = {flat}
            out[j // {c} * {3*c} + j % {c}] = q[i]
            out[j // {c} * {3*c} + {c} + j % {c}] = k[i]
            out[j // {c} * {3*c} + {2*c} + j % {c}] = v[i]'''
        else:
            args=[a('x',n),a('out',n)]
            expression=f'out[{flat}] = x[i]' if kind=='merge_heads' else f'out[i] = x[{flat}]'
        return emit(args,simple.replace('N',str(n))+'            '+expression)
    if kind=='softmax':
        s,bh,d=p['s'],p['bh'],p['d'];tile=1<<(s-1).bit_length();scale=d**-0.5
        return emit([a('scores',bh*s*s),a('prob',bh*s*s),a('saved',bh*s*s,f32)],f'''with T.Kernel({bh*s}, threads=256) as row:
    values = T.alloc_fragment(({tile},), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    total = T.alloc_fragment((1,), "float32")
    for i in T.Parallel({tile}):
        values[i] = T.if_then_else((i < {s}) & (i <= row % {s}), T.cast(T.cast(T.cast(scores[row * {s} + i], "float32") * {scale}, "float16"), "float32"), -T.infinity("float32"))
    T.reduce_max(values, maximum, dim=0)
    for i in T.Parallel({tile}):
        values[i] = T.exp(values[i] - maximum[0])
    T.reduce_sum(values, total, dim=0)
    for i in T.Parallel({tile}):
        if i < {s}:
            saved[row * {s} + i] = values[i] / total[0]
            prob[row * {s} + i] = values[i] / total[0]''')
    if kind=='softmax_backward':
        s,bh,d=p['s'],p['bh'],p['d'];tile=1<<(s-1).bit_length();scale=d**-0.5
        return emit([a('dp',bh*s*s),a('prob',bh*s*s,f32),a('ds',bh*s*s)],f'''with T.Kernel({bh*s}, threads=256) as row:
    product = T.alloc_fragment(({tile},), "float32")
    total = T.alloc_fragment((1,), "float32")
    for i in T.Parallel({tile}):
        product[i] = T.if_then_else(i < {s}, T.cast(dp[row * {s} + i], "float32") * prob[row * {s} + i], 0)
    T.reduce_sum(product, total, dim=0)
    for i in T.Parallel({tile}):
        if i < {s}:
            ds[row * {s} + i] = T.cast(T.cast((T.cast(dp[row * {s} + i], "float32") - total[0]) * prob[row * {s} + i], "float16"), "float32") * {scale}''')
    if kind=='ce_parts':
        r,v=p['r'],p['v'];chunks=math.ceil(v/1024)
        return emit([a('x',r*v),a('parts',r*chunks*2,f32)],f'''with T.Kernel({chunks}, {r}, threads=256) as (block, row):
    values = T.alloc_fragment((1024,), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    total = T.alloc_fragment((1,), "float32")
    for i in T.Parallel(1024):
        values[i] = T.if_then_else(block * 1024 + i < {v}, T.cast(x[row * {v} + block * 1024 + i], "float32"), -T.infinity("float32"))
    T.reduce_max(values, maximum, dim=0)
    for i in T.Parallel(1024):
        values[i] = T.exp(values[i] - maximum[0])
    T.reduce_sum(values, total, dim=0)
    parts[(row * {chunks} + block) * 2] = maximum[0]
    parts[(row * {chunks} + block) * 2 + 1] = total[0]''')
    if kind=='ce_loss':
        r,v=p['r'],p['v'];chunks=math.ceil(v/1024);tile=1<<(chunks-1).bit_length()
        return emit([a('x',r*v),a('target',r,'int32'),a('parts',r*chunks*2,f32),a('lse',r,f32),a('loss',r,f32)],f'''with T.Kernel({r}, threads=128) as row:
    maxima = T.alloc_fragment(({tile},), "float32")
    sums = T.alloc_fragment(({tile},), "float32")
    maximum = T.alloc_fragment((1,), "float32")
    total = T.alloc_fragment((1,), "float32")
    for i in T.Parallel({tile}):
        maxima[i] = T.if_then_else(i < {chunks}, parts[(row * {chunks} + i) * 2], -T.infinity("float32"))
    T.reduce_max(maxima, maximum, dim=0)
    for i in T.Parallel({tile}):
        sums[i] = T.if_then_else(i < {chunks}, parts[(row * {chunks} + i) * 2 + 1] * T.exp(maxima[i] - maximum[0]), 0)
    T.reduce_sum(sums, total, dim=0)
    lse[row] = maximum[0] + T.log(total[0])
    loss[row] = maximum[0] + T.log(total[0]) - T.cast(x[row * {v} + target[row]], "float32")''')
    if kind=='ce_backward':
        r,v=p['r'],p['v'];n=r*v
        return emit([a('x',n),a('target',r,'int32'),a('lse',r,f32),a('dy',r,f32),a('out',n)],simple.replace('N',str(n))+f'''            out[i] = (T.exp(T.cast(x[i], "float32") - lse[i // {v}]) - T.if_then_else(i % {v} == target[i // {v}], 1.0, 0.0)) * dy[i // {v}]''')
    if kind=='sumsq':
        chunks=math.ceil(n/1024);total=p['total'];scale=p['scale']
        return emit([a('x',n,f32),a('out',total,f32),('offset',None,'int32')],f'''with T.Kernel({chunks}, threads=256) as block:
    values = T.alloc_fragment((1024,), "float32")
    total = T.alloc_fragment((1,), "float32")
    for i in T.Parallel(1024):
        values[i] = T.if_then_else(block * 1024 + i < {n}, (x[block * 1024 + i] / {scale}) * (x[block * 1024 + i] / {scale}), 0)
    T.reduce_sum(values, total, dim=0)
    out[offset + block] = total[0]''')
    if kind in ('sum_parts','clip'):
        chunks=math.ceil(n/1024)
        args=[a('x',n,f32),a('out',2 if kind=='clip' else chunks,f32)]
        tile=1<<(n-1).bit_length() if kind=='clip' else 1024
        blocks=1 if kind=='clip' else chunks
        store='out[block] = total[0]' if kind=='sum_parts' else f'''out[0] = T.if_then_else(T.isnan(total[0]) | (total[0] == T.infinity("float32")), T.infinity("float32"), T.min(1.0, {p['limit']} / (T.sqrt(total[0]) + 0.000001)))
    out[1] = T.sqrt(total[0])'''
        return emit(args,f'''with T.Kernel({blocks}, threads=256) as block:
    values = T.alloc_fragment(({tile},), "float32")
    total = T.alloc_fragment((1,), "float32")
    for i in T.Parallel({tile}):
        values[i] = T.if_then_else(block * {tile} + i < {n}, x[block * {tile} + i], 0)
    T.reduce_sum(values, total, dim=0)
    {store}''')
    if kind=='adamw':
        return emit([a('weight',n,f32),a('half',n),a('grad',n,f32),a('moment',n,f32),a('variance',n,f32),a('clip',2,f32),scalar('lr'),scalar('correction1'),scalar('correction2')],simple.replace('N',str(n))+f'''            if clip[0] != T.infinity("float32"):
                g = grad[i] / {p['scale']} * clip[0]
                m = 0.9 * moment[i] + 0.1 * g
                v = 0.95 * variance[i] + 0.05 * g * g
                value = weight[i] * (1 - lr * {p['decay']}) - lr * (m / correction1) / (T.sqrt(v / correction2) + 0.00000001)
                moment[i] = m
                variance[i] = v
                weight[i] = value
                half[i] = value''')
    raise ValueError(f'unknown training kernel {kind}')
