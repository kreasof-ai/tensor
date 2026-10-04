"""Opt-in CUDA schedules transferred from the packed Vulkan inference work.

Runtime imports remain compiler-free. Templates produce NVRTC kernels; decode
keeps FP32 operands/accumulation and never expands packed weights in global memory.
"""
import re
import textwrap
from .kernels import emit, source as baseline_source
from .gguf import TYPES

HALF2 = '#include <cuda_fp16.h>\n'

CUDA_PROFILES = ('default', 'optimized')


def gemv_source(kind, p):
    """One warp per output row; read packed words once and reuse block scales.

    GGML Q4/Q6 rows are only two-byte aligned. Two 16-bit reads assemble a
    word without an unaligned uint32 load. Adjacent activation floats use
    aligned float4 loads. Scales are applied to FP32 dot products.
    """
    k, o, q = p['k'], p['o'], p['type']
    if q not in (0, 1, 2, 14) or k % 256:
        return fused_source(kind, p)
    threads = p.get('threads', 128)
    unroll = p.get('unroll', 4)
    if threads not in (64, 128, 256) or unroll not in (1, 2, 4, 8):
        raise ValueError('unsupported CUDA packed GEMV schedule')
    _, block, size = TYPES[q]
    row_bytes = k * (4 if q == 0 else 2) if q in (0, 1) else k // block * size
    if q == 2:
        dot = f'''
        const unsigned char* p = w + row * {row_bytes} + (tile * 8 + lane / 4) * 18;
        float scale = __half2float(*reinterpret_cast<const __half*>(p));
        unsigned int packed = load_word(p + 2 + (lane % 4) * 4);
        int j = tile * 256 + (lane / 4) * 32 + (lane % 4) * 4;
        float4 a = *reinterpret_cast<const float4*>(x + j);
        float4 b = *reinterpret_cast<const float4*>(x + j + 16);
        float value = a.x * float(int(packed & 15) - 8);
        value = fmaf(a.y, float(int((packed >> 8) & 15) - 8), value);
        value = fmaf(a.z, float(int((packed >> 16) & 15) - 8), value);
        value = fmaf(a.w, float(int((packed >> 24) & 15) - 8), value);
        value = fmaf(b.x, float(int((packed >> 4) & 15) - 8), value);
        value = fmaf(b.y, float(int((packed >> 12) & 15) - 8), value);
        value = fmaf(b.z, float(int((packed >> 20) & 15) - 8), value);
        value = fmaf(b.w, float(int((packed >> 28) & 15) - 8), value);
        return scale * value;
'''
        tiles = k // 256
    elif q == 14:
        dot = f'''
        const unsigned char* p = w + row * {row_bytes} + tile * 210;
        float total = 0;
        #pragma unroll
        for (int half = 0; half < 2; ++half) {{
            unsigned int low = load_word(p + half * 64 + (lane / 8 % 2) * 32 + lane % 8 * 4);
            unsigned int high = load_word(p + 128 + half * 32 + lane % 8 * 4);
            float4 a = *reinterpret_cast<const float4*>(x + tile * 256 + half * 128 + lane * 4);
            float values[4] = {{a.x, a.y, a.z, a.w}};
            float value = 0;
            #pragma unroll
            for (int byte = 0; byte < 4; ++byte) {{
                int v = int((low >> (byte * 8 + lane / 16 * 4)) & 15)
                      | int(((high >> (byte * 8 + lane / 8 * 2)) & 3) << 4);
                value = fmaf(values[byte], float(v - 32), value);
            }}
            float scale = float(reinterpret_cast<const signed char*>(p)[192 + half * 8 + lane / 4]);
            total = fmaf(scale, value, total);
        }}
        return __half2float(*reinterpret_cast<const __half*>(p + 208)) * total;
'''
        tiles = k // 256
    else:
        read = ('__half2float(reinterpret_cast<const __half*>(w)[row * '+str(k)+' + j + i])'
                if q == 1 else 'reinterpret_cast<const float*>(w)[row * '+str(k)+' + j + i]')
        dot = f'''
        int j = tile * 128 + lane * 4;
        float4 a = *reinterpret_cast<const float4*>(x + j);
        float values[4] = {{a.x, a.y, a.z, a.w}};
        float value = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) value = fmaf(values[i], {read}, value);
        return value;
'''
        tiles = k // 128
    paired = kind == 'ffn'
    residual = kind == 'linear_add'
    prelude = HALF2 + f'''
__device__ __forceinline__ unsigned int load_word(const unsigned char* p) {{
    return unsigned(*reinterpret_cast<const unsigned short*>(p)) |
           (unsigned(*reinterpret_cast<const unsigned short*>(p + 2)) << 16);
}}
__device__ __forceinline__ float packed_dot(const float* x, const unsigned char* w, int row, int lane, int tile) {{
{dot}
}}
__device__ __forceinline__ void packed_gemv(const float* x, const void* wptr,
        const void* w2ptr, const float* residual, float* out) {{
    const unsigned char* w = reinterpret_cast<const unsigned char*>(wptr);
    const unsigned char* w2 = reinterpret_cast<const unsigned char*>(w2ptr);
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * {threads // 32} + threadIdx.x / 32;
    float sums[{unroll}] = {{0}};
    {'float ups['+str(unroll)+'] = {0};' if paired else ''}
    if (row < {o}) {{
        for (int base = 0; base < {tiles}; base += {unroll}) {{
            #pragma unroll
            for (int slot = 0; slot < {unroll}; ++slot) {{
                if (base + slot < {tiles}) {{
                    sums[slot] += packed_dot(x, w, row, lane, base + slot);
                    {'ups[slot] += packed_dot(x, w2, row, lane, base + slot);' if paired else ''}
                }}
            }}
        }}
    }}
    float total = 0;
    {'float up = 0;' if paired else ''}
    #pragma unroll
    for (int slot = 0; slot < {unroll}; ++slot) {{
        total += sums[slot];
        {'up += ups[slot];' if paired else ''}
    }}
    for (int delta = 16; delta > 0; delta /= 2) {{
        total += __shfl_down_sync(0xffffffff, total, delta);
        {'up += __shfl_down_sync(0xffffffff, up, delta);' if paired else ''}
    }}
    if (lane == 0 && row < {o}) out[row] = {'total / (1 + __expf(-total)) * up' if paired else 'residual[row] + total' if residual else 'total'};
}}
'''
    count = k * o if q in (0, 1) else row_bytes * o
    dtype = 'float16' if q == 1 else 'float32' if q == 0 else 'uint8'
    args = [('x', k, 'float32'), ('w', count, dtype)]
    if paired: args.append(('w2', count, dtype))
    if residual: args.append(('residual', o, 'float32'))
    args.append(('out', o, 'float32'))
    w2 = 'T.address_of(w2[0])' if paired else 'T.address_of(w[0])'
    skip = 'T.address_of(residual[0])' if residual else 'T.address_of(out[0])'
    return emit(args, f'''with T.Kernel(T.ceildiv({o}, {threads // 32}), threads={threads}, prelude={prelude!r}) as bx:
    T.evaluate(T.call_extern("void", "packed_gemv", T.address_of(x[0]), T.address_of(w[0]), {w2}, {skip}, T.address_of(out[0])))''')


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
        return gemv_source(kind, p) if p['r'] == 1 else fused_source(kind, p)
    if kind == 'attention_partial': return warp_partial_source(p, p['splits'])
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

def warp_partial_source(p,splits):
    if type(splits) is not int or splits<1:raise ValueError('positive split count required')
    h,kh,d,cap=p['h'],p['kh'],p['d'],p['cap']
    if d!=64:raise ValueError('warp decode requires head dimension 64')
    stride=d+2
    prelude=HALF2+f'''
__device__ __forceinline__ void warp_attention(const float* q,const void* kptr,const void* vptr,float* parts,const int* pos) {{
    __shared__ float maxima[4], sums[4], values[4][64];
    const __half* kc=reinterpret_cast<const __half*>(kptr);
    const __half* vc=reinterpret_cast<const __half*>(vptr);
    const int lane=threadIdx.x%32, warp=threadIdx.x/32;
    const int head=blockIdx.x, split=blockIdx.y, length=pos[0]+1;
    const int chunk=(length+{splits}-1)/{splits};
    const int begin=split*chunk, end=min(begin+chunk,length);
    const int kvhead=head/{h//kh};
    const float q0=q[head*64+lane], q1=q[head*64+lane+32];
    float maximum=-__int_as_float(0x7f800000), normalizer=0, a0=0, a1=0;
    for(int token=begin+warp;token<end;token+=4) {{
        const int base=(token*{kh}+kvhead)*64;
        float score=q0*__half2float(kc[base+lane])+q1*__half2float(kc[base+lane+32]);
        for(int delta=16;delta>0;delta/=2) score+=__shfl_down_sync(0xffffffff,score,delta);
        score=__shfl_sync(0xffffffff,score,0)*0.125f;
        float next=fmaxf(maximum,score), correction=__expf(maximum-next), probability=__expf(score-next);
        a0=a0*correction+probability*__half2float(vc[base+lane]);
        a1=a1*correction+probability*__half2float(vc[base+lane+32]);
        normalizer=normalizer*correction+probability;maximum=next;
    }}
    values[warp][lane]=a0;values[warp][lane+32]=a1;
    if(lane==0) {{ maxima[warp]=maximum;sums[warp]=normalizer; }}
    __syncthreads();
    if(warp==0) {{
        float maximum=fmaxf(fmaxf(maxima[0],maxima[1]),fmaxf(maxima[2],maxima[3]));
        float sum=0,result0=0,result1=0;
        #pragma unroll
        for(int i=0;i<4;++i) {{
            float factor=isfinite(maximum) ? __expf(maxima[i]-maximum) : 0;
            sum+=sums[i]*factor;result0+=values[i][lane]*factor;result1+=values[i][lane+32]*factor;
        }}
        int base=(head*{splits}+split)*66;
        parts[base+lane]=result0;parts[base+lane+32]=result1;
        if(lane==0) {{parts[base+64]=maximum;parts[base+65]=sum;}}
    }}
}}
'''
    return emit([('q',h*d,'float32'),('kc',cap*kh*d,'float16'),('vc',cap*kh*d,'float16'),('parts',h*splits*stride,'float32'),('pos',2,'int32')],f'''with T.Kernel({h}, {splits}, threads=128,prelude={prelude!r}) as (head, split):
    T.evaluate(T.call_extern("void", "warp_attention", T.address_of(q[0]), T.address_of(kc[0]), T.address_of(vc[0]), T.address_of(parts[0]), T.address_of(pos[0])))''')

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
