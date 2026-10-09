"""Experimental packed-load pipeline and split-KV decode; core defaults unchanged."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse
import copy
import hashlib
import json
import shutil
import textwrap
from pathlib import Path
import numpy as np
import tensor
from tensor.artifacts.format import read_artifact
from tensor.runtime.abi import BoundCall
from tensor_llm import LFM2
from tensor_llm.lfm2.kernels.baseline import source
from benchmarks.lfm2.text_helpers import emit, reference_attention_source
from benchmarks.lfm2.fp16_decode import linear_source, HALF2


def compile_source(text,out,target='sm_86'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256(text.encode()).hexdigest()[:24]
    src=out/(key+'.py');artifact=out/(key+'.tbin')
    if not src.exists() or src.read_text()!=text:
        src.write_text(text);artifact.unlink(missing_ok=True)
    if artifact.exists() and read_artifact(artifact)[0]['target']!=target:artifact.unlink()
    if not artifact.exists():tensor.build(src,artifact,compiler='nvrtc',target=target,cache_dir=out/'compiler-cache')
    return artifact


UNPACK=r'''
__device__ __forceinline__ float packed_weight(const unsigned char* p, int j, int q) {
    float d=__half2float(*reinterpret_cast<const __half*>(p));
    if(q==2) return d * float(int((p[2+j%16] >> (4*(j/16))) & 15)-8);
    if(q==12) {
        int g=j/32;
        int s=g<4 ? (p[4+g]&63) : ((p[8+g]&15) | ((p[g]>>6)<<4));
        int m=g<4 ? (p[8+g]&63) : ((p[8+g]>>4) | ((p[4+g]>>6)<<4));
        int v=(p[16+j/64*32+j%32] >> (4*(g%2))) & 15;
        float minimum=__half2float(*reinterpret_cast<const __half*>(p+2));
        return d*float(s)*float(v)-minimum*float(m);
    }
    int g=j%128/32;
    int low=p[j/128*64+g%2*32+j%32];
    int high=p[128+j/128*32+j%32];
    int v=((low>>(4*(g/2)))&15) | (((high>>(2*g))&3)<<4);
    float scale=__half2float(*reinterpret_cast<const __half*>(p+208));
    return scale*float(reinterpret_cast<const signed char*>(p)[192+j/16])*float(v-32);
}
'''


def pipelined_source(p,*,async_load=True):
    """Two 512-column packed shared-memory stages; never writes expanded global weights."""
    k,o,q=p['k'],p['o'],p['type']
    if p['r']!=1:raise ValueError('one-row decode required')
    if q not in (2,12,14) or k%512:return linear_source(p,'fp16_half2')
    from tensor_llm.common.gguf import TYPES
    _,block,size=TYPES[q];packed=512//block*size;row_bytes=k//block*size
    vector=4 if q==14 else 16
    copy_expression=(f'__pipeline_memcpy_async(stage[slot]+i*{vector}, w+row*{row_bytes}+tile*{packed}+offset, {vector});'
                     if async_load else f'*reinterpret_cast<{"uint4" if vector==16 else "unsigned int"}*>(stage[slot]+i*{vector}) = *reinterpret_cast<const {"uint4" if vector==16 else "unsigned int"}*>(w+row*{row_bytes}+tile*{packed}+offset);')
    prelude=HALF2+'\n#include <cuda_pipeline_primitives.h>\n'+UNPACK+f'''
__device__ __forceinline__ void packed_pipeline(const float* x,const unsigned char* w,float* out) {{
    __shared__ __align__(16) unsigned char stage[2][{4*packed}];
    const int tid=threadIdx.x, lane=tid%32, local_row=tid/32;
    const int row=blockIdx.x*4+local_row;
    float sum=0;
    for(int tile=0;tile<{k//512};++tile) {{
        // Prime tile zero, then initiate the next load before consuming this tile.
        if(tile==0) {{
            const int slot=0;
            for(int i=tid;i<{4*packed//vector};i+=128) {{
                int row=blockIdx.x*4+i/({packed//vector});int offset=i%({packed//vector})*{vector};
                if(row<{o}) {{ {copy_expression} }}
            }}
            {'__pipeline_commit(); __pipeline_wait_prior(0);' if async_load else ''}
            __syncthreads();
        }}
        if(tile+1<{k//512}) {{
            const int slot=(tile+1)%2;
            for(int i=tid;i<{4*packed//vector};i+=128) {{
                int row=blockIdx.x*4+i/({packed//vector});int offset=i%({packed//vector})*{vector};
                if(row<{o}) {{ {copy_expression.replace('tile*','(tile+1)*')} }}
            }}
            {'__pipeline_commit();' if async_load else ''}
        }}
        if(row<{o}) {{
            const unsigned char* bytes=stage[tile%2]+local_row*{packed};
            #pragma unroll
            for(int pair=0;pair<8;++pair) {{
                int j=pair*64+lane*2;
                float a=packed_weight(bytes+j/{block}*{size},j%{block},{q});
                float b=packed_weight(bytes+(j+1)/{block}*{size},(j+1)%{block},{q});
                sum+=tensor_half2_dot(x[tile*512+j],x[tile*512+j+1],a,b);
            }}
        }}
        {'__pipeline_wait_prior(0);' if async_load else ''}
        __syncthreads();
    }}
    for(int offset=16;offset>0;offset/=2) sum+=__shfl_down_sync(0xffffffff,sum,offset);
    if(lane==0 && row<{o}) out[row]=sum;
}}
'''
    return emit([('x',k,'float32'),('w',o*row_bytes,'uint8'),('out',o,'float32')],f'''with T.Kernel(T.ceildiv({o},4),threads=128,prelude={prelude!r}) as bx:
    T.evaluate(T.call_extern("void", "packed_pipeline", T.address_of(x[0]), T.address_of(w[0]), T.address_of(out[0])))''')



def prefetch_source(p,ahead=8):
    text=linear_source(p,'fp16_half2')
    if p['type'] not in (2,12,14):return text
    from tensor_llm.common.gguf import TYPES
    _,block,size=TYPES[p['type']]
    # Hint only bytes belonging to the upcoming block(s), including scale metadata.
    span=2*size if block==32 else size
    offsets=(0,span-1) if span<=128 else (0,127,span-1)
    instructions='\n'.join(f'    asm volatile("prefetch.global.L2 [%0];" :: "l"(p+{offset}));' for offset in offsets)
    prelude=HALF2+'\n__device__ __forceinline__ void prefetch_packed(const unsigned char* p) {\n'+instructions+'\n}\n'
    text=text.replace(repr(HALF2),repr(prelude))
    marker=f"        for tile in T.serial({p['k']//64}):\n"
    insertion=f'''            if (tile % {max(block//64,1)} == 0) & (tile + {ahead} < {p['k']//64}):
                for row, lane in T.Parallel(4,32):
                    if (lane == 0) & (bx * 4 + row < {p['o']}):
                        T.evaluate(T.call_extern("void", "prefetch_packed", T.address_of(w[((bx * 4 + row) * {p['k']} + (tile + {ahead}) * 64) // {block} * {size}])))
'''
    if marker not in text:raise ValueError('half2 loop template changed')
    return text.replace(marker,marker+insertion)

def partial_source(p,splits):
    if type(splits) is not int or splits<1:raise ValueError('positive split count required')
    h,kh,d,cap=p['h'],p['kh'],p['d'],p['cap'];stride=d+2
    original=reference_attention_source('attention',p)
    # Retain the original online-softmax arithmetic within each partition.
    body=textwrap.dedent(original.split(' as head:\n',1)[1].split('\ndef tensor_export',1)[0]).rstrip()
    body=body.replace('for tile in T.serial(T.ceildiv(pos[0] + 1, 64)):',
        f'partition = T.ceildiv(T.ceildiv(pos[0] + 1, 64), {splits})\nfor local_tile in T.serial(T.max(T.min(partition, T.ceildiv(pos[0] + 1, 64) - split * partition), 0)):\n    tile = split * partition + local_tile')
    body=body.replace('out[head * '+str(d)+' + j] = result[j] / normalizer[0]',f'parts[(head * {splits} + split) * {stride} + j] = result[j]')
    body+=f'\nparts[(head * {splits} + split) * {stride} + {d}] = maximum[0]\nparts[(head * {splits} + split) * {stride} + {d+1}] = normalizer[0]'
    return emit([('q',h*d,'float32'),('kc',cap*kh*d,'float16'),('vc',cap*kh*d,'float16'),('parts',h*splits*stride,'float32'),('pos',2,'int32')],
                f'with T.Kernel({h}, {splits}, threads=128) as (head, split):\n'+'\n'.join('    '+line for line in body.splitlines()))



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


def bundle(base,out,*,projection='unchanged',splits=0):
    if projection not in ('unchanged','pipeline','staged_sync'):raise ValueError('unknown projection schedule')
    if type(splits) is not int or splits<0:raise ValueError('nonnegative splits required')
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=copy.deepcopy(json.loads((base/'inference.json').read_text()))
    for key,record in manifest['kernels'].items():
        dst=out/record['artifact'];dst.parent.mkdir(parents=True,exist_ok=True)
        src=base/record['artifact']
        if projection!='unchanged' and record['kind']=='linear' and record['parameters']['r']==1:
            src=compile_source(pipelined_source(record['parameters'],async_load=projection=='pipeline'),out/'candidates',manifest['target'])
        shutil.copyfile(src,dst);record['sha256']=hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()
    attention=next(r['parameters'] for r in manifest['kernels'].values() if r['kind']=='attention' and r['parameters']['r']==1)
    extra={}
    if splits:
        for kind,text in (('partial',warp_partial_source(attention,splits)),('merge',merge_source(attention,splits))):
            src=compile_source(text,out/'candidates',manifest['target']);dst=out/'artifacts'/('decode-'+kind+'.tbin');shutil.copyfile(src,dst)
            extra[kind]={'artifact':dst.relative_to(out).as_posix(),'sha256':hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()}
    manifest['decode_optimization']={'projection':projection,'splits':splits,'artifacts':extra,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_bundle_sha256':hashlib.sha256((base/'inference.json').read_bytes()).hexdigest()}
    (out/'inference.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest


class OptimizedLFM2(LFM2):
    """Experimental two-kernel attention plan, with one reusable partial buffer."""
    def _plan(self,r):
        plan=super()._plan(r)
        experiment=self.manifest.get('decode_optimization',{})
        splits=experiment.get('splits',0)
        if r!=1 or not splits:return plan
        cfg=self.config
        scratch=self.device.empty((cfg.heads*splits*(cfg.head_dim+2),))
        self.buffers.append(scratch)
        kernels={}
        for name,record in experiment['artifacts'].items():
            path=(self.directory/record['artifact']).resolve()
            if not path.is_relative_to(self.directory.resolve()):raise ValueError('experimental artifact escapes bundle')
            if hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=record['sha256']:raise ValueError('experimental checksum mismatch')
            kernel=self.device.load(path);kernels[name]=kernel;self.kernels['decode-'+name]=kernel
        targets={self.kernels[key] for key,record in self.manifest['kernels'].items() if record['kind']=='attention' and record['parameters']['r']==1}
        replaced=[]
        def bind(name,args):
            kernel=kernels[name];values,symbols,launch=kernel._bind(args,{},include_outputs=True)
            return kernel,BoundCall(self.device,kernel.manifest,values,symbols,launch,validated=True)
        for kernel,call in plan:
            if kernel not in targets:replaced.append((kernel,call));continue
            values=dict(zip((item['name'] for item in kernel.manifest['abi']),call.storage))
            q,kc,vc,out,pos=(values[name] for name in ('q','kc','vc','out','pos'))
            replaced.extend((bind('partial',(q,kc,vc,scratch,pos)),bind('merge',(scratch,out))))
        return replaced


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--projection',choices=('unchanged','pipeline','staged_sync'),default='unchanged')
    p.add_argument('--splits',type=int,default=0)
    a=p.parse_args();bundle(a.base,a.out,projection=a.projection,splits=a.splits)
