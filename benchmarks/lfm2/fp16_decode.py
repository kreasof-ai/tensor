"""Experimental packed-weight decode schedules; production kernels stay unchanged.

All candidates consume the same FP32 buffers and packed GGML weights, and return
FP32 outputs. fp16_cast rounds operands then uses FP32 multiplication;
fp16_half2 multiplies two FP16 values together (FP16 product rounding) and sums
into FP32; fp16_mma32 uses a padded 32-row FP16 tensor-core tile/FP32 accumulator.
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import tensor
from tensor_llm import GGUF
from tensor_llm.kernels import source
from benchmarks.lfm2.text_helpers import emit, weight
from tensor.providers.cuda_graph import CudaGraph
from tensor.compiler.tuning import measure_cuda
from tensor.artifacts.format import read_artifact

MODES=('fp32','fp16_cast','fp16_half2','fp16_mma32')
HALF2=r'''#include <cuda_fp16.h>
__device__ __forceinline__ float tensor_half2_dot(float x0,float x1,float w0,float w1) {
    const __half2 x=__floats2half2_rn(x0,x1);
    const __half2 w=__floats2half2_rn(w0,w1);
    const float2 product=__half22float2(__hmul2(x,w));
    return product.x+product.y;
}
'''


def linear_source(parameters,mode):
    if mode not in MODES:raise ValueError('unknown FP16 decode experiment')
    if mode=='fp32':return source('linear',parameters)
    if parameters['r']!=1:raise ValueError('decode candidates require one row')
    k,o,q=parameters['k'],parameters['o'],parameters['type']
    if mode=='fp16_half2' and k%64:raise ValueError('half2 requires K divisible by 64')
    from tensor_llm.gguf import TYPES
    _,block,size=TYPES[q]
    dtype='float16' if q==1 else 'float32' if q==0 else 'uint8'
    args=[('x',k,'float32'),('w',k*o if q in (0,1) else k*o//block*size,dtype),('out',o,'float32')]
    if mode=='fp16_mma32':
        return emit(args,f'''with T.Kernel(1, T.ceildiv({o}, 64), threads=128) as (by, bx):
    lhs = T.alloc_shared((32, 32), "float16")
    rhs = T.alloc_shared((64, 32), "float16")
    accum = T.alloc_fragment((32, 64), "float32")
    T.clear(accum)
    for tile in T.Pipelined({k//32}, num_stages=2):
        for i, j in T.Parallel(32, 32):
            lhs[i, j] = T.if_then_else(i == 0, x[tile * 32 + j], 0)
        for i, j in T.Parallel(64, 32):
            rhs[i, j] = T.if_then_else(bx * 64 + i < {o}, {weight(q,'bx * 64 + i','tile * 32 + j',k)}, 0)
        T.gemm(lhs, rhs, accum, transpose_B=True)
    for i, j in T.Parallel(32, 64):
        if (i == 0) & (bx * 64 + j < {o}):
            out[bx * 64 + j] = accum[i, j]''')
    if mode=='fp16_cast':
        col='tile * 32 + lane'
        expression=f'T.cast(T.cast(x[{col}], "float16"), "float32") * T.cast(T.cast(({weight(q,"bx * 4 + row",col,k)}), "float16"), "float32")'
        tiles=k//32;prelude=''
    else:
        col0='tile * 64 + lane * 2';col1=col0+' + 1'
        expression=f'T.call_extern("float32", "tensor_half2_dot", x[{col0}], x[{col1}], {weight(q,"bx * 4 + row",col0,k)}, {weight(q,"bx * 4 + row",col1,k)})'
        tiles=k//64;prelude=', prelude='+repr(HALF2)
    return emit(args,f'''with T.Kernel(T.ceildiv({o}, 4), threads=128{prelude}) as bx:
    accum = T.alloc_fragment((4, 32), "float32")
    total = T.alloc_fragment((4,), "float32")
    T.clear(accum)
    for tile in T.serial({tiles}):
        for row, lane in T.Parallel(4, 32):
            accum[row, lane] += {expression}
    T.reduce_sum(accum, total, dim=1)
    for row in T.Parallel(4):
        if bx * 4 + row < {o}:
            out[bx * 4 + row] = total[row]''')


def compile_candidate(parameters,mode,out,target='sm_86'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    text=linear_source(parameters,mode);key=hashlib.sha256(text.encode()).hexdigest()[:24]
    src=out/(key+'.py');artifact=out/(key+'.tbin')
    if not src.exists() or src.read_text()!=text:src.write_text(text);artifact.unlink(missing_ok=True)
    if artifact.exists():
        manifest,_=read_artifact(artifact)
        if manifest['target']!=target:artifact.unlink()
    if not artifact.exists():tensor.build(src,artifact,compiler='nvrtc',target=target,cache_dir=out/'compiler-cache')
    return artifact


def bundle(base,out,mode):
    """Overlay the linear specialization binaries of an existing verified plan."""
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    manifest=copy.deepcopy(json.loads((base/'inference.json').read_text()))
    for key,record in manifest['kernels'].items():
        if record['kind']=='linear' and record['parameters']['r']==1:
            src=compile_candidate(record['parameters'],mode,out/'candidates',manifest['target'])
            dst=out/'artifacts'/(key+'.tbin');dst.parent.mkdir(exist_ok=True);shutil.copyfile(src,dst)
            record['artifact']=dst.relative_to(out).as_posix()
            record['sha256']=hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()
        else:
            dst=out/record['artifact'];dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(base/record['artifact'],dst)
    manifest['experiment']={'name':'packed-fp16-decode','mode':mode,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'base_bundle_sha256':hashlib.sha256((base/'inference.json').read_bytes()).hexdigest(),
        'note':'all r=1 projections, including the final output projection during prefill'}
    (out/'inference.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def microbench(models,out):
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for model,name in ((models/'LFM2.5-2.6B-Q4_0.gguf','blk.0.ffn_gate.weight'),
                       (models/'LFM2.5-2.6B-Q4_K_M.gguf','blk.0.ffn_gate.weight'),
                       (models/'LFM2.5-2.6B-Q4_0.gguf','token_embd.weight')):
        g=GGUF(model);info=g.tensors[name];o,k=info.shape;p={'r':1,'k':k,'o':o,'type':info.type}
        paths={mode:compile_candidate(p,mode,out/'kernels') for mode in MODES}
        x=np.random.default_rng(101).standard_normal(k).astype(np.float32)
        decoded=torch.from_numpy(np.array(g.array(name),copy=True)).cuda();tx=torch.from_numpy(x).cuda()
        expected_fp32=(tx[None] @ decoded.T)[0].cpu().numpy()
        expected_fp16=torch.mm(tx[None].half(),decoded.half().T,out_dtype=torch.float32)[0].cpu().numpy()
        expected_half2=(tx.half()[None,:]*decoded.half()).sum(-1,dtype=torch.float32).cpu().numpy()
        del decoded,tx;torch.cuda.empty_cache()
        row={'model':model.name,'tensor':name,'encoding':info.encoding,'shape':info.shape,'candidates':{}}
        with tensor.Device() as device:
            dx=device.from_numpy(x);dw=device.from_numpy(g.packed(name));dy=device.empty((o,))
            for mode,path in paths.items():
                kernel=device.load(path);kernel.launch(dx,dw,dy);actual=dy.to_numpy()
                expected=expected_fp32 if mode=='fp32' else expected_half2 if mode=='fp16_half2' else expected_fp16
                relative=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
                if not np.isfinite(actual).all() or relative>=.001:raise AssertionError((mode,info.encoding,relative))
                with CudaGraph(device,lambda:kernel.launch(dx,dw,dy),resources=(dx,dw,dy,kernel)) as graph:
                    timing=measure_cuda(device,graph.launch,warmup=5,samples=7,repeats=30)
                row['candidates'][mode]={'relative_rms':relative,'max_error':float(np.max(np.abs(actual-expected))),
                    'artifact_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'timing':timing}
                kernel.release();print(info.encoding,mode,relative,timing['median_gpu_seconds']*1e6,'us',flush=True)
            adapter=device.info
        cases.append(row)
    report={'schema':'tensor.lfm2-fp16-micro.v1','status':'passed','adapter':adapter,'cases':cases,
        'protocol':'isolated GEMV; packed checkpoint weights; random FP32 input seed 101; CUDA graph/event; 7 samples of 30 replays after 5 warmups; no transfer in timing',
        'precision':{'fp32':'FP32 operands/multiply/accumulation','fp16_cast':'FP16 operands converted to FP32 multiply/accumulation',
        'fp16_half2':'FP16 operands/product, FP32 pair sum and accumulation','fp16_mma32':'FP16 operands, tensor-core multiplication with FP32 accumulation; padded M=32'}}
    (out/'micro.json').write_text(json.dumps(report,indent=2)+'\n');return report


def metrics(actual,expected):
    a=actual.astype(np.float64);e=expected.astype(np.float64)
    return {'relative_rms':float(np.linalg.norm(a-e)/np.linalg.norm(e)),
            'cosine':float(np.dot(a,e)/(np.linalg.norm(a)*np.linalg.norm(e))),
            'max_error':float(np.max(np.abs(a-e))),
            'top1_equal':bool(np.argmax(actual)==np.argmax(expected))}


def validate_variant(model,plan,fixture,out,mode):
    """23 real-model steps against independent precision-specific Torch math."""
    from tensor_llm import LFM2
    from benchmarks.lfm2.torch_reference import Reference
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'validation.json').unlink(missing_ok=True)
    manifest=json.loads((plan/'inference.json').read_text())
    if manifest.get('experiment',{}).get('mode')!=mode:raise ValueError('bundle uses a different experiment mode')
    items=json.loads((fixture/'reference-spec.json').read_text())['validation']
    baseline_report=json.loads((fixture/'validation.json').read_text())
    if baseline_report['status']!='passed':raise ValueError('requires a passed baseline fixture')
    model_hash=hashlib.file_digest(model.open('rb'),'sha256').hexdigest()
    if model_hash!=baseline_report['model_sha256']:raise ValueError('checkpoint differs from fixture')
    saved=[];observations=[]
    with tensor.Device() as device,LFM2(model,plan,device) as network:
        for i,item in enumerate(items):
            if item.get('reset'):network.reset()
            actual=network.forward(item['tokens'])
            if not np.isfinite(actual).all():raise AssertionError('non-finite variant logits')
            np.save(out/f'tensor-{i}-logits.npy',actual);saved.append(actual)
            original=np.load(fixture/f'tensor-{i}-logits.npy')
            native=np.fromfile(fixture/'llama'/f'{i}-logits.bin',np.float32)
            observations.append({'step':i,'position':network.position,
                'original_fp32':metrics(actual,original),'llama_cpp':metrics(actual,native)})
        np.testing.assert_array_equal(saved[0],saved[-1])
        adapter=device.info;allocated=network.allocated_bytes
    reference=Reference(model,decode_mode='half2' if mode=='fp16_half2' else 'fp16')
    for i,item in enumerate(items):
        if item.get('reset'):reference.reset()
        for start in range(0,len(item['tokens']),128):expected=reference.forward(item['tokens'][start:start+128])
        values=metrics(saved[i],expected);observations[i]['independent_reference']=values
        print(model.name,mode,'validate',i,values,flush=True)
        assert values['relative_rms']<.01 and values['cosine']>.9999,values
        assert observations[i]['original_fp32']['relative_rms']<.01,observations[i]
    report={'schema':'tensor.lfm2-fp16-validation.v1','status':'passed','mode':mode,
        'model_sha256':model_hash,'bundle_sha256':hashlib.sha256((plan/'inference.json').read_bytes()).hexdigest(),
        'adapter':adapter,'owned_device_bytes':allocated,'steps':observations,
        'gates':{'finite':True,'reset_bitwise_equal':True,'independent_relative_rms_max':.01,
                 'independent_cosine_min':.9999,'relative_rms_vs_original_max':.01}}
    shutil.copyfile(fixture/'reference-spec.json',out/'reference-spec.json')
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');return report


def paired_benchmark(model,base,candidate,out,*,depths=(128,512,2048,8192),generated=256,repeats=5):
    """Alternate two full-model CUDA graph engines on identical tokens/weights."""
    import time
    from tensor_llm import LFM2,Tokenizer
    if generated<1 or repeats<1 or not depths or min(depths)<1:raise ValueError('positive benchmark counts required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'paired.json').unlink(missing_ok=True)
    cached_torch=_sys.modules.get('torch')
    if cached_torch is not None:cached_torch.cuda.empty_cache()
    if json.loads((candidate/'inference.json').read_text()).get('experiment',{}).get('mode')!='fp16_half2':
        raise ValueError('paired benchmark requires a half2 candidate')
    tokenizer=Tokenizer(GGUF(model).metadata)
    pattern=tokenizer.encode('Tensor evaluates a fixed sequence to compare cached language model inference. The same tokens are used by both engines. ',add_bos=False)
    cases=[]
    with tensor.Device() as device,LFM2(model,base,device,context=max(depths)+generated) as fp32,LFM2(model,candidate,device,context=max(depths)+generated) as fp16:
        assert fp32.allocated_bytes==fp16.allocated_bytes
        for depth in depths:
            prompt=[tokenizer.bos]+[pattern[i%len(pattern)] for i in range(depth-1)]
            decode=[pattern[i%len(pattern)] for i in range(generated)]
            samples={'fp32':[],'fp16_half2':[]}
            for repeat in range(-1,repeats):
                order=(('fp32',fp32),('fp16_half2',fp16))
                if repeat%2:order=order[::-1]
                for name,network in order:
                    network.reset();begin=time.perf_counter();network.forward(prompt);prefill=time.perf_counter()-begin
                    latencies=[];begin=time.perf_counter()
                    for token in decode:
                        step=time.perf_counter();network.forward([token]);latencies.append(time.perf_counter()-step)
                    total=time.perf_counter()-begin
                    if repeat>=0:samples[name].append({'prefill_seconds':prefill,'decode_seconds':total,'decode_latencies':latencies})
            medians={name:float(np.median([s['decode_seconds'] for s in values])) for name,values in samples.items()}
            row={'prompt_tokens':depth,'decode_tokens':generated,'samples':samples,
                'fp32_tokens_per_second':generated/medians['fp32'],'fp16_half2_tokens_per_second':generated/medians['fp16_half2'],
                'fp32_milliseconds':medians['fp32']/generated*1000,'fp16_half2_milliseconds':medians['fp16_half2']/generated*1000,
                'speedup':medians['fp32']/medians['fp16_half2']}
            print(model.name,{k:v for k,v in row.items() if k!='samples'},flush=True);cases.append(row)
        report={'schema':'tensor.lfm2-fp16-paired.v1','status':'passed','model':model.name,'adapter':device.info,
            'mode':'fp16_half2','model_sha256':hashlib.file_digest(model.open('rb'),'sha256').hexdigest(),
            'base_bundle_sha256':hashlib.sha256((base/'inference.json').read_bytes()).hexdigest(),
            'candidate_bundle_sha256':hashlib.sha256((candidate/'inference.json').read_bytes()).hexdigest(),
            'owned_device_bytes_per_engine':fp32.allocated_bytes,'launches_per_decode':len(fp32.plans[1]),
            'protocol':'same GGUF and token IDs; two resident engines run sequentially in alternating order; 1 excluded warmup and 5 repeats; prompt chunks <=128; 256 forced tokens; last-token logits on host; excludes loading/reset/tokenization/sampling',
            'cases':cases}
    (out/'paired.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    m=commands.add_parser('micro');m.add_argument('--models',type=Path,default=Path('build/lfm2-models'));m.add_argument('--out',type=Path,required=True)
    b=commands.add_parser('bundle');b.add_argument('--base',type=Path,required=True);b.add_argument('--out',type=Path,required=True);b.add_argument('--mode',choices=MODES,required=True)
    v=commands.add_parser('validate')
    for name in ('model','plan','fixture','out'):v.add_argument('--'+name,type=Path,required=True)
    v.add_argument('--mode',choices=MODES[1:],required=True)
    bench=commands.add_parser('bench')
    for name in ('model','base','candidate','out'):bench.add_argument('--'+name,type=Path,required=True)
    bench.add_argument('--depths',type=int,nargs='+',default=[128,512,2048,8192]);bench.add_argument('--generated',type=int,default=256);bench.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.command=='micro':microbench(args.models,args.out)
    elif args.command=='bundle':bundle(args.base,args.out,args.mode)
    elif args.command=='validate':validate_variant(args.model,args.plan,args.fixture,args.out,args.mode)
    else:paired_benchmark(args.model,args.base,args.candidate,args.out,depths=args.depths,generated=args.generated,repeats=args.repeats)
