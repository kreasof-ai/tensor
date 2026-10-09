"""Numerical and timing controls for experimental LFM2 decode schedules."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import tensor
from tensor_llm import GGUF,LFM2,Tokenizer
from tensor.providers.cuda_graph import CudaGraph
from tensor.compiler.tuning import measure_cuda
from benchmarks.lfm2.decode_optimization import compile_source,pipelined_source,prefetch_source,partial_source,warp_partial_source,merge_source,OptimizedLFM2
from benchmarks.lfm2.fp16_decode import linear_source,metrics
from tensor_llm.lfm2.kernels.baseline import source


def micro_attention(out,*,implementation='warp'):
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p={'r':1,'h':32,'kh':8,'d':64,'cap':8576}
    if implementation not in ('warp','fragment'):raise ValueError('unknown attention implementation')
    generator=warp_partial_source if implementation=='warp' else partial_source
    serial=compile_source(source('attention',p),out/'kernels')
    paths={s:(compile_source(generator(p,s),out/'kernels'),compile_source(merge_source(p,s),out/'kernels')) for s in (4,8,16,32)}
    rng=np.random.default_rng(102)
    q=rng.standard_normal((32,64)).astype(np.float32)
    k=rng.standard_normal((8576,8,64)).astype(np.float16)
    v=rng.standard_normal((8576,8,64)).astype(np.float16)
    cases=[]
    with tensor.Device() as device:
        dq=device.from_numpy(q.ravel());dk=device.from_numpy(k.ravel());dv=device.from_numpy(v.ravel());dy=device.empty((32*64,))
        pos=device.from_numpy(np.array([0,1],np.int32))
        baseline=device.load(serial)
        kernels={s:(device.load(a),device.load(b)) for s,(a,b) in paths.items()}
        # Split-specific scratch ABI requires buffers with their exact declared extent.
        scratch={s:device.empty((32*s*66,)) for s in paths}
        tq=torch.from_numpy(q).cuda();tk=torch.from_numpy(k).cuda().float();tv=torch.from_numpy(v).cuda().float()
        references={}
        for length in (1,64,65,128,129,512,2048,8192,8193):
            keys=tk[:length].repeat_interleave(4,dim=1).permute(1,0,2)
            values=tv[:length].repeat_interleave(4,dim=1).permute(1,0,2)
            score=(keys*tq[:,None,:]).sum(-1)*.125
            references[length]=(torch.softmax(score,dim=-1)[:,:,None]*values).sum(1).cpu().numpy().ravel()
        del tq,tk,tv,keys,values,score;torch.cuda.empty_cache()
        for length,expected in references.items():
            host=np.array([length-1,1],np.int32);device.driver.call('cuMemcpyHtoD_v2',pos.pointer,host.ctypes.data,host.nbytes)
            candidates={'serial':(lambda:baseline.launch(dq,dk,dv,dy,pos),(dq,dk,dv,dy,pos,baseline))}
            for s,(a,b) in kernels.items():
                def launch(a=a,b=b,part=scratch[s]):a.launch(dq,dk,dv,part,pos);b.launch(part,dy)
                candidates['split'+str(s)]=(launch,(dq,dk,dv,dy,pos,scratch[s],a,b))
            row={'context_tokens':length,'candidates':{}}
            for name,(launch,resources) in candidates.items():
                launch();actual=dy.to_numpy();error=metrics(actual,expected)
                assert np.isfinite(actual).all() and error['relative_rms']<1e-5,(length,name,error)
                with CudaGraph(device,launch,resources=resources) as graph:
                    timing=measure_cuda(device,graph.launch,warmup=5,samples=7,repeats=30)
                row['candidates'][name]={'error':error,'timing':timing}
                print('attention',length,name,error['relative_rms'],timing['median_gpu_seconds']*1e6,'us',flush=True)
            cases.append(row)
        adapter=device.info
    report={'schema':'tensor.lfm2-decode-attention-micro.v1','status':'passed','adapter':adapter,'parameters':p,'implementation':implementation,
        'artifacts':{'serial':hashlib.file_digest(serial.open('rb'),'sha256').hexdigest(),
            'split':{str(s):{'partial':hashlib.file_digest(a.open('rb'),'sha256').hexdigest(),'merge':hashlib.file_digest(b.open('rb'),'sha256').hexdigest()} for s,(a,b) in paths.items()}},
        'protocol':'FP32 queries, FP16 K/V; independent Torch softmax; CUDA graphs/events; 7 samples x30 replays, 5 warmups; transfers/reference excluded; seed102; serial and two-kernel split controls',
        'cases':cases}
    (out/'attention.json').write_text(json.dumps(report,indent=2)+'\n');return report


def micro_projection(models,out):
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=True);cases=[]
    for filename,name in (('Q4_0','blk.0.ffn_gate.weight'),('Q4_K_M','blk.0.ffn_gate.weight'),('Q4_0','token_embd.weight'),('Q4_K_M','blk.0.ffn_down.weight')):
        g=GGUF(models/f'LFM2.5-2.6B-{filename}.gguf');info=g.tensors[name];o,k=info.shape;p={'r':1,'k':k,'o':o,'type':info.type}
        paths={label:compile_source(text,out/'kernels') for label,text in (
            ('half2',linear_source(p,'fp16_half2')),('staged_sync',pipelined_source(p,async_load=False)),('pipeline',pipelined_source(p)),('prefetch4',prefetch_source(p,4)),('prefetch8',prefetch_source(p,8)),('prefetch16',prefetch_source(p,16)))}
        x=np.random.default_rng(101).standard_normal(k).astype(np.float32)
        decoded=torch.from_numpy(np.array(g.array(name),copy=True)).cuda();tx=torch.from_numpy(x).cuda()
        expected=(tx.half()[None,:]*decoded.half()).sum(-1,dtype=torch.float32).cpu().numpy()
        del decoded,tx;torch.cuda.empty_cache()
        row={'model':filename,'tensor':name,'encoding':info.encoding,'shape':info.shape,'candidates':{}}
        with tensor.Device() as device:
            dx=device.from_numpy(x);dw=device.from_numpy(g.packed(name));dy=device.empty((o,))
            for label,path in paths.items():
                kernel=device.load(path);kernel.launch(dx,dw,dy);actual=dy.to_numpy();error=metrics(actual,expected)
                assert np.isfinite(actual).all() and error['relative_rms']<.001,(label,error)
                with CudaGraph(device,lambda:kernel.launch(dx,dw,dy),resources=(dx,dw,dy,kernel)) as graph:
                    timing=measure_cuda(device,graph.launch,warmup=5,samples=7,repeats=30)
                row['candidates'][label]={'error':error,'artifact_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'timing':timing}
                print('projection',info.encoding,name,label,error['relative_rms'],timing['median_gpu_seconds']*1e6,'us',flush=True)
                kernel.release()
            adapter=device.info
        cases.append(row)
    report={'schema':'tensor.lfm2-decode-projection-micro.v1','status':'passed','adapter':adapter,'cases':cases,
        'protocol':'packed checkpoint matrices; half2 product rounding, FP32 accumulation; independent Torch; CUDA graphs/events; 7 samples x30replays, 5warmups; seed101; transfers/reference excluded; same packed staging for synchronous/asynchronous controls'}
    (out/'projection.json').write_text(json.dumps(report,indent=2)+'\n');return report


def validate(model,plan,fixture,out):
    from benchmarks.lfm2.torch_reference import Reference
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'validation.json').unlink(missing_ok=True)
    manifest=json.loads((plan/'inference.json').read_text());half2=manifest.get('experiment',{}).get('mode')=='fp16_half2' or manifest.get('decode_optimization',{}).get('projection','unchanged')!='unchanged'
    baseline=json.loads((fixture/'validation.json').read_text());model_sha=hashlib.file_digest(model.open('rb'),'sha256').hexdigest()
    assert baseline['status']=='passed' and model_sha==baseline['model_sha256']
    items=json.loads((fixture/'reference-spec.json').read_text())['validation'];observations=[];saved=[]
    with tensor.Device() as device,OptimizedLFM2(model,plan,device) as network:
        for i,item in enumerate(items):
            if item.get('reset'):network.reset()
            actual=network.forward(item['tokens']);assert np.isfinite(actual).all()
            np.save(out/f'tensor-{i}-logits.npy',actual);saved.append(actual)
            original=np.load(fixture/f'tensor-{i}-logits.npy');native=np.fromfile(fixture/'llama'/f'{i}-logits.bin',np.float32)
            observations.append({'step':i,'position':network.position,'original_fp32':metrics(actual,original),'llama_cpp':metrics(actual,native)})
        np.testing.assert_array_equal(saved[0],saved[-1]);allocated=network.allocated_bytes;adapter=device.info;nodes=len(network.plans[1])
    ref=Reference(model,decode_mode='half2' if half2 else 'fp32')
    for i,item in enumerate(items):
        if item.get('reset'):ref.reset()
        for start in range(0,len(item['tokens']),128):expected=ref.forward(item['tokens'][start:start+128])
        error=metrics(saved[i],expected);observations[i]['independent_reference']=error
        assert error['relative_rms']<.01 and error['cosine']>.9999,(i,error)
        assert observations[i]['original_fp32']['relative_rms']<.01,(i,observations[i])
        print('validate',model.name,i,error,flush=True)
    del ref
    shutil.copyfile(fixture/'reference-spec.json',out/'reference-spec.json')
    report={'schema':'tensor.lfm2-decode-optimization-validation.v1','status':'passed','model_sha256':model_sha,
        'bundle_sha256':hashlib.sha256((plan/'inference.json').read_bytes()).hexdigest(),'adapter':adapter,'owned_device_bytes':allocated,
        'launches_per_decode':nodes,'gates':{'finite':True,'reset_bitwise_equal':True,'independent_relative_rms_max':.01,'independent_cosine_min':.9999,'relative_rms_vs_original_max':.01},'steps':observations}
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n');return report


def benchmark(model,plans,out,*,depths=(128,512,2048,8192),generated=256,repeats=5):
    if not plans or generated<1 or repeats<1 or not depths or min(depths)<1:raise ValueError('positive counts and plans required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True);(out/'benchmark.json').unlink(missing_ok=True)
    cached_torch=_sys.modules.get('torch')
    if cached_torch is not None:cached_torch.cuda.empty_cache()
    import contextlib
    tokenizer=Tokenizer(GGUF(model).metadata)
    pattern=tokenizer.encode('Tensor evaluates a fixed sequence to compare cached language model inference. The same tokens are used by both engines. ',add_bos=False)
    cases=[]
    with contextlib.ExitStack() as stack:
        device=stack.enter_context(tensor.Device())
        networks={name:stack.enter_context(OptimizedLFM2(model,path,device,context=max(depths)+generated)) for name,path in plans.items()}
        for depth in depths:
            prompt=[tokenizer.bos]+[pattern[i%len(pattern)] for i in range(depth-1)];decode=[pattern[i%len(pattern)] for i in range(generated)]
            samples={name:[] for name in networks}
            for repeat in range(-1,repeats):
                order=list(networks.items());rotation=(repeat+1)%len(order);order=order[rotation:]+order[:rotation]
                if (repeat+1)//len(order)%2:order.reverse()
                for name,network in order:
                    network.reset();start=time.perf_counter();network.forward(prompt);prefill=time.perf_counter()-start
                    latencies=[];start=time.perf_counter()
                    for token in decode:
                        step=time.perf_counter();network.forward([token]);latencies.append(time.perf_counter()-step)
                    total=time.perf_counter()-start
                    if repeat>=0:samples[name].append({'prefill_seconds':prefill,'decode_seconds':total,'decode_latencies':latencies})
            medians={name:float(np.median([s['decode_seconds'] for s in values])) for name,values in samples.items()}
            row={'prompt_tokens':depth,'decode_tokens':generated,'samples':samples,
                'tokens_per_second':{name:generated/seconds for name,seconds in medians.items()},
                'milliseconds_per_token':{name:seconds/generated*1000 for name,seconds in medians.items()}}
            print('benchmark',model.name,{k:v for k,v in row.items() if k!='samples'},flush=True);cases.append(row)
        report={'schema':'tensor.lfm2-decode-optimization-benchmark.v1','status':'passed','model':model.name,'adapter':device.info,
            'model_sha256':hashlib.file_digest(model.open('rb'),'sha256').hexdigest(),
            'plans':{name:{'bundle_sha256':hashlib.sha256((path/'inference.json').read_bytes()).hexdigest(),'owned_device_bytes':networks[name].allocated_bytes,'launches_per_decode':len(networks[name].plans[1])} for name,path in plans.items()},
            'protocol':{'same_checkpoint_and_tokens':True,'engines_resident_and_sequential':True,'rotating_order':True,'warmup':1,'repeats':repeats,'decode_tokens':generated,'prompt_chunk_max':128,'host_visible_logits':True,'excluded':['loading','reset','tokenization','sampling']},'cases':cases}
    (out/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n');return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('attention');a.add_argument('--out',type=Path,required=True);a.add_argument('--implementation',choices=('warp','fragment'),default='warp')
    a=sub.add_parser('projection');a.add_argument('--out',type=Path,required=True);a.add_argument('--models',type=Path,default=Path('build/lfm2-models'))
    a=sub.add_parser('validate')
    for name in ('model','plan','fixture','out'):a.add_argument('--'+name,type=Path,required=True)
    a=sub.add_parser('bench');a.add_argument('--model',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--plan',action='append',required=True,help='label=directory');a.add_argument('--depths',type=int,nargs='+',default=[128,512,2048,8192]);a.add_argument('--generated',type=int,default=256);a.add_argument('--repeats',type=int,default=5)
    a=sub.add_parser('compare')
    for name in ('model','bundle','reference','out'):a.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    if args.command=='attention':micro_attention(args.out,implementation=args.implementation)
    elif args.command=='projection':micro_projection(args.models,args.out)
    elif args.command=='validate':validate(args.model,args.plan,args.fixture,args.out)
    elif args.command=='compare':
        from benchmarks.lfm2.benchmark import benchmark as matched
        matched(args.model,args.bundle,args.reference,args.out,engine_cls=OptimizedLFM2)
    else:benchmark(args.model,{name:Path(path) for name,path in (item.split('=',1) for item in args.plan)},args.out,depths=args.depths,generated=args.generated,repeats=args.repeats)
