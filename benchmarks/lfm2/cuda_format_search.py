"""Bounded CUDA projection search with checkpoint weights and independent math."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.cuda_graph import CudaGraph
from tensor.compiler.tuning import measure_cuda
from tensor_llm import GGUF
from tensor_llm.cuda_kernels import gemv_source, fused_source, prefill_source
from tensor_llm.kernels import source as baseline_source


def current_source(kind,p):
    if kind in ('linear','linear_add','ffn'):
        return gemv_source(kind,p) if p['r']==1 else fused_source(kind,p)
    return baseline_source(kind,p)

def compile_source(text, out):
    key=hashlib.sha256(text.encode()).hexdigest()[:24]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/(key+'.py');artifact=path.with_suffix('.tbin')
    if not path.exists() or path.read_text()!=text:path.write_text(text);artifact.unlink(missing_ok=True)
    if not artifact.exists():tensor.build(path,artifact,compiler='nvrtc',target='sm_86',cache_dir=out/'cache')
    return artifact


PREFILL=[dict(block_m=m,block_n=n,block_k=k,stages=s,threads=t)
         for m,n,k,s,t in ((32,64,64,2,128),(32,128,64,2,128),
                           (64,64,64,2,128),(64,128,64,2,128),(64,128,32,3,128),
                           (64,128,64,2,256),(32,128,64,2,256),(16,64,64,2,128))]
DECODE=[dict(threads=t,unroll=u) for t,u in ((64,4),(128,4),(256,4),(128,8))]
PAIRED=[dict(block_m=m,block_n=n,block_k=k,stages=s,threads=t,packed_pairs=True)
        for m,n,k,s,t in ((32,64,32,2,128),(32,64,64,2,128),(64,64,64,2,128),
                         (32,64,128,2,128),(32,128,32,2,128),(64,128,32,2,256))]
SMALL=[dict(block_m=m,block_n=64,block_k=k,stages=2,threads=128,packed_pairs=packed)
       for m,k,packed in ((16,64,False),(32,64,False),(32,64,True),(32,128,True))]


def search(models, out, phase='prefill', rows=(128,), family='tiles', formats=('F16','Q4_0','Q4_K_M')):
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    models=Path(models);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cases=[];seen=set();model_hashes={}
    for fmt in formats:
        path=models/f'LFM2.5-2.6B-{fmt}.gguf';gguf=GGUF(path)
        with path.open('rb') as stream:model_hashes[path.name]=hashlib.file_digest(stream,'sha256').hexdigest()
        for name,info in gguf.tensors.items():
            if len(info.shape)!=2 or name.endswith('conv.weight'):continue
            o,k=info.shape;q=info.type
            if phase=='prefill' and name=='token_embd.weight':continue
            kind='ffn' if name.endswith('ffn_gate.weight') else 'linear'
            if name.endswith('ffn_up.weight'):continue
            for r in rows if phase=='prefill' else (1,):
                key=(kind,r,k,o,q)
                if key in seen:continue
                seen.add(key);parameters=dict(r=r,k=k,o=o,type=q)
                configs=({'paired-loads':PAIRED,'small-rows':SMALL}.get(family,PREFILL)) if phase=='prefill' else DECODE
                if phase=='decode' and q==1:
                    configs=[*configs,dict(threads=128,unroll=4,f16_values=8),dict(threads=64,unroll=4,f16_values=8),dict(threads=128,unroll=4,f16_values=16)]
                candidates={'before':(None,current_source(kind,parameters))}
                for index,config in enumerate(configs):
                    p={**parameters,**config}
                    text=prefill_source(kind,p) if phase=='prefill' else current_source(kind,p)
                    candidates[f'candidate-{index}']=(config,text)
                if phase=='decode' and kind=='ffn' and q==1:
                    candidates['separate']=(dict(paired=False),current_source('linear',parameters))
                rng=np.random.default_rng(2093)
                x=rng.normal(size=(r,k)).astype(np.float32)*.3
                decoded=torch.from_numpy(np.array(gguf.array(name),copy=True)).cuda()
                tx=torch.from_numpy(x).cuda()
                expected=torch.mm(tx.half(),decoded.half().T,out_dtype=torch.float32) if r>1 else torch.mm(tx,decoded.T)
                arrays=[x.ravel(),np.array(gguf.packed(name),copy=True)]
                if q in (0,1):arrays[1]=arrays[1].view(np.float32 if q==0 else np.float16)
                if kind=='ffn':
                    upname=name.replace('ffn_gate','ffn_up')
                    up=torch.from_numpy(np.array(gguf.array(upname),copy=True)).cuda()
                    updot=torch.mm(tx.half(),up.half().T,out_dtype=torch.float32) if r>1 else torch.mm(tx,up.T)
                    expected=torch.nn.functional.silu(expected)*updot
                    packed=np.array(gguf.packed(upname),copy=True)
                    arrays.append(packed.view(np.float16 if q==1 else np.float32) if q in (0,1) else packed)
                    del up,updot
                expected=expected.cpu().numpy().ravel()
                del decoded,tx;torch.cuda.empty_cache()
                row=dict(model=path.name,tensor=name,kind=kind,parameters=parameters,candidates={})
                with tensor.Device() as device:
                    buffers=[device.from_numpy(a) for a in arrays];output=device.empty(r*o)
                    for label,(config,text) in candidates.items():
                        try:
                            artifact=compile_source(text,out/'kernels');kernel=device.load(artifact)
                            if label=='separate':
                                gate=device.empty(o);up=device.empty(o)
                                swiglu=device.load(compile_source(current_source('swiglu',dict(r=1,c=o)),out/'kernels'))
                                def launch():
                                    kernel.launch(buffers[0],buffers[1],gate)
                                    kernel.launch(buffers[0],buffers[2],up)
                                    swiglu.launch(gate,up,output)
                            else:
                                def launch():kernel.launch(*buffers,output)
                            launch();actual=output.to_numpy()
                            relative=float(np.linalg.norm(actual.astype(np.float64)-expected)/np.linalg.norm(expected))
                            assert np.isfinite(actual).all() and relative<.002,(label,relative)
                            def batch():
                                for _ in range(20):launch()
                            with CudaGraph(device,batch,resources=(*buffers,output,kernel)) as graph:
                                timing=measure_cuda(device,graph.launch,warmup=3,samples=5,repeats=1)
                            for metric in ('gpu_seconds','completed_seconds'):timing[metric]=[x/20 for x in timing[metric]]
                            for metric in ('median_gpu_seconds','median_completed_seconds'):timing[metric]/=20
                            timing['graph_nodes']=20
                            row['candidates'][label]=dict(status='passed',config=config,relative_rms=relative,timing=timing,
                                                          source_sha256=hashlib.sha256(text.encode()).hexdigest())
                            print(key,label,round(timing['median_gpu_seconds']*1e6,2),'us',flush=True)
                            kernel.release()
                            if label=='separate':swiglu.release();gate.release();up.release()
                        except Exception as exc:
                            row['candidates'][label]=dict(status='rejected',config=config,reason=str(exc))
                            print(key,label,'rejected',str(exc)[:160],flush=True)
                    passed=[n for n,v in row['candidates'].items() if v['status']=='passed']
                    assert passed,row
                    row['selected']=min(passed,key=lambda n:row['candidates'][n]['timing']['median_gpu_seconds'])
                    adapter=device.info
                cases.append(row)
                report=dict(schema='tensor.lfm2-cuda-format-search.v1',phase=phase,family=family,
                            adapter=adapter,cases=cases,model_sha256=model_hashes,
                            protocol=dict(seed=2093,gpu_concurrency=1,warmups=3,samples=5,
                                          graph_repetitions=20,times_divided_by=20,relative_rms_max=.002,
                                          baseline='generic packed schedule, not the frozen full-model before profile',
                                          reference='independent Torch FP16 prefill/FP32 decode operators'))
                (out/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    return cases


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,default=Path('build/lfm2-models'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=('prefill','decode'),default='prefill')
    p.add_argument('--rows',type=int,nargs='+',default=[128])
    p.add_argument('--family',choices=('tiles','paired-loads','small-rows'),default='tiles')
    p.add_argument('--formats',nargs='+',choices=('F16','Q4_0','Q4_K_M'),default=['F16','Q4_0','Q4_K_M'])
    args=p.parse_args();search(args.models,args.out,args.phase,tuple(args.rows),args.family,tuple(args.formats))
