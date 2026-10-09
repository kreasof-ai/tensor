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
from tensor.compiler.search import ScheduleSearch,ScheduleProfile
from tensor.compiler.cuda_schedules import projection_space,projection_legal
from tensor.artifacts.format import read_artifact
from tensor_llm import GGUF
from tensor_llm.lfm2.kernels.cuda import gemv_source, fused_source, prefill_source
from tensor_llm.lfm2.kernels.baseline import source as baseline_source


def current_source(kind,p):
    if kind in ('linear','linear_add','ffn'):
        return gemv_source(kind,p) if p['r']==1 else fused_source(kind,p)
    return baseline_source(kind,p)

def compile_source(text, out):
    key=hashlib.sha256(text.encode()).hexdigest()[:24]
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    path=out/(key+'.py');artifact=path.with_suffix('.tbin')
    if not path.exists() or path.read_text()!=text:path.write_text(text);artifact.unlink(missing_ok=True)
    if artifact.exists():
        manifest,_=read_artifact(artifact)
        expected=hashlib.sha256(Path(tensor.__file__).parent.joinpath('compiler/cuda_lowering.py').read_bytes()).hexdigest()
        if manifest['compiler'].get('lowering_sha256')!=expected:artifact.unlink()
    if not artifact.exists():tensor.build(path,artifact,compiler='nvrtc',target='sm_86',cache_dir=out/'cache')
    return artifact


def search(models, out, phase='prefill', rows=(128,), family='tiles', formats=('F16','Q4_0','Q4_K_M'), max_candidates=8):
    if type(max_candidates) is not int or max_candidates<1:raise ValueError('positive candidate budget required')
    import torch
    torch.backends.cuda.matmul.allow_tf32=False
    models=Path(models);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[2]
    sources={name:dict(sha256=hashlib.sha256((repo/name).read_bytes()).hexdigest(),text=(repo/name).read_text()) for name in ('benchmarks/lfm2/cuda_format_search.py', 'src/tensor/compiler/search.py', 'src/tensor/compiler/cuda_schedules.py', 'src/tensor/compiler/cuda_lowering.py', 'packages/tensor-llm/src/tensor_llm/lfm2/kernels/cuda.py')}
    cases=[];seen=set();model_hashes={};profile_entries=[]
    profile=ScheduleProfile(json.loads(Path(__file__).with_name('profiles').joinpath('cuda-sm86-lfm2.5-2.6b.json').read_text()))
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
                spaces=projection_space(phase,f16_vectors=q==1,family=family)
                defaults=(dict(block_m=32,block_n=64,block_k=64,stages=2,threads=128,packed_pairs=family!='tiles')
                          if phase=='prefill' else dict(threads=128,unroll=4,**({'f16_values':4} if q==1 else {})))
                seed={**defaults,**profile.select(kind,parameters,provider='cuda',target='sm_86')}
                if phase=='prefill' and family=='paired-loads':seed['packed_pairs']=True
                seed['family']=next(iter(spaces))
                # A restricted family can exclude an otherwise valid measured
                # seed (for example a 64-row tile in the small-row family).
                for axis,values in spaces[seed['family']].items():
                    if seed.get(axis,values[0]) not in values:seed[axis]=values[0]
                discovery=ScheduleSearch([seed],spaces=spaces,width=4,
                    legal=lambda cfg:projection_legal(cfg,depth=k,paired=kind=='ffn',packed_pairs=q in (1,2,12,14)))
                def candidates():
                    yield 'before',(None,current_source(kind,parameters))
                    if phase=='decode' and kind=='ffn' and q==1:
                        yield 'separate',(dict(paired=False),current_source('linear',parameters))
                    for index in range(max_candidates):
                        try:config=discovery.next()
                        except StopIteration:break
                        p={**parameters,**{key:value for key,value in config.items() if key!='family'}}
                        text=prefill_source(kind,p) if phase=='prefill' else current_source(kind,p)
                        yield f'candidate-{index}',(config,text)
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
                    for label,(config,text) in candidates():
                        kernel=swiglu=gate=up=None
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
                            if config and 'family' in config:discovery.record(config,timing['median_gpu_seconds'])
                        except Exception as exc:
                            row['candidates'][label]=dict(status='rejected',config=config,reason=str(exc))
                            print(key,label,'rejected',str(exc)[:160],flush=True)
                        finally:
                            for resource in (kernel,swiglu,gate,up):
                                if resource is not None:resource.release()
                    passed=[n for n,v in row['candidates'].items() if v['status']=='passed' and n!='separate']
                    assert passed,row
                    row['selected']=min(passed,key=lambda n:row['candidates'][n]['timing']['median_gpu_seconds'])
                    selected=row['candidates'][row['selected']]
                    config={name:value for name,value in (selected['config'] or {}).items() if name!='family'}
                    profile_entries.append(dict(operation=kind,parameters=parameters,schedule=config,
                                                source_sha256=selected['source_sha256'],median_gpu_seconds=selected['timing']['median_gpu_seconds']))
                    adapter=device.info
                cases.append(row)
                report=dict(schema='tensor.lfm2-cuda-format-search.v1',phase=phase,family=family,
                            adapter=adapter,cases=cases,model_sha256=model_hashes,sources=sources,
                            protocol=dict(seed=2093,gpu_concurrency=1,warmups=3,samples=5,
                                          graph_repetitions=20,times_divided_by=20,relative_rms_max=.002,
                                          baseline='generic packed schedule, not the frozen full-model before profile',
                                          reference='independent Torch FP16 prefill/FP32 decode operators',
                                          discovery='tensor.compiler.search.ScheduleSearch',max_candidates_per_shape=max_candidates))
                selected_profile=dict(schema='tensor.schedule-profile.v1',provider='cuda',target='sm_86',entries=profile_entries,
                                      provenance=dict(method='shared beam discovery; independent correctness gate then GPU timing'))
                (out/'profile.json').write_text(json.dumps(selected_profile,indent=2)+'\n')
                (out/'search.json').write_text(json.dumps(report,indent=2)+'\n')
    return cases


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,default=Path('build/lfm2-models'))
    p.add_argument('--out',type=Path,required=True);p.add_argument('--phase',choices=('prefill','decode'),default='prefill')
    p.add_argument('--rows',type=int,nargs='+',default=[128])
    p.add_argument('--family',choices=('tiles','paired-loads','small-rows'),default='tiles')
    p.add_argument('--formats',nargs='+',choices=('F16','Q4_0','Q4_K_M'),default=['F16','Q4_0','Q4_K_M'])
    p.add_argument('--candidates',type=int,default=8)
    args=p.parse_args();search(args.models,args.out,args.phase,tuple(args.rows),args.family,tuple(args.formats),args.candidates)
