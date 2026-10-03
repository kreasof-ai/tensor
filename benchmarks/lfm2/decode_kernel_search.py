"""Oracle-gated decode search with the complete F16 matrix traffic per sample."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor,wgpu
from tensor.providers.webgpu import Device
from tensor.compiler.webgpu_search import ScheduleSearch
from tensor_llm import GGUF
from tensor_llm.webgpu_kernels import source
from tensor_llm.model import DECODE_GEMV,DECODE_GEMV_COMMON
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,bind,check
from wgpu.backends.wgpu_native.extras import write_timestamp

SPACE=dict(lanes=(8,16,32,64,128),threads=(64,128,256,512),micro_rows=(1,2,4,8),
           dot_width=(1,2,4),unroll=(1,2,4,8),accumulators=(1,2,4),
           k_layout=('striped','blocked'),shared_input=(False,True))
DEFAULT=dict(lanes=32,threads=128,micro_rows=1,dot_width=4,unroll=1,
             accumulators=1,k_layout='striped',shared_input=False)


class TrafficTimer:
    def __init__(self,device,plan,targets):
        self.device=device;self.plan=plan;self.targets=set(targets)
        self.count=2*(len(targets)+1)
        self.query=device._gpu.create_query_set(type='timestamp',count=self.count)
        self.result=device._gpu.create_buffer(size=self.count*8,usage=wgpu.BufferUsage.QUERY_RESOLVE|wgpu.BufferUsage.COPY_SRC)
    def sample(self):
        gpu=self.device._gpu;encoder=gpu.create_command_encoder();compute=encoder.begin_compute_pass()
        write_timestamp(compute,self.query,0);index=2
        for i,(pipeline,group,grid) in enumerate(self.plan.nodes):
            if i in self.targets:write_timestamp(compute,self.query,index)
            compute.set_pipeline(pipeline);compute.set_bind_group(0,group);compute.dispatch_workgroups(*grid)
            if i in self.targets:write_timestamp(compute,self.query,index+1);index+=2
        write_timestamp(compute,self.query,1);compute.end()
        encoder.resolve_query_set(self.query,0,self.count,self.result,0);gpu.queue.submit([encoder.finish()])
        ticks=np.frombuffer(gpu.queue.read_buffer(self.result),dtype=np.uint64)
        return {'whole_seconds':int(ticks[1]-ticks[0])*1e-8,
                'target_seconds':sum(int(b-a) for a,b in zip(ticks[2::2],ticks[3::2]))*1e-8}
    def close(self):self.result.destroy();self.query.destroy()


def group_name(info):
    if '.ffn_gate.' in info.name or '.ffn_up.' in info.name:return 'ffn_gate_up'
    return str(info.shape[0])+'x'+str(info.shape[1])


def run(model,out,minutes,extended=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);gguf=GGUF(model)
    infos=[t for t in gguf.tensors.values() if len(t.shape)==2 and not t.name.endswith('conv.weight')]
    if any(t.type!=1 for t in infos):raise ValueError('this experiment requires native F16 matrices')
    weights={t.name:np.array(gguf.array(t.name,dtype=np.float16),copy=True) for t in infos}
    inputs={k:(np.random.default_rng(29).normal(size=k)).astype(np.float32) for _,k in {t.shape for t in infos}}
    references={}
    for t in infos:
        w=weights[t.name].astype(np.float64);x=inputs[t.shape[1]].astype(np.float64)
        references[t.name]=(w@x,np.abs(w)@np.abs(x)*3e-6+1e-10)
    device=Device();device._adapter=TimestampAdapter(device._adapter)
    report={'status':'searching','model_sha256':hashlib.sha256(Path(model).read_bytes()).hexdigest(),
            'protocol':'all model matrices streamed once per sample; target timestamps within the full traffic plan; 250 ms continuous warmup, 7 GPU samples; independent float64 F32-input/F16-weight oracle',
            'records':[],'budget_seconds':minutes*60,'sources':{}}
    for path in (Path(__file__),Path('src/tensor/compiler/webgpu_lowering.py'),Path('src/tensor/compiler/webgpu_search.py'),
                 Path('packages/tensor-llm/src/tensor_llm/webgpu_kernels.py')):
        report['sources'][path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'text':path.read_text()}
    started=time.perf_counter();deadline=started+minutes*60
    space={**SPACE,'unroll':(1,2,4,5,8,10,16,20),'accumulators':(1,2,4,8)} if extended else SPACE
    report['search_space']=space;report['extended']=extended
    def save():
        report['wall_seconds']=time.perf_counter()-started
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with device:
        report['adapter']=device.info
        buffers={n:device.from_numpy(w.ravel()) for n,w in weights.items()}
        ins={k:device.from_numpy(x) for k,x in inputs.items()};outs={o:device.full(o,np.nan) for o,_ in {t.shape for t in infos}}
        baseline={};calls=[];shapes=set(t.shape for t in infos)
        for o,k in shapes:
            path=out/f'baseline-{o}x{k}.py';artifact=path.with_suffix('.tbin')
            path.write_text(source('linear',dict(r=1,k=k,o=o,type=1,sg=True)));artifact.unlink(missing_ok=True)
            tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');baseline[(o,k)]=device.load(artifact)
        for t in infos:
            kernel=baseline[t.shape];plan=bind(device,kernel,(ins[t.shape[1]],buffers[t.name],outs[t.shape[0]]))
            calls.append(plan.calls[0]);plan.close()
        groups={group_name(t):[i for i,v in enumerate(infos) if group_name(v)==group_name(t)] for t in infos}
        order=sorted(groups,key=lambda name:-sum(np.prod(infos[i].shape) for i in groups[name]))
        for gi,name in enumerate(order):
            group_start=time.perf_counter();group_deadline=min(deadline-3,group_start+(deadline-group_start)/(len(order)-gi)-3)
            indices=groups[name];first=infos[indices[0]];o,k=first.shape
            families=('separate','paired') if name=='ffn_gate_up' else ('separate',)
            seeds=[]
            for family in families:
                selected={**DECODE_GEMV_COMMON,**DECODE_GEMV[(k,o)]}
                selected['family']=family
                seeds += ([selected]+[{**selected,'unroll':u,'accumulators':a} for u,a in ((16,8),(20,4),(10,2),(5,1),(8,8))] if extended else [])
                seeds += [dict(family=family,**DEFAULT)]+[dict(family=family,**{**DEFAULT,**v}) for v in
                  ({'lanes':64},{'lanes':16},{'lanes':8},{'threads':64},{'threads':256},{'micro_rows':2},
                   {'micro_rows':4},{'unroll':4,'accumulators':4},{'lanes':64,'threads':256,'micro_rows':2},
                   {'lanes':16,'threads':64,'micro_rows':2},{'shared_input':True})]
            search=ScheduleSearch(seeds,width=16 if extended else 12,spaces={family:space for family in families})
            row={'group':name,'shape':[o,k],'matrices':[infos[i].name for i in indices],'candidates':[]};report['records'].append(row)
            # Keep the previous default as an independently measured control.
            base_plan=device.prepare_plan(calls);base_timer=TrafficTimer(device,base_plan,indices)
            warm=time.perf_counter()+.5
            while time.perf_counter()<warm:base_plan.launch();device.synchronize()
            row['baseline_samples']=[base_timer.sample() for _ in range(7)]
            row['baseline_seconds']=statistics.median(s['target_seconds'] for s in row['baseline_samples'])
            base_timer.close();base_plan.close();save()
            while time.perf_counter()<group_deadline:
                config=search.next();index=len(row['candidates']);record={'index':index,'config':config}
                kernel=plan=timer=None
                try:
                    p={key:value for key,value in config.items() if key!='family'}
                    p.update(r=1,k=k,o=o,type=1,sg=True,decode_schedule='streamed')
                    kind='ffn' if config['family']=='paired' else 'linear'
                    path=out/f'{name}-{index}.py';artifact=path.with_suffix('.tbin');path.write_text(source(kind,p));artifact.unlink(missing_ok=True)
                    tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');kernel=device.load(artifact)
                    updated=[];targets=[];checks=[]
                    for i,(t,call) in enumerate(zip(infos,calls)):
                        if i not in indices:updated.append(call);continue
                        if kind=='ffn' and '.ffn_up.' in t.name:continue
                        args=(ins[k],buffers[t.name],outs[o]);expected,bound=references[t.name]
                        if kind=='ffn':
                            upname=t.name.replace('_gate.','_up.');u,ub=references[upname]
                            args=(ins[k],buffers[t.name],buffers[upname],outs[o])
                            silu=expected/(1+np.exp(-expected))
                            bound=np.abs(u)*1.1*bound+np.abs(silu)*ub+1.1*bound*ub+np.abs(silu*u)*2e-6+1e-10
                            expected=silu*u
                        single=bind(device,kernel,args)
                        device._gpu.queue.write_buffer(outs[o]._storage,0,np.full(o,np.nan,np.float32).tobytes())
                        single.launch();checks.append({'weight':t.name,**check(outs[o].to_numpy(),expected,bound)})
                        targets.append(len(updated));updated.append(single.calls[0]);single.close()
                    plan=device.prepare_plan(updated);timer=TrafficTimer(device,plan,targets)
                    warm=time.perf_counter()+.25
                    while time.perf_counter()<warm:plan.launch();device.synchronize()
                    samples=[timer.sample() for _ in range(7)];score=statistics.median(s['target_seconds'] for s in samples)
                    record.update(status='passed',samples=samples,median_gpu_seconds=score,validation=checks,
                                  artifact=str(artifact.relative_to(out)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                    search.record(config,score)
                    if 'best' not in row or score<row['best']['median_gpu_seconds']:
                        row['best']=dict(record);print('best',name,index,round(score*1e6,2),config,flush=True)
                except Exception as error:record.update(status='rejected',error=f'{type(error).__name__}: {str(error)[-800:]}')
                finally:
                    if timer:timer.close()
                    if plan:plan.close()
                    if kernel:kernel._dispose()
                row['candidates'].append(record);save()
            row['wall_seconds']=time.perf_counter()-group_start;save()
        for kernel in baseline.values():kernel._dispose()
        for buf in [*buffers.values(),*ins.values(),*outs.values()]:buf.release()
    report['status']='finished';save()
    print('finished',report['wall_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minutes',type=float,default=15);p.add_argument('--extended',action='store_true');a=p.parse_args()
    if not 0<a.minutes<=30:p.error('minutes must be in (0,30]')
    run(a.model,a.out,a.minutes,a.extended)
