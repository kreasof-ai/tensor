"""Oracle-gated search over Tensor-produced Vulkan WebGPU kernels."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor,wgpu
from tensor.providers.webgpu import Device
from tensor.runtime.abi import BoundCall
from tensor.artifacts.format import read_artifact
from tensor.compiler.webgpu_lowering import partitioned_matmul_schedule
from tensor.compiler.search import ScheduleSearch
from tensor.compiler.webgpu_schedules import SPACES, coupled_moves
from tensor_llm import GGUF
from tensor_llm.kernels import emit
from tensor_llm.webgpu_kernels import source,round_half
from wgpu.backends.wgpu_native.extras import write_timestamp

class TimestampAdapter:
    def __init__(self,adapter):self.adapter=adapter
    def __getattr__(self,name):return getattr(self.adapter,name)
    def request_device_sync(self,**kwargs):
        kwargs['required_features']+=['timestamp-query','timestamp-query-inside-passes']
        return self.adapter.request_device_sync(**kwargs)

def program(r,k,o,config):
    p=dict(config);family=p.pop('family')
    if family=='staged':
        tm,tn,bk=(p.pop(a) for a in ('tile_m','tile_n','tile_k'))
        # Producer helper now honors explicit thread counts for staged schedules.
        return source('linear',dict(r=r,k=k,o=o,type=1,tile=(tm,tn,bk),**p))
    if family=='partitioned_rows':p['owner_axis']='row'
    lhs=round_half('x[({row}) * '+str(k)+' + ({k})]')
    rhs='T.cast(w[({column}) * '+str(k)+' + ({k})], "float32")'
    body=partitioned_matmul_schedule(r,k,o,lhs,rhs,**p)
    return emit([('x',r*k,'float32'),('w',o*k,'float16'),('out',r*o,'float32')],body)

def oracle(inputs,weight):
    lhs=inputs.astype(np.float16).astype(np.float64);rhs=weight.astype(np.float64)
    return lhs@rhs.T,np.abs(lhs)@np.abs(rhs).T*3e-6+1e-10

def check(actual,expected,bound):
    if not np.isfinite(actual).all() or not np.all(np.abs(actual-expected)<=bound):
        raise AssertionError('independent FP16-operand oracle failed')
    return {'maximum_absolute_error':float(np.max(np.abs(actual-expected))),
            'maximum_error_over_bound':float(np.max(np.abs(actual-expected)/bound))}

class Timer:
    def __init__(self,device,plan,period=10):
        self.device=device;self.plan=plan;self.period=period
        self.query=device._gpu.create_query_set(type='timestamp',count=2)
        self.result=device._gpu.create_buffer(size=16,usage=wgpu.BufferUsage.QUERY_RESOLVE|wgpu.BufferUsage.COPY_SRC)
    def sample(self,dispatches=1):
        pipeline,group,grid=self.plan.nodes[0];gpu=self.device._gpu
        encoder=gpu.create_command_encoder();compute=encoder.begin_compute_pass()
        write_timestamp(compute,self.query,0);compute.set_pipeline(pipeline);compute.set_bind_group(0,group)
        for _ in range(dispatches):compute.dispatch_workgroups(*grid)
        write_timestamp(compute,self.query,1);compute.end()
        encoder.resolve_query_set(self.query,0,2,self.result,0);gpu.queue.submit([encoder.finish()])
        ticks=np.frombuffer(gpu.queue.read_buffer(self.result),dtype=np.uint64)
        return int(ticks[1]-ticks[0])*self.period*1e-9
    def close(self):self.result.destroy();self.query.destroy()

def bind(device,kernel,args):
    values,symbols,launch=kernel._bind(args,{},include_outputs=True)
    return device.prepare_plan([(kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True))])

def seeds():
    staged=dict(family='staged',tile_m=16,tile_n=32,tile_k=64,threads=128,dot_width=4,
                unroll=False,lhs_pad=0,lhs_transpose=False)
    direct=dict(family='partitioned',tile_m=8,tile_n=32,threads=128,partitions=8,unroll=4,dot_width=1)
    row=dict(family='partitioned_rows',tile_m=32,tile_n=5,threads=128,partitions=8,unroll=8,dot_width=2,k_layout='striped')
    return [staged,row,{**row,'tile_n':4},{**row,'tile_n':8},{**row,'partitions':16},{**row,'unroll':4}]+[{**direct,'tile_m':m,'partitions':p,'unroll':u}
                     for m,p,u in ((8,8,4),(8,16,4),(8,8,8),(16,8,4),(4,8,4),(8,4,4),(8,16,8),(8,8,16))]

def run(model,out,minutes,resume=False):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True);gguf=GGUF(model)
    previous=json.loads((out/'report.json').read_text()) if resume else None
    prior_wall=previous['wall_seconds'] if previous else 0
    started=time.perf_counter();deadline=started+minutes*60-prior_wall
    report={'status':'searching','combined_budget_seconds':minutes*60,'model_sha256':hashlib.sha256(Path(model).read_bytes()).hexdigest(),
            'compiler_sha256':hashlib.sha256(Path(tensor.__file__).parent.joinpath('compiler/webgpu_lowering.py').read_bytes()).hexdigest(),
            'search':'beam8 then beam16 with deterministic restarts; GPU median of 7 batches of 20 dispatches after 3 batches of 100 warmups',
            'oracle':'float64 dot of nearest-even FP16 operands; abs-product * 3e-6 + 1e-10',
            'timestamp_period_ns':10,'records':[]}
    if previous:
        report=previous;report['status']='searching'
        report.setdefault('phases',[]).append({'previous_wall_seconds':prior_wall,'reason':'add row ownership, non-power-of-two column tiles, striped K and vec2 dot',
          'compiler_sha256':hashlib.sha256(Path(tensor.__file__).parent.joinpath('compiler/webgpu_lowering.py').read_bytes()).hexdigest()})
        report['search']='beam8 then beam16, coordinate and coupled moves, deterministic restarts; direct column/row ownership and blocked/striped K; GPU median of 7 batches of 20 after 3 batches of 100 warmups'
    def save():
        report['wall_seconds']=prior_wall+time.perf_counter()-started
        temporary=out/'report.json.tmp'
        temporary.write_text(json.dumps(report,indent=2)+'\n')
        temporary.replace(out/'report.json')
    device=Device();device._adapter=TimestampAdapter(device._adapter)
    with device:
        report['adapter']=device.info
        for index,suffix in enumerate(('ffn_gate','ffn_down')):
            stage_start=time.perf_counter()
            stage_deadline=min(deadline-3,started+minutes*30-prior_wall-3) if index==0 else deadline-3
            weight=np.array(gguf.array('blk.0.'+suffix+'.weight',dtype=np.float16),copy=True);o,k=weight.shape;r=32
            inputs=(np.random.default_rng(29).normal(size=(r,k))*.01).astype(np.float32)
            expected,bound=oracle(inputs,weight);directory=out/suffix;directory.mkdir(exist_ok=True)
            args=(device.from_numpy(inputs.ravel()),device.from_numpy(weight.ravel()),device.full(r*o,np.nan))
            sentinel=np.full(r*o,np.nan,dtype=np.float32).tobytes()
            search=ScheduleSearch(seeds(),spaces=SPACES,coupled=coupled_moves);row=next((v for v in report['records'] if v['weight']==suffix),None)
            if row is None:
                row={'weight':suffix,'shape':[r,k,o],'candidates':[],'incumbents':[]};report['records'].append(row)
            else:
                from tensor.compiler.search import key
                search.seen={key(c['config']) for c in row['candidates'] if 'output must not exist' not in c.get('error','')}
                for c in row['candidates']:
                    if c['status']=='passed':search.record(c['config'],c['median_gpu_seconds'])
            save()
            while time.perf_counter()<stage_deadline:
                search.width=8 if time.perf_counter()-stage_start<(stage_deadline-stage_start)/2 else 16
                config=search.next();number=len(row['candidates']);record={'config':config,'index':number,'beam':search.width}
                path=directory/f'candidate-{number}.py';artifact=path.with_suffix('.tbin');kernel=plan=timer=None
                try:
                    text=program(r,k,o,config);path.write_text(text)
                    artifact.unlink(missing_ok=True)  # incomplete candidate left by an interrupted search
                    build_start=time.perf_counter();tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache')
                    record['build_seconds']=time.perf_counter()-build_start
                    kernel=device.load(artifact);plan=bind(device,kernel,args)
                    device._gpu.queue.write_buffer(args[-1]._storage,0,sentinel);plan.launch()
                    record['validation']=check(args[-1].to_numpy().reshape(r,o),expected,bound)
                    timer=Timer(device,plan)
                    for _ in range(3):timer.sample(100)
                    samples=[timer.sample(20)/20 for _ in range(7)];median=statistics.median(samples)
                    if median<=0:raise ValueError('nonpositive GPU timestamp')
                    record.update(status='passed',gpu_samples_seconds=samples,median_gpu_seconds=median,
                                  artifact=str(artifact.relative_to(out)))
                    search.record(config,median)
                    if 'best' not in row or median<row['best']['median_gpu_seconds']:
                        row['best']=dict(record);row['incumbents'].append({'index':number,'seconds':time.perf_counter()-stage_start,'median_gpu_seconds':median})
                        print('best',suffix,number,round(median*1e6,2),config,flush=True)
                    if config==seeds()[0]:row['baseline']=dict(record)
                except Exception as error:
                    record.update(status='rejected',error=f'{type(error).__name__}: {str(error)[-1200:]}')
                finally:
                    if timer:timer.close()
                    if plan:plan.close()
                    if kernel:kernel._dispose()
                row['candidates'].append(record);save()
                if number%25==0:print('progress',suffix,number,'elapsed',round(time.perf_counter()-stage_start),flush=True)
            row['wall_seconds']=row.get('wall_seconds',0)+time.perf_counter()-stage_start
            for buffer in args:buffer.release()
            save()
    report['status']='finished';save();print('finished',report['wall_seconds'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--minutes',type=float,default=30);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not 0<a.minutes<=30:p.error('minutes must be in (0,30]')
    run(a.model,a.out,a.minutes,a.resume)
