"""Discover fused QK/softmax/V schedules against the complete split baseline."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics
from pathlib import Path
import numpy as np
import tensor
from wgpu.backends.wgpu_native.extras import write_timestamp
from tensor.providers.webgpu import Device
from tensor_llm.webgpu_kernels import source
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,Timer,bind,check


class PlanTimer(Timer):
    def sample(self,dispatches=1):
        gpu=self.device._gpu;encoder=gpu.create_command_encoder();compute=encoder.begin_compute_pass()
        write_timestamp(compute,self.query,0)
        if self.plan._encode:
            from wgpu.backends.wgpu_native._ffi import ffi
            self.plan._encode(int(ffi.cast('uintptr_t',compute._internal)),
                              self.plan._records*dispatches,*self.plan._function_pointers)
        else:
            for _ in range(dispatches):
                for pipeline,group,grid in self.plan.nodes:
                    compute.set_pipeline(pipeline);compute.set_bind_group(0,group);compute.dispatch_workgroups(*grid)
        write_timestamp(compute,self.query,1);compute.end()
        encoder.resolve_query_set(self.query,0,2,self.result,0);gpu.queue.submit([encoder.finish()])
        ticks=np.frombuffer(gpu.queue.read_buffer(self.result),dtype=np.uint64)
        return int(ticks[1]-ticks[0])*self.period*1e-9


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    device=Device();device._adapter=TimestampAdapter(device._adapter)
    h,kh,d,cap=16,8,64,576;positions=(0,31,128,384,510);records=[]
    rng=np.random.default_rng(917)
    q=rng.normal(size=(h,d)).astype(np.float32)
    keys=rng.normal(size=(cap,kh,d)).astype(np.float16)
    values=rng.normal(size=(cap,kh,d)).astype(np.float16)
    expected={};bounds={}
    for pos in positions:
        k=keys[:pos+1].astype(np.float64)[:,np.arange(h)//2,:].transpose(1,0,2)
        logits=(k*q[:,None,:]).sum(axis=2)*d**-.5
        prob=np.exp(logits-logits.max(axis=1,keepdims=True));prob/=prob.sum(axis=1,keepdims=True)
        v=values[:pos+1].astype(np.float64)[:,np.arange(h)//2,:].transpose(1,0,2)
        expected[pos]=(prob[:,:,None]*v).sum(axis=1)
        # Propagate an absolute QK rounding bound through softmax and V.
        qk_bound=(np.abs(k)*np.abs(q[:,None,:])).sum(axis=2)*d**-.5*3e-6
        bounds[pos]=(prob[:,:,None]*np.abs(v)).sum(axis=1)*(2*qk_bound.max(axis=1)[:,None]+5e-6)+1e-7
    specs=[dict(name='split',channels=64,value_parts=16,fused_scores=False)]
    specs += [dict(name=f'fused-c{c}-p{parts}',channels=c,value_parts=parts,fused_scores=True)
              for c in (16,32,64) for parts in (2,4,8,16)]
    with device:
        qbuf=device.from_numpy(q.ravel());kbuf=device.from_numpy(keys.ravel());vbuf=device.from_numpy(values.ravel())
        scorebuf=device.full(h*cap,np.nan);outbuf=device.full(h*d,np.nan);control=device.from_numpy(np.array([0,1],np.int32))
        scorepath=out/'scores.py';scorepath.write_text(source('attention_scores',dict(r=1,h=h,kh=kh,d=d,cap=cap,sg=True)))
        scoreartifact=scorepath.with_suffix('.tbin');tensor.build(scorepath,scoreartifact,provider='webgpu',cache_dir=out/'cache')
        scorekernel=device.load(scoreartifact);scoreplan=bind(device,scorekernel,(qbuf,kbuf,scorebuf,control))
        runners={}
        for spec in specs:
            p=dict(r=1,h=h,kh=kh,d=d,cap=cap,sg=True,attention_schedule='partitioned_values',
                   **{k:v for k,v in spec.items() if k!='name'})
            path=out/(spec['name']+'.py');path.write_text(source('attention',p));artifact=path.with_suffix('.tbin')
            row=dict(name=spec['name'],parameters=p,validation=[],positions={});records.append(row)
            kernel=single=plan=timer=None
            try:
                tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');kernel=device.load(artifact)
                args=(qbuf,kbuf,vbuf,outbuf,control) if spec['fused_scores'] else (scorebuf,vbuf,outbuf,control)
                single=bind(device,kernel,args)
                plan=device.prepare_plan(single.calls if spec['fused_scores'] else (*scoreplan.calls,*single.calls));single.close()
                timer=PlanTimer(device,plan)
                for pos in positions:
                    device.write(control,np.array([pos,1],np.int32))
                    k=keys.copy();v=values.copy();k[pos+1:]=np.nan;v[pos+1:]=np.nan
                    device.write(kbuf,k.ravel());device.write(vbuf,v.ravel());device.write(outbuf,np.full(h*d,np.nan,np.float32))
                    actual=plan.launch(readback=outbuf).reshape(h,d)
                    row['validation'].append(dict(position=pos,**check(actual,expected[pos],bounds[pos])))
                row.update(status='passed',artifact=artifact.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                runners[spec['name']]=(plan,timer,kernel)
            except Exception as error:
                row.update(status='rejected',error=f'{type(error).__name__}: {str(error)[-800:]}')
                if timer:timer.close()
                if plan:plan.close()
                if single:single.close()
                if kernel:kernel._dispose()
            print('validated',row['name'],row['status'],row.get('error'),flush=True)
        # Rotate candidates at every position to reduce order/clock bias.
        for pos in positions:
            device.write(control,np.array([pos,1],np.int32))
            k=keys.copy();v=values.copy();k[pos+1:]=np.nan;v[pos+1:]=np.nan
            device.write(kbuf,k.ravel());device.write(vbuf,v.ravel())
            names=list(runners);samples={n:[] for n in names}
            for n in names:
                for _ in range(3):runners[n][1].sample(100)
            for repeat in range(8):
                for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                    value=runners[name][1].sample(20)/20
                    if repeat:samples[name].append(value)
            for row in records:
                if row['status']=='passed':row['positions'][str(pos)]=dict(samples_seconds=samples[row['name']],median_gpu_seconds=statistics.median(samples[row['name']]))
            print('position',pos,{n:round(statistics.median(samples[n])*1e6,2) for n in names},flush=True)
        for row in records:
            if row['status']=='passed':row['score']=statistics.mean(row['positions'][str(pos)]['median_gpu_seconds'] for pos in (31,128,384))
        for plan,timer,kernel in runners.values():timer.close();plan.close();kernel._dispose()
        scoreplan.close();scorekernel._dispose()
        report=dict(status='finished',adapter=device.info,records=records,
                    best=min((row for row in records if row['status']=='passed'),key=lambda row:row['score']),
                    protocol='float64 QK/softmax/V oracle with propagated rounding bounds; inactive K/V and output NaN sentinels; all 5 positions; complete split baseline vs fused; rotate 7 timestamp samples after warmup')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);run(p.parse_args().out)
