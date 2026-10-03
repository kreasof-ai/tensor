"""Search decode value reductions across short and long active prefixes."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.webgpu import Device
from tensor_llm.webgpu_kernels import source
from benchmarks.lfm2.tensor_projection_search import TimestampAdapter,Timer,bind,check


def run(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);device=Device();device._adapter=TimestampAdapter(device._adapter)
    records=[];h,kh,d,cap=16,8,64,576
    rng=np.random.default_rng(79);scores=(rng.normal(size=(h,cap))*3).astype(np.float32)
    values=rng.normal(size=(cap,kh,d)).astype(np.float16)
    expected={};bounds={}
    for position in (0,31,128,384,510):
        s=scores[:,:position+1].astype(np.float64);p=np.exp(s-s.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
        v=values[:position+1].astype(np.float64)[:,np.arange(h)//(h//kh),:].transpose(1,0,2)
        expected[position]=(p[:,:,None]*v).sum(axis=1)
        bounds[position]=(p[:,:,None]*np.abs(v)).sum(axis=1)*5e-6+1e-7
    specs=[{'name':'baseline','p':{}}]+[{'name':f'c{c}-p{parts}','p':dict(attention_schedule='partitioned_values',channels=c,value_parts=parts)}
                                    for c in (16,32,64) for parts in (1,2,4,8,16)]
    with device:
        args=(device.from_numpy(scores.ravel()),device.from_numpy(values.ravel()),device.full(h*d,np.nan),device.from_numpy(np.array([0,1],np.int32)))
        for spec in specs:
            p=dict(r=1,h=h,kh=kh,d=d,cap=cap,sg=True,**spec['p']);name=spec['name']
            path=out/(name+'.py');artifact=path.with_suffix('.tbin');path.write_text(source('attention',p));artifact.unlink(missing_ok=True)
            row={'name':name,'parameters':p,'validation':[],'positions':{}}
            try:
                tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');kernel=device.load(artifact);plan=bind(device,kernel,args);timer=Timer(device,plan)
                for position in expected:
                    device._gpu.queue.write_buffer(args[-1]._storage,0,np.array([position,1],np.int32).tobytes())
                    # The score producer's inactive entries are -infinity;
                    # inactive KV entries are poisoned to catch unwanted reads.
                    masked_scores=scores.copy();masked_scores[:,position+1:]=float('-inf')
                    masked_values=values.copy();masked_values[position+1:]=np.nan
                    device._gpu.queue.write_buffer(args[0]._storage,0,masked_scores.tobytes())
                    device._gpu.queue.write_buffer(args[1]._storage,0,masked_values.tobytes())
                    device._gpu.queue.write_buffer(args[2]._storage,0,np.full(h*d,np.nan,np.float32).tobytes())
                    plan.launch();row['validation'].append({'position':position,**check(args[2].to_numpy().reshape(h,d),expected[position],bounds[position])})
                    for _ in range(3):timer.sample(100)
                    samples=[timer.sample(20)/20 for _ in range(7)]
                    row['positions'][str(position)]={'samples_seconds':samples,'median_gpu_seconds':statistics.median(samples)}
                row.update(status='passed',score=statistics.mean(row['positions'][str(pos)]['median_gpu_seconds'] for pos in (31,128,384)),
                           source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                timer.close();plan.close();kernel._dispose()
            except Exception as error:row.update(status='rejected',error=f'{type(error).__name__}: {str(error)[-800:]}')
            records.append(row);print(name,row.get('score'),row.get('error'),flush=True)
            (out/'report.json').write_text(json.dumps({'status':'searching','records':records},indent=2)+'\n')
        for arg in args:arg.release()
        report={'status':'finished','adapter':device.info,'records':records,
                'protocol':'float64 softmax/value reference, NaN sentinel, 5 active prefix positions; 3x100 warmups and 7x20 timestamp samples per position',
                'best':min((row for row in records if row['status']=='passed'),key=lambda row:row['score'])}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    run(p.parse_args().out)
