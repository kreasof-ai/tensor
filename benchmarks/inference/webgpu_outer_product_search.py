"""Oracle-gated outer-product schedule discovery on RX 6700 XT Vulkan."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.providers.webgpu import Device
from tensor.compiler.webgpu_lowering import outer_product_matmul_schedule
from tensor.compiler.webgpu_search import ScheduleSearch
from benchmarks.inference.clblast_comparison import TimestampAdapter,check
from benchmarks.lfm2.tensor_projection_search import Timer,bind


def source(m,n,k,config,mode='gemm',dtype='float32'):
    body=outer_product_matmul_schedule(m,k,n,dtype=dtype,epilogue=mode,**config)
    args=[('x',m*k,dtype),('w',n*k,dtype)]
    if mode=='linear':args.append(('bias',n,dtype))
    args.append(('out',m*n,'float32'))
    declarations=', '.join(f'{name}: T.Tensor(({size},), "{kind}")' for name,size,kind in args)
    return 'import tilelang.language as T\n\n@T.prim_func\ndef kernel('+declarations+'):\n'+ '\n'.join('    '+line for line in body.splitlines())+'\n\ndef tensor_export():return {"kernel":kernel,"outputs":["out"]}\n'


BASE=dict(tile_m=64,tile_n=64,tile_k=16,micro_m=4,micro_n=4,threads=256,
          lhs_layout='km',lhs_pad=0,rhs_pad=0,owner_axis='column',unroll=4,fma=True)
SPACE=dict(tile_m=(16,32,64,128),tile_n=(32,64,128),tile_k=(8,16,32,64),
           micro_m=(2,4,8),micro_n=(2,4,8),threads=(64,128,256,512),
           lhs_layout=('km','mk'),lhs_pad=(0,1),rhs_pad=(0,1),
           owner_axis=('column','row'),unroll=(1,2,4,8,16),fma=(True,False))


def seeds():
    configs=[]
    for tm,tn,mm,mn in ((64,64,4,4),(64,64,4,8),(64,64,8,4),(64,64,8,8),
                        (128,128,8,8),(64,128,4,8),(128,64,8,4),(32,128,4,8),
                        (64,128,8,8),(128,64,8,8)):
        for tk,pad in ((16,0),(16,1),(32,0),(32,1)):
            configs.append({**BASE,'tile_m':tm,'tile_n':tn,'micro_m':mm,'micro_n':mn,
                            'threads':tm*tn//(mm*mn),'tile_k':tk,'lhs_pad':pad,'rhs_pad':pad})
    return configs


def run(out,minutes,size,explicit=False,seeds_report=None,shape=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Only CPU reference construction uses BLAS; it is finished before timing.
    rng=np.random.default_rng(431)
    m,n,k=shape if shape else (size,size,size)
    if any(type(v) is not int or v<=0 for v in (m,n,k)):raise ValueError('shape must be positive integers')
    x=rng.normal(size=(m,k)).astype(np.float32);w=rng.normal(size=(n,k)).astype(np.float32)
    reference=x@w.T
    device=Device();device._adapter=TimestampAdapter(device._adapter)
    dispatches,warm_dispatches=(20,100) if m<=32 else (1,1)
    report=dict(status='searching',shape=[m,n,k],seed=431,budget_seconds=minutes*60,records=[],
                input_sha256=[hashlib.sha256(v.tobytes()).hexdigest() for v in (x,w)],search_space=SPACE,
                protocol='native F32 operands/accumulators/output; same FP32 CLBlast reference gate; output NaN sentinel; 3 warm and 7 timestamp samples; compilation/download/validation excluded from candidate score',
                sources={p:dict(sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest(),text=Path(p).read_text())
                         for p in (__file__,'src/tensor/compiler/webgpu.py','src/tensor/compiler/webgpu_lowering.py','src/tensor/compiler/webgpu_search.py')})
    report.update(timestamp_period_ns=10,dispatches_per_sample=dispatches,warm_dispatches_per_sample=warm_dispatches)
    report['protocol']=f'Native F32 operands/accumulators/output; FP32 CLBlast reference gate; NaN output sentinel; 3 warm batches of {warm_dispatches} and 7 timestamp batches of {dispatches}, normalized per dispatch. Compilation/download/validation excluded from score.'
    start=time.perf_counter();deadline=start+minutes*60
    def save():
        report['wall_seconds']=time.perf_counter()-start
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with device:
        report['adapter']=device.info;args=(device.from_numpy(x.ravel()),device.from_numpy(w.ravel()),device.full(m*n,np.nan))
        initial=seeds()
        if m<=32:
            small=[]
            for tm,tn,mm,mn in ((16,64,2,4),(32,64,4,4),(16,64,2,8),(32,64,2,4),
                                (32,32,2,2),(32,32,2,4),(16,32,2,4)):
                for tk in (16,32,64):
                    small.append({**BASE,'tile_m':tm,'tile_n':tn,'tile_k':tk,
                                  'micro_m':mm,'micro_n':mn,'threads':tm*tn//(mm*mn)})
            initial=small+initial
            # Keep the validated large-GEMM schedule alive as a skinny control.
            initial=[{**BASE,'tile_m':64,'tile_n':128,'micro_m':4,'micro_n':8,
                      'lhs_layout':'mk','unroll':16,'fma':False}]+initial
        if seeds_report:
            previous=json.loads(Path(seeds_report).read_text())
            initial=[r['config'] for r in sorted(previous['records'],key=lambda r:r.get('median_gpu_seconds',float('inf'))) if r['status']=='passed'][:12]+initial
            report['seeds_report_sha256']=hashlib.sha256(Path(seeds_report).read_bytes()).hexdigest()
        space={**SPACE,'explicit_unroll':(explicit,)}
        report['search_space']=space
        search=ScheduleSearch([dict(family='outer',**{**c,'explicit_unroll':explicit}) for c in initial],width=12,spaces={'outer':space})
        while time.perf_counter()<deadline:
            config=search.next();config.pop('family');i=len(report['records']);row=dict(index=i,config=config)
            kernel=plan=timer=None
            try:
                text=source(m,n,k,config);path=out/f'{i}.py';path.write_text(text);artifact=path.with_suffix('.tbin');artifact.unlink(missing_ok=True)
                tensor.build(path,artifact,provider='webgpu',cache_dir=out/'cache');kernel=device.load(artifact);plan=bind(device,kernel,args)
                device.write(args[-1],np.full(m*n,np.nan,np.float32))
                actual=plan.launch(readback=args[-1]).reshape(m,n);quality=check(actual,reference)
                if not quality['passed']:raise AssertionError(quality)
                timer=Timer(device,plan)
                for _ in range(3):timer.sample(warm_dispatches)
                samples=[timer.sample(dispatches)/dispatches for _ in range(7)];score=statistics.median(samples)
                row.update(status='passed',validation=quality,samples_seconds=samples,median_gpu_seconds=score,
                           artifact=artifact.name,source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                search.record(dict(family='outer',**config),score)
                if 'best' not in report or score<report['best']['median_gpu_seconds']:
                    report['best']=dict(row);print('best',i,round(score*1000,3),config,flush=True)
            except Exception as error:row.update(status='rejected',error=f'{type(error).__name__}: {str(error)[-600:]}')
            finally:
                if timer:timer.close()
                if plan:plan.close()
                if kernel:kernel._dispose()
            report['records'].append(row);save()
    report['status']='finished';save();print('finished',report['wall_seconds'],len(report['records']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--minutes',type=float,default=8);p.add_argument('--size',type=int,default=4096)
    p.add_argument('--explicit-unroll',action='store_true');p.add_argument('--seeds-report',type=Path)
    p.add_argument('--shape',type=int,nargs=3,metavar=('M','N','K'))
    a=p.parse_args()
    if not 0<a.minutes<=30 or a.size not in (512,1024,2048,4096):p.error('requires 0..30 minutes and a supported square size')
    run(a.out,a.minutes,a.size,a.explicit_unroll,a.seeds_report,a.shape)
