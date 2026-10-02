"""Compare actual LFM2 projections, including tinygrad's optional BEAM search."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,statistics,time
from pathlib import Path
import numpy as np
import tensor
from tensor.runtime.abi import BoundCall
from tensor_llm import GGUF
from tinygrad import Tensor,TinyJit,Device
from tinygrad.llm.gguf import ggml_data_to_tensor
from tinygrad.uop.ops import Ops
from benchmarks.lfm2.tinygrad_compare import tiny_info
from benchmarks.lfm2.tinygrad_reference import rounded_half


def run(model,bundle,out,rows=(1,32)):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);gguf=GGUF(model)
    manifest=json.loads((Path(bundle)/'inference.json').read_text());records=[]
    for suffix in ('ffn_gate','ffn_down'):
        name='blk.0.'+suffix+'.weight';info=gguf.tensors[name];o,k=info.shape
        decoded=gguf.array(name).reshape(o,k)
        for r in rows:
            x=(np.random.default_rng(29).normal(size=(r,k))*.01).astype(np.float32)
            left=x.astype(np.float16).astype(np.float64) if r>1 else x.astype(np.float64)
            right=decoded.astype(np.float16).astype(np.float64) if r>1 else decoded.astype(np.float64)
            expected=left@right.T;tolerance=np.abs(left)@np.abs(right).T*3e-6+1e-10
            raw=gguf.packed(name)
            if info.type in (0,1):weight=Tensor(np.array(raw.view(np.float16 if info.type==1 else np.float32).reshape(o,k),copy=True)).realize()
            else:weight=ggml_data_to_tensor(Tensor(np.array(raw,copy=True)).realize(),o*k,info.type).reshape(o,k)
            input=Tensor(x).realize()
            def operation(value):
                lhs,rhs=(rounded_half(value),rounded_half(weight)) if r>1 else (value.float(),weight.float())
                return (lhs@rhs.T).realize()
            jit=TinyJit(operation);start=time.perf_counter();result=jit(input);result.numpy();cold=time.perf_counter()-start
            start=time.perf_counter();result=jit(input);result.numpy();capture=time.perf_counter()-start
            actual=jit(input).numpy()
            if not np.all(np.isfinite(actual)) or not np.all(np.abs(actual-expected)<=tolerance):raise AssertionError(('tinygrad',name,r,float(np.max(np.abs(actual-expected)))))
            record={'weight':name,'shape':[r,k,o],'type':info.type,'tinygrad':{'first_seconds':cold,'capture_and_search_seconds':capture,
                    'maximum_absolute_error':float(np.max(np.abs(actual-expected)))}}
            programs=[]
            for prg in jit.captured._linear.toposort():
                if prg.op is not Ops.PROGRAM:continue
                filename=f'{suffix}-r{r}-{len(programs)}.txt'
                source=next(u.arg for u in prg.src if u.op is Ops.SOURCE)
                (out/filename).write_text(source)
                programs.append({'source':filename,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                                 'global_size':str(prg.arg.global_size),'local_size':str(prg.arg.local_size),
                                 'applied_opts':str(prg.src[0].arg.applied_opts)})
            record['tinygrad']['programs']=programs
            kernel_record=next(v for v in manifest['kernels'].values() if v['kind']=='linear' and
                all(v['parameters'].get(key)==value for key,value in dict(r=r,k=k,o=o,type=info.type).items()))
            with tensor.Device(provider='webgpu') as device:
                source_weight=raw.view(np.float16 if info.type==1 else np.float32 if info.type==0 else np.uint32)
                args=(device.from_numpy(x.ravel()),device.from_numpy(source_weight),device.zeros(r*o))
                start=time.perf_counter();kernel=device.load(Path(bundle)/kernel_record['artifact'])
                values,symbols,launch=kernel._bind(args,{},include_outputs=True)
                plan=device.prepare_plan([(kernel,BoundCall(device,kernel.manifest,values,symbols,launch,validated=True))])
                plan.launch();actual=args[-1].to_numpy().reshape(r,o);tensor_first=time.perf_counter()-start
                if not np.all(np.isfinite(actual)) or not np.all(np.abs(actual-expected)<=tolerance):raise AssertionError(('tensor',name,r))
                record['tensor']={'first_consumer_seconds':tensor_first,'maximum_absolute_error':float(np.max(np.abs(actual-expected))),
                                  'artifact_sha256':kernel_record['sha256'],'parameters':kernel_record['parameters']}
                # Warm both devices continuously after compilation, then rotate
                # order. One hot matrix isolates scheduling, not model bandwidth.
                for name_runner in ('tensor','tinygrad'):
                    deadline=time.perf_counter()+1
                    while time.perf_counter()<deadline:
                        for _ in range(10):plan.launch() if name_runner=='tensor' else jit(input)
                        device.synchronize() if name_runner=='tensor' else Device[Device.DEFAULT].synchronize()
                samples={'tensor':[],'tinygrad':[]}
                for repeat in range(8):
                    for name_runner in (('tensor','tinygrad') if repeat%2==0 else ('tinygrad','tensor')):
                        start=time.perf_counter()
                        for _ in range(20):plan.launch() if name_runner=='tensor' else jit(input)
                        device.synchronize() if name_runner=='tensor' else Device[Device.DEFAULT].synchronize()
                        elapsed=(time.perf_counter()-start)/20
                        if repeat:samples[name_runner].append(elapsed)
                for name_runner,values in samples.items():record[name_runner].update(samples_seconds=values,median_seconds=statistics.median(values))
                adapter=device.info
            records.append(record);print(record,flush=True)
    report={'status':'passed','model':str(Path(model).resolve()),'model_sha256':hashlib.file_digest(Path(model).open('rb'),'sha256').hexdigest(),
            'tinygrad':tiny_info(),'tensor_adapter':adapter,'records':records,
            'protocol':'FP16-rounded operands/FP32 products for r>1; FP32 operands/products for r=1; independent float64 oracle, absolute-product error bound; compiled prepared/JIT execution, completion included, host output excluded; 1 second warmup, one discarded then seven batches of 20; hot single matrix'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--rows',type=int,nargs='+',default=[1,32],choices=(1,32))
    a=p.parse_args();run(a.model,a.bundle,a.out,tuple(a.rows))
