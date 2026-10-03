"""Fresh Tensor/tinygrad/llama.cpp projection comparison after all search stops."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,ctypes,hashlib,json,pickle,re,sqlite3,statistics,time
from dataclasses import replace
from pathlib import Path
import numpy as np
from tensor_llm import GGUF
from tensor.artifacts.format import read_artifact
from tinygrad import Tensor,TinyJit,Device as TinyDevice,UOp
from tinygrad.codegen import to_program_cache
from tinygrad.codegen.opt import search
from tinygrad.engine.realize import time_call
from tinygrad.uop.ops import Ops
from benchmarks.lfm2.tinygrad_reference import rounded_half
from benchmarks.lfm2.tensor_projection_search import Device,TimestampAdapter,Timer,bind,oracle,check,seeds
from benchmarks.lfm2.llama_projection_compare import VulkanProjection

def tiny_runner(inputs,weights,cache,width=8):
    input=Tensor(inputs).realize();weight=Tensor(weights).realize()
    def select(s,rawbufs,var_vals,amt,allow_test_size=True):
        key=s.ast.replace(arg=replace(s.ast.arg,beam=width)).key
        with sqlite3.connect('file:'+str(Path(cache).resolve())+'?mode=ro',uri=True) as db:
            found=db.execute('select val from beam_search_24 where ast=? and amt=? and allow_test_size=0 and device=?',
                             (key,width,s.ren.target.device)).fetchone()
        if found is None:raise ValueError('missing tinygrad cached winner')
        candidate=s.copy()
        for opt in pickle.loads(found[0]):candidate.apply_opt(opt)
        return candidate
    def operation(value):return (rounded_half(value)@rounded_half(weight).T).realize()
    to_program_cache.clear();jit=TinyJit(operation);jit(input)
    original=search.beam_search;search.beam_search=select
    try:jit(input)
    finally:search.beam_search=original
    jit(input).numpy();programs=[u for u in jit.captured._linear.toposort() if u.op is Ops.PROGRAM]
    if len(programs)!=1:raise ValueError('expected one tinygrad projection program')
    output=Tensor.zeros(*jit(input).shape).realize();program=programs[0]
    timer=time_call(program.call(*(UOp.from_buffer(t.uop.buffer) for t in (output,input,weight))))
    return jit,input,weight,program,timer,output

class TinyBatchTimer:
    """Profile the cached OpenCL source in a queued batch, releasing events."""
    def __init__(self,source,program,buffers):
        from tinygrad.runtime.ops_cl import cl,check,checked,BP_CB
        from tinygrad.helpers import to_char_p_p
        self.cl=cl;self.check=check;self.device=TinyDevice[TinyDevice.DEFAULT]
        binary=self.device.cl_compiler.compile_cached(source)
        self.program=checked(cl.clCreateProgramWithBinary(self.device.context,1,self.device.cl_dev,
            (ctypes.c_size_t*1)(len(binary)),to_char_p_p([binary],ctypes.c_ubyte),binary_status:=ctypes.c_int32(),
            error:=ctypes.c_int32()),error)
        check(binary_status.value);check(cl.clBuildProgram(self.program,1,self.device.cl_dev,None,BP_CB(),None))
        name=re.search(r'__kernel void (\w+)\(',source).group(1)
        self.kernel=checked(cl.clCreateKernel(self.program,name.encode(),status:=ctypes.c_int32()),status)
        for i,value in enumerate(buffers):
            raw=value.uop.buffer._buf
            check(cl.clSetKernelArg(self.kernel,i,ctypes.sizeof(raw),ctypes.byref(raw)))
        self.local=tuple(int(v) for v in program.arg.local_size)
        self.global_=tuple(int(g)*l for g,l in zip(program.arg.global_size,self.local))
    def sample(self,count=20):
        cl=self.cl;events=[]
        try:
            for _ in range(count):
                event=cl.cl_event();self.check(cl.clEnqueueNDRangeKernel(self.device.queue,self.kernel,len(self.global_),None,
                    (ctypes.c_size_t*len(self.global_))(*self.global_),(ctypes.c_size_t*len(self.local))(*self.local),0,None,event));events.append(event)
            self.check(cl.clWaitForEvents(1,events[-1]));samples=[];starts=[];ends=[]
            for event in events:
                self.check(cl.clGetEventProfilingInfo(event,cl.CL_PROFILING_COMMAND_START,8,ctypes.byref(start:=ctypes.c_uint64()),None))
                self.check(cl.clGetEventProfilingInfo(event,cl.CL_PROFILING_COMMAND_END,8,ctypes.byref(end:=ctypes.c_uint64()),None))
                starts.append(start.value);ends.append(end.value);samples.append((end.value-start.value)*1e-9)
            return {'individual_seconds':samples,'kernel_mean_seconds':statistics.mean(samples),
                    'interval_per_dispatch_seconds':(ends[-1]-starts[0])*1e-9/count}
        finally:
            for event in events:self.check(cl.clReleaseEvent(event))
    def close(self):
        self.check(self.cl.clReleaseKernel(self.kernel));self.check(self.cl.clReleaseProgram(self.program))

def run(model,search_root,tiny_root,llama_directory,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    root=Path(search_root).resolve();summary=json.loads((root/'report.json').read_text())
    if summary['status']!='finished':raise ValueError('search must finish before final recheck')
    for row in summary['records']:
        # Resumed search stages may have added seeds before the default tile.
        # Identify the baseline by its schedule rather than evaluation order.
        row['baseline']=next(c for c in row['candidates'] if c['config']==seeds()[0] and c['status']=='passed')
    gguf=GGUF(model);records=[];device=Device();device._adapter=TimestampAdapter(device._adapter)
    with device:
        for row in summary['records']:
            suffix=row['weight'];weight=np.array(gguf.array('blk.0.'+suffix+'.weight',dtype=np.float16),copy=True);o,k=weight.shape
            rounded=lambda values:values.astype(np.float32).astype(np.float16).astype(np.float32)
            # llama.cpp exposes F16 weights/F32 activations. Upload identical
            # nearest-even-rounded F32 bytes to all three, preserving the common
            # FP16-operand oracle without timing host conversion in any framework.
            inputs=rounded(np.random.default_rng(29).normal(size=(32,k))*.01)
            fixtures=[(29,.01,inputs)]+[(seed,scale,rounded(np.random.default_rng(seed).normal(size=(32,k))*scale))
                       for seed,scale in ((101,.01),(202,1.0),(303,.00001))]
            references={seed:oracle(values,weight) for seed,scale,values in fixtures}
            args=(device.from_numpy(inputs.ravel()),device.from_numpy(weight.ravel()),device.full(32*o,np.nan))
            variants={};runners={};timers={};kernels=[];plans=[];sources=set()
            candidates=[row['baseline']]+sorted((c for c in row['candidates'] if c['status']=='passed'),key=lambda c:c['median_gpu_seconds'])[:12]
            try:
                for candidate in candidates:
                    artifact=root/candidate['artifact'];manifest,files=read_artifact(artifact)
                    shader=files['kernel.wgsl'];shader_hash=hashlib.sha256(shader).hexdigest()
                    if shader_hash in sources:continue
                    sources.add(shader_hash)
                    name='tensor-baseline' if candidate['index']==row['baseline']['index'] else 'tensor-'+str(candidate['index'])
                    kernel=device.load(artifact);plan=bind(device,kernel,args);kernels.append(kernel);plans.append(plan)
                    checks=[]
                    for seed,scale,values in fixtures:
                        device._gpu.queue.write_buffer(args[0]._storage,0,values.tobytes())
                        device._gpu.queue.write_buffer(args[-1]._storage,0,np.full(32*o,np.nan,dtype=np.float32).tobytes())
                        plan.launch();checks.append({'seed':seed,'scale':scale,**check(args[-1].to_numpy().reshape(32,o),*references[seed])})
                    device._gpu.queue.write_buffer(args[0]._storage,0,inputs.tobytes())
                    (out/f'{suffix}-{name}.wgsl').write_bytes(shader)
                    variants[name]={'config':candidate['config'],'artifact':str(artifact),'shader_sha256':shader_hash,
                                    'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),'validation':checks,
                                    'gpu_samples_seconds':[],'completed_samples_seconds':[]}
                    runners[name]=(plan.launch,device.synchronize);timers[name]=Timer(device,plan)
                    if len(variants)>=5:break
                cache=Path(tiny_root)/f'{suffix}-beam8-up128-local1024/cache.db'
                jit,input,tiny_weight,program,tiny_timer,tiny_output=tiny_runner(inputs,weight,cache)
                source=next(u.arg for u in program.src if u.op is Ops.SOURCE);(out/f'{suffix}-tinygrad.cl').write_text(source)
                checks=[{'seed':seed,'scale':scale,**check(jit(Tensor(values).realize()).numpy(),*references[seed])} for seed,scale,values in fixtures]
                variants['tinygrad']={'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'applied_opts':str(program.src[0].arg.applied_opts),
                                      'validation':checks,'gpu_samples_seconds':[],'completed_samples_seconds':[]}
                runners['tinygrad']=(lambda:jit(input),lambda:TinyDevice[TinyDevice.DEFAULT].synchronize())
                next(tiny_timer);check(tiny_output.numpy(),*references[29])
                tiny_batch=TinyBatchTimer(source,program,(tiny_output,input,tiny_weight))
                tiny_batch.sample();check(tiny_output.numpy(),*references[29])
                native=VulkanProjection(llama_directory,weight,inputs)
                checks=[]
                for seed,scale,values in fixtures:
                    native.update(values);native.launch();checks.append({'seed':seed,'scale':scale,**check(native.download(),*references[seed])})
                native.update(inputs)
                variants['llama.cpp']={'validation':checks,'completed_samples_seconds':[],
                                       'revision':'f872b591121761ac7b2af18283bd99bdc092a63a',
                                       'precision':'native F16 weights, host nearest-even-rounded F32 input, FP32 accumulators; internal input conversion included'}
                runners['llama.cpp']=(native.launch,native.synchronize)
                for launch,sync in runners.values():
                    deadline=time.perf_counter()+1
                    while time.perf_counter()<deadline:
                        for _ in range(20):launch()
                        sync()
                names=list(runners)
                for repeat in range(8):
                    for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                        launch,sync=runners[name];start=time.perf_counter()
                        for _ in range(20):launch()
                        sync();completed=(time.perf_counter()-start)/20
                        # Both single-dispatch and batched timestamps are retained
                        # for Tensor so query/readback idle time is visible.
                        if name in timers:
                            timer=timers[name];gpu=[timer.sample() for _ in range(20)]
                            for _ in range(3):timer.sample(100)
                            batched=timer.sample(20)/20
                            if repeat:variants[name].setdefault('batched_gpu_samples_seconds',[]).append(batched)
                        elif name=='tinygrad':
                            gpu=[next(tiny_timer) for _ in range(20)]
                            for _ in range(3):tiny_batch.sample(100)
                            batch=tiny_batch.sample()
                            if repeat:variants[name].setdefault('batched_gpu_samples',[]).append(batch)
                        else:gpu=None
                        if repeat:
                            variants[name]['completed_samples_seconds'].append(completed)
                            if gpu is not None:variants[name]['gpu_samples_seconds'].append(gpu)
                for name,value in variants.items():
                    value['median_completed_seconds']=statistics.median(value['completed_samples_seconds'])
                    if 'gpu_samples_seconds' in value:value['median_gpu_seconds']=statistics.median(statistics.median(b) for b in value['gpu_samples_seconds'])
                    if 'batched_gpu_samples_seconds' in value:value['median_batched_gpu_seconds']=statistics.median(value['batched_gpu_samples_seconds'])
                    if 'batched_gpu_samples' in value:
                        value['median_batched_gpu_seconds']=statistics.median(b['interval_per_dispatch_seconds'] for b in value['batched_gpu_samples'])
                        value['median_batched_kernel_seconds']=statistics.median(b['kernel_mean_seconds'] for b in value['batched_gpu_samples'])
                winner=min((name for name in variants if name.startswith('tensor-')),key=lambda name:variants[name]['median_batched_gpu_seconds'])
                records.append({'weight':suffix,'shape':[32,k,o],'tensor_winner':winner,'variants':variants})
                native.close();tiny_batch.close();print('rechecked',suffix,{name:value['median_completed_seconds'] for name,value in variants.items()},flush=True)
            finally:
                for timer in timers.values():timer.close()
                for plan in plans:plan.close()
                for kernel in kernels:kernel._dispose()
                for buffer in args:buffer.release()
            report={'status':'passed','records':records,'adapter':device.info,'timestamp_period_ns':10,
                    'protocol':'hot real F16 weights; four NumPy fixtures; rotate frameworks/finalists; 1 second warmup each; discard first then 7 batches of 20 completed calls; host copies excluded; no search',
                    'input_contract':'same host nearest-even F16-rounded values uploaded as F32 to Tensor, tinygrad and llama.cpp; host preparation excluded for all three; search itself checked unrounded F32 inputs',
                    'model_sha256':summary['model_sha256']}
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','search-root','tiny-root','llama-directory','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.model,a.search_root,a.tiny_root,a.llama_directory,a.out)
