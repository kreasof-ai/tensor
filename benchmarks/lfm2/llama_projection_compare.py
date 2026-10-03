"""Call the pinned llama.cpp GGML Vulkan MUL_MAT directly, without an LLM."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,ctypes as c,hashlib,json,os,statistics,time
from pathlib import Path
import numpy as np
from tensor_llm import GGUF

class InitParams(c.Structure):
    _fields_=[('mem_size',c.c_size_t),('mem_buffer',c.c_void_p),('no_alloc',c.c_bool)]

class VulkanProjection:
    def __init__(self,directory,weight,inputs,graph_repeats=1,accumulator='f32'):
        if type(graph_repeats) is not int or not 1<=graph_repeats<=256:raise ValueError('invalid graph repetitions')
        directory=Path(directory).resolve();self.dll_path=os.add_dll_directory(str(directory))
        self.base=c.CDLL(str(directory/'ggml-base.dll'));self.vk=c.CDLL(str(directory/'ggml-vulkan.dll'))
        def fn(lib,name,result,args):
            value=getattr(lib,name);value.restype=result;value.argtypes=args;return value
        P=c.c_void_p;S=c.c_size_t;I=c.c_int64
        self.init=fn(self.base,'ggml_init',P,[InitParams]);self.free=fn(self.base,'ggml_free',None,[P])
        self.new=fn(self.base,'ggml_new_tensor_2d',P,[P,c.c_int,I,I])
        self.mul=fn(self.base,'ggml_mul_mat',P,[P,P,P]);self.graph_new=fn(self.base,'ggml_new_graph',P,[P])
        self.expand=fn(self.base,'ggml_build_forward_expand',None,[P,P])
        self.prec=fn(self.base,'ggml_prec_set_acc',c.c_bool,[P,c.c_int])
        self.backend_init=fn(self.vk,'ggml_backend_vk_init',P,[S])
        self.alloc=fn(self.base,'ggml_backend_alloc_ctx_tensors',P,[P,P])
        self.buffer_free=fn(self.base,'ggml_backend_buffer_free',None,[P])
        self.backend_free=fn(self.base,'ggml_backend_free',None,[P])
        self.set=fn(self.base,'ggml_backend_tensor_set',None,[P,P,S,S])
        self.get=fn(self.base,'ggml_backend_tensor_get',None,[P,P,S,S])
        self.compute=fn(self.base,'ggml_backend_graph_compute',c.c_int,[P,P])
        self.sync=fn(self.base,'ggml_backend_synchronize',None,[P])
        self.ctx=self.init(InitParams(16*1024*1024,None,True));self.backend=self.backend_init(0)
        if not self.ctx or not self.backend:raise RuntimeError('GGML context/backend initialization failed')
        self.o,self.k=weight.shape;self.r=inputs.shape[0]
        self.weight=self.new(self.ctx,1,self.k,self.o);self.input=self.new(self.ctx,0,self.k,self.r)
        self.outputs=[self.mul(self.ctx,self.weight,self.input) for _ in range(graph_repeats)]
        for output in self.outputs:
            if accumulator=='f32' and not self.prec(output,10):raise ValueError('FP32 accumulator request rejected')
        self.output=self.outputs[-1]
        self.graph=self.graph_new(self.ctx)
        for output in self.outputs:self.expand(self.graph,output)
        self.buffer=self.alloc(self.ctx,self.backend)
        if not self.buffer:raise RuntimeError('GGML device buffer allocation failed')
        self.upload(self.weight,np.ascontiguousarray(weight,dtype=np.float16))
        self.update(inputs)
    def upload(self,tensor,value):self.set(tensor,value.ctypes.data,0,value.nbytes)
    def update(self,inputs):
        # Explicit oracle rounding avoids a backend-dependent float->half cast.
        self.upload(self.input,np.ascontiguousarray(inputs.astype(np.float16).astype(np.float32)))
    def launch(self):
        status=self.compute(self.backend,self.graph)
        if status:raise RuntimeError(f'GGML graph status {status}')
    def synchronize(self):self.sync(self.backend)
    def download(self,index=-1):
        out=np.empty((self.r,self.o),dtype=np.float32);self.get(self.outputs[index],out.ctypes.data,0,out.nbytes);return out
    def close(self):
        self.buffer_free(self.buffer);self.free(self.ctx);self.backend_free(self.backend);self.dll_path.close()

def validate(actual,inputs,weight):
    lhs=inputs.astype(np.float16).astype(np.float64);rhs=weight.astype(np.float64)
    reference=lhs@rhs.T;bound=np.abs(lhs)@np.abs(rhs).T*3e-6+1e-10
    passed=bool(np.isfinite(actual).all() and np.all(np.abs(actual-reference)<=bound))
    return {'passed':passed,'maximum_absolute_error':float(np.max(np.abs(actual-reference))),
            'maximum_error_over_bound':float(np.max(np.abs(actual-reference)/bound))}

def run(model,directory,out,profile=False,weight_names=('ffn_gate','ffn_down'),graph_repeats=1,accumulator='f32'):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if profile:
        os.environ['GGML_VK_PERF_LOGGER']='1';os.environ['GGML_VK_PERF_LOGGER_FREQUENCY']='1'
    gguf=GGUF(model);records=[]
    for suffix in weight_names:
        weight=np.array(gguf.array('blk.0.'+suffix+'.weight',dtype=np.float16),copy=True);o,k=weight.shape
        inputs=(np.random.default_rng(29).normal(size=(32,k))*.01).astype(np.float32)
        runner=VulkanProjection(directory,weight,inputs,graph_repeats,accumulator)
        try:
            runner.launch();actual=runner.download();checks=[{'seed':29,'scale':.01,**validate(actual,inputs,weight)}]
            all_outputs=[validate(runner.download(i),inputs,weight) for i in range(graph_repeats)]
            samples=[]
            if profile:
                deadline=time.perf_counter()+1
                while time.perf_counter()<deadline:runner.launch()
                print('PROFILE_BEGIN',suffix,flush=True)
                for repeat in range(8):
                    print('PROFILE_BATCH',suffix,repeat,flush=True)
                    for _ in range(20 if graph_repeats==1 else 1):runner.launch()
                print('PROFILE_END',suffix,flush=True)
            else:
                deadline=time.perf_counter()+1
                while time.perf_counter()<deadline:
                    for _ in range(20):runner.launch()
                    runner.synchronize()
                for repeat in range(8):
                    start=time.perf_counter()
                    for _ in range(20):runner.launch()
                    runner.synchronize();elapsed=(time.perf_counter()-start)/20
                    if repeat:samples.append(elapsed)
            for seed,scale in ((101,.01),(202,1.0),(303,.00001)):
                values=(np.random.default_rng(seed).normal(size=(32,k))*scale).astype(np.float32)
                runner.update(values);runner.launch();checks.append({'seed':seed,'scale':scale,**validate(runner.download(),values,weight)})
            record={'weight':suffix,'shape':[32,k,o],'validation':checks,'completed_samples_seconds':samples,
                    'all_graph_outputs_validation':all_outputs,'graph_repeats':graph_repeats,
                    'median_completed_seconds':statistics.median(samples) if samples else None}
            records.append(record);print('result',record,flush=True)
        finally:runner.close()
    report={'status':'passed' if all(c['passed'] for r in records for c in r['validation']) else 'oracle_failed',
            'revision':'f872b591121761ac7b2af18283bd99bdc092a63a','records':records,'profile_enabled':profile,
            'precision':f'native F16 weights; host nearest-even F16-rounded inputs uploaded as F32; {accumulator} accumulator request; FP32 output',
            'accumulator':accumulator,
            'dll_sha256':hashlib.sha256((Path(directory)/'ggml-vulkan.dll').read_bytes()).hexdigest()}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','directory','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--profile',action='store_true');p.add_argument('--weights',nargs='+',default=['ffn_gate','ffn_down'])
    p.add_argument('--graph-repeats',type=int,default=1)
    p.add_argument('--accumulator',choices=('f32','default'),default='f32')
    a=p.parse_args();run(a.model,a.directory,a.out,a.profile,tuple(a.weights),a.graph_repeats,a.accumulator)
