"""Matched LFM2 comparison against an experimental tinygrad adapter.

Upstream tinygrad does not implement LFM2. This uses its generic Tensor/JIT
compiler on the same checkpoint and precision contract, not custom AMD kernels.
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,os,statistics,subprocess,time
from contextlib import ExitStack
from pathlib import Path
import numpy as np
import tensor
from tensor_llm import LFM2
from benchmarks.lfm2.webgpu_run import cached_reference,metrics
from benchmarks.lfm2.vulkan_reference import Reference as NativeReference,COMMIT
from benchmarks.lfm2.tinygrad_reference import Reference as TinyReference


def tiny_info():
    import tinygrad
    from tinygrad import Device
    from tinygrad.helpers import CACHEDB
    device=Device[Device.DEFAULT]
    repository=Path(tinygrad.__file__).resolve().parents[1]
    result={'revision':subprocess.check_output(['git','-C',str(repository),'rev-parse','HEAD'],text=True).strip(),
            'device':Device.DEFAULT,'renderer':str(device.renderer.target),
            'beam':int(os.environ.get('BEAM','0')),'jitbeam':int(os.environ.get('JITBEAM',os.environ.get('BEAM','0'))),
            'cache':CACHEDB}
    result['search_controls']={name:os.environ[name] for name in ('BEAM_ESTIMATE','BEAM_UPCAST_MAX','BEAM_LOCAL_MAX','NOOPT') if name in os.environ}
    if Device.DEFAULT.startswith('CL'):
        result.update(name=device.device_name,driver=device.driver_version)
        if device.device_name!='gfx1031':raise RuntimeError('requires the recorded Radeon adapter')
    elif Device.DEFAULT.startswith('WEBGPU'):
        from tinygrad.runtime.ops_webgpu import webgpu,instance,InstanceRequestAdapter,backend_types,from_wgpu_str
        adapter=InstanceRequestAdapter(instance,webgpu.WGPURequestAdapterOptions(powerPreference=webgpu.WGPUPowerPreference_HighPerformance,
            backendType=backend_types.get(os.environ.get('WEBGPU_BACKEND',''),0)))
        info=webgpu.WGPUAdapterInfo();webgpu.wgpuAdapterGetInfo(adapter,info)
        result.update({name:from_wgpu_str(getattr(info,name)) for name in ('vendor','architecture','description')})
        result['name']=from_wgpu_str(info.device)
        result['backend']=webgpu.enum_WGPUBackendType[info.backendType]
        result['vendor_id']=info.vendorID;result['device_id']=info.deviceID
        webgpu.wgpuAdapterInfoFreeMembers(info);webgpu.wgpuAdapterRelease(adapter)
        if result['backend']!='WGPUBackendType_Vulkan' or result['vendor_id']!=4098 or result['device_id']!=29663:
            raise RuntimeError('requires physical RX 6700 XT Vulkan adapter')
    else:raise RuntimeError('comparison requires CL or Vulkan WEBGPU')
    return result


def check(name,actual,expected):
    result=metrics(actual,expected)
    if not np.all(np.isfinite(actual)) or result['relative_rms']>=.01 or result['cosine']<=.9999 or result['argmax'][0]!=result['argmax'][1]:
        raise AssertionError((name,result))
    return result


def run(model,bundle,reference,fixtures,out,*,repeats=5,decode=64,weight_mode='packed',validation_only=False,baseline_bundle=None,tinygrad_search_root=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    cases=[{key:case[key] for key in ('name','tokens','reset')} for case in json.loads((Path(fixtures)/'report.json').read_text())['validation']]
    expected,oracle=cached_reference(fixtures,model,cases)
    start=time.perf_counter();tiny=TinyReference(model,weight_mode=weight_mode);load=time.perf_counter()-start
    tiny_searched=TinyReference(model,weight_mode=weight_mode,search_root=tinygrad_search_root) if tinygrad_search_root else None
    info=tiny_info();print('tinygrad',info,'load seconds',load,flush=True)
    native=NativeReference(model,reference,context=512)
    try:
        with ExitStack() as stack:
            device=stack.enter_context(tensor.Device(provider='webgpu'))
            engine=stack.enter_context(LFM2(model,bundle,device,context=512))
            if device.info['adapter']['backend_type']!='Vulkan':raise RuntimeError('requires Vulkan')
            first_calls={};prompt=cases[0]['tokens'];runners={'tensor':engine,'tinygrad':tiny,'llama_cpp':native}
            if baseline_bundle is not None:
                runners['tensor_baseline']=stack.enter_context(LFM2(model,baseline_bundle,device,context=512))
            if tiny_searched is not None:runners['tinygrad_searched']=tiny_searched
            for name,runner in runners.items():
                runner.reset();start=time.perf_counter();actual=runner.forward(prompt);elapsed=time.perf_counter()-start
                comparison=metrics(actual,expected[0]) if name=='llama_cpp' else check('first_'+name,actual,expected[0])
                first_calls[name]={'seconds':elapsed,'numpy':comparison}
                print('first_call',name,elapsed,flush=True)
            validation=[];first={}
            for i,case in enumerate(cases):
                row={**case,'completed_seconds':{}}
                for name,runner in runners.items():
                    if case['reset']:runner.reset()
                    start=time.perf_counter();actual=runner.forward(case['tokens'])
                    row['completed_seconds'][name]=time.perf_counter()-start
                    np.save(out/f'{i}-{name}-logits.npy',actual)
                    try:
                        row[name]=metrics(actual,expected[i]) if name=='llama_cpp' else check(case['name']+'_'+name,actual,expected[i])
                    except AssertionError:
                        failure={'status':'failed','model':str(Path(model).resolve()),'model_sha256':oracle['model_sha256'],
                                 'tinygrad':info,'first_call':first_calls,'independent_reference':oracle,
                                 'tinygrad_schedule_replay':tiny_searched.replay.report() if tiny_searched is not None else None,
                                 'validation':validation,'failed_fixture':{**row,'runner':name,'metrics':metrics(actual,expected[i])}}
                        (out/'report.json').write_text(json.dumps(failure,indent=2)+'\n')
                        raise
                    if i==0:first[name]=actual.copy()
                    if case['name']=='reset_chat' and not np.array_equal(first[name],actual):raise AssertionError('reset must be bitwise identical: '+name)
                validation.append(row);print('validation',case['name'],row['tinygrad'],flush=True)
            benchmarks=[];forced=np.resize(prompt,decode).tolist()
            if tiny_searched is not None:
                matched={row['weight_shape'] for row in tiny_searched.replay.records.values()}
                if matched!={'ffn_gate','ffn_down'}:raise RuntimeError('searched tinygrad FFN ASTs did not both match: '+str(matched))
            report={'status':'measuring','model':str(Path(model).resolve()),'model_sha256':oracle['model_sha256'],
                    'tinygrad':info,'tinygrad_adapter_sha256':hashlib.sha256(Path(__file__).with_name('tinygrad_reference.py').read_bytes()).hexdigest(),
                    'weight_mode':weight_mode,'weight_storage':{'f16_gguf':'native FP16','f32_gguf':'native FP32'},
                    'adapter':device.info,'implementation':engine.manifest['implementation'],
                    'tensor_profiles':{name:runner.webgpu_profile for name,runner in runners.items() if name.startswith('tensor')},
                    'tensor_bundles':{name:runner.manifest for name,runner in runners.items() if name.startswith('tensor')},
                    'tinygrad_schedule_replay':tiny_searched.replay.report() if tiny_searched is not None else None,
                    'native_commit':COMMIT,'independent_reference':oracle,'loading_seconds':{'tinygrad':load},
                    'first_call':first_calls,'validation':validation,'benchmarks':benchmarks,
                    'protocol':{'context':512,'prefill_chunk':32,'warmups':3,'repeats':repeats,'host_logits':'FP32','sampling':'excluded',
                                'loading':'excluded from warmed timings','order':'rotate all runners, sequential GPU execution',
                                'first_call':'includes graph/kernel compilation where needed; isolated persistent tinygrad cache; Tensor producer AOT compilation excluded',
                                'tinygrad_attention':'fixed 576-entry FP16 KV buffers with causal mask; generic compiler, no upstream AMD LLM kernels'}}
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            for length in (() if validation_only else (32,128,384)):
                prefix=np.resize(prompt,length).tolist();samples={name:[] for name in runners};names=list(runners)
                # Three warmups allow tinygrad's initial call and JIT capture
                # to complete before timing either static row profile.
                for repeat in range(repeats+3):
                    order=names[repeat%len(names):]+names[:repeat%len(names)]
                    for name in order:
                        runner=runners[name];runner.reset();start=time.perf_counter();runner.forward(prefix);prefill=time.perf_counter()-start
                        start=time.perf_counter()
                        for token in forced:runner.forward([token])
                        elapsed=time.perf_counter()-start
                        if repeat>=3:samples[name].append({'prefill_seconds':prefill,'decode_seconds':elapsed})
                row={'prompt_tokens':length,'decode_tokens':decode,'samples':samples}
                for name,values in samples.items():
                    row[name]={'prefill_tokens_per_second':length/statistics.median(x['prefill_seconds'] for x in values),
                               'decode_tokens_per_second':decode/statistics.median(x['decode_seconds'] for x in values)}
                benchmarks.append(row);print('benchmark',length,{name:row[name] for name in names},flush=True)
                (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            report['status']='passed'
            (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:native.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','reference','fixtures','out'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--repeats',type=int,default=5);p.add_argument('--decode',type=int,default=64)
    p.add_argument('--weight-mode',choices=('packed','decoded'),default='packed')
    p.add_argument('--validation-only',action='store_true')
    p.add_argument('--baseline-bundle',type=Path)
    p.add_argument('--tinygrad-search-root',type=Path)
    a=p.parse_args()
    if a.repeats<1 or not 1<=a.decode<=128:p.error('requires positive repeats and 1..128 decode tokens')
    run(a.model,a.bundle,a.reference,a.fixtures,a.out,repeats=a.repeats,decode=a.decode,weight_mode=a.weight_mode,validation_only=a.validation_only,baseline_bundle=a.baseline_bundle,tinygrad_search_root=a.tinygrad_search_root)
