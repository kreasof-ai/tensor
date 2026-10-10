"""H200 KV-load scheduling and streaming-copy diagnostics.

These microbenchmarks do not substitute for the unchanged C8 client replay.
"""
import modal
from benchmarks.qwen35.modal_h200 import image, volume, app as base_app, prepare as prepare_base
from benchmarks.qwen35.modal_hopper_long import app as serving_app, prepare_long, prepare_dense_tiles
from benchmarks.qwen35.modal_compact import measure_compact, prepare_compact

app = modal.App('tensor-qwen35-h200-bandwidth')
app.include(base_app).include(serving_app)
SCHEDULES = [(128,256,False),(128,256,True),(64,128,True),
             (128,128,True),(256,256,True)]


@app.function(image=image,cpu=16,memory=32768,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_micro():
    import hashlib,json,os
    from pathlib import Path
    from tensor.compiler.entry import export_source
    from benchmarks.qwen35.build import build_artifact
    os.chdir('/workspace');volume.reload()
    paths=[Path(__file__),Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_prefill_attention.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/bandwidth')/key;root.mkdir(parents=True,exist_ok=True)
    result=dict(root=str(root),source_hashes=hashes,candidates=[])
    for q,t,joint in SCHEDULES:
        schedule=dict(slots=8,chunk=512,capacity=48000,packed_loads=True,
                      query_rows=q,threads=t,joint_kv=joint)
        entry=root/f'attention-q{q}-t{t}-j{int(joint)}.py'
        entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_prefill_attention',
                         'attention',schedule,dependencies=('tensor.compiler.entry',)))
        artifact=entry.with_suffix('.tbin')
        row=dict(schedule=schedule,path=str(artifact))
        try:
            build_artifact(entry,artifact,target='sm_90a')
            row['compiled']=True
        except Exception as error:
            row.update(compiled=False,error=str(error))
        result['candidates'].append(row);print('Prepared KV candidate',row,flush=True)
    entry=root/'copy.py'
    entry.write_text('''import tilelang.language as T
@T.prim_func
def kernel(src:T.Tensor((268435456,), 'uint32'),dst:T.Tensor((268435456,), 'uint32')):
    with T.Kernel(65536,threads=256) as block:
        for i in T.Parallel(4096):dst[block*4096+i]=src[block*4096+i]
def tensor_export():return {'kernel':kernel,'outputs':['dst']}
''')
    result['copy']=str(entry.with_suffix('.tbin'))
    build_artifact(entry,Path(result['copy']),target='sm_90')
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def measure_micro(prepared):
    import ctypes as ct
    import json,os,shutil,subprocess
    from pathlib import Path
    import numpy as np
    import tensor
    from benchmarks.qwen35.kernel_timing import time_calls
    os.chdir('/workspace');volume.reload()
    result=dict(prepared=prepared,cases=[],timing='CUDA graph events; isolated kernels, not model throughput')
    profiler=shutil.which('ncu')
    if profiler is None:
        profiler=next((str(p) for p in Path('/usr/local').glob('cuda*/nsight-compute*/ncu')),None)
    result['nsight_compute']=profiler
    result['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True)
    rng=np.random.default_rng(90210)
    with tensor.Device() as dev:
        result['device']=dev.info
        if dev.info['name']!='NVIDIA H200':raise RuntimeError('H200 required')
        def timed(kernel,args,repeats=3,setup=None):
            calls=([setup] if setup else [])+[(kernel,args)]
            return time_calls(dev,calls,repeats=repeats)
        src=dev.empty((268435456,),'uint32');dst=dev.empty(src.shape,'uint32')
        fill=dev.driver.lib.cuMemsetD32_v2
        fill.argtypes=[ct.c_uint64,ct.c_uint,ct.c_size_t];fill.restype=ct.c_int
        dev.driver.call('cuMemsetD32_v2',src.pointer,123456789,268435456)
        copy=dev.load(prepared['copy'])
        samples=timed(copy,[src,dst],4)
        probe=(ct.c_uint*16)()
        dev.driver.call('cuMemcpyDtoH_v2',ct.cast(probe,ct.c_void_p),dst.pointer,ct.sizeof(probe))
        if any(x!=123456789 for x in probe):raise RuntimeError('streaming copy check failed')
        result['streaming_copy']=dict(bytes_read_per_launch=1073741824,bytes_written_per_launch=1073741824,
            milliseconds=samples,useful_read_write_GB_per_second=[2147483648/ms/1e6 for ms in samples],
            note='Useful copy traffic; not hardware DRAM counters or achieved model bandwidth')
        print('Streaming copy',result['streaming_copy'],flush=True)
        src.release();dst.release();copy.release()
        s,c,cap,d=8,512,48000,256;rows=s*c
        q=dev.from_numpy(rng.standard_normal((rows,16,d),dtype='float32'),dtype='bfloat16')
        caches=[dev.from_numpy(rng.integers(0,112,(s,2,cap,d),dtype='uint8')) for _ in range(2)]
        scales=[dev.from_numpy(rng.uniform(.001,.02,(s,2,cap,2)).astype('float32')) for _ in range(2)]
        projection=dev.from_numpy(rng.standard_normal((rows,8192),dtype='float32'))
        out=dev.empty((rows,4096),'bfloat16')
        control=dev.load(prepared['candidates'][0]['path'])
        for context in (0,32000):
            pos=dev.from_numpy(np.full(s,context,'int32'))
            lengths=dev.from_numpy(np.full(s,c,'int32'))
            args=[q,*caches,*scales,projection,pos,lengths,out]
            control.launch(*args);expected=out.to_numpy()
            record=dict(context=context,candidates=[])
            for candidate in prepared['candidates']:
                if not candidate['compiled']:continue
                kernel=dev.load(candidate['path'])
                scratch=[];decoder=None
                try:
                    selected=args;setup=None
                    if candidate['schedule'].get('decoded_kv'):
                        decoder=dev.load(prepared['decode'])
                        scratch=[dev.empty((s,2,cap,d),'bfloat16') for _ in range(2)]
                        decode_args=[*caches,*scales,pos,lengths,*scratch]
                        decoder.launch(*decode_args)
                        selected=[q,*scratch,projection,pos,lengths,out]
                        setup=(decoder,decode_args)
                    kernel.launch(*selected);actual=out.to_numpy()
                    row=dict(schedule=candidate['schedule'],bitwise_equal=bool(np.array_equal(actual,expected)),
                        finite=bool(np.isfinite(actual).all()),
                        relative_rms=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)))
                    row['milliseconds']=timed(kernel,selected,setup=setup) if row['finite'] else None
                    if setup:row['timing_includes_workspace_decode']=True
                    record['candidates'].append(row);print('KV timing',context,row,flush=True)
                finally:
                    kernel.release()
                    if decoder:decoder.release()
                    for buffer in scratch:buffer.release()
            result['cases'].append(record);pos.release();lengths.release()
        control.release()
    (Path(prepared['root'])/'measured.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def main(out:str='build/qwen35-h200-bandwidth'):
    import json
    from pathlib import Path
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare_micro.remote();(root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    result=measure_micro.remote(ready);(root/'measured.json').write_text(json.dumps(result,indent=2)+'\n')


@app.function(image=image,cpu=16,memory=32768,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_decoded_micro(prepared):
    import hashlib,json,os
    from pathlib import Path
    from tensor.compiler.entry import export_source
    from benchmarks.qwen35.build import build_artifact
    os.chdir('/workspace');volume.reload()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_prefill_attention.py'),
        Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/attention_workspace.py'))}
    key=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/bandwidth-decoded')/key;root.mkdir(parents=True,exist_ok=True)
    result=dict(prepared,root=str(root),decoded_source_hashes=hashes,candidates=list(prepared['candidates']))
    for name,module,factory,schedule in (
        ('decode','attention_workspace','decode',dict(slots=8,capacity=48000)),
        ('attention','hopper_prefill_attention','attention',dict(slots=8,chunk=512,capacity=48000,
                packed_loads=True,query_rows=128,threads=256,decoded_kv=True,joint_kv=False)),
        ('attention-split2','hopper_prefill_attention','attention',dict(slots=8,chunk=512,capacity=48000,
                packed_loads=True,query_rows=128,threads=256,decoded_kv=True,joint_kv=False,value_splits=2)),
        ('attention-512','hopper_prefill_attention','attention',dict(slots=8,chunk=512,capacity=48000,
                packed_loads=True,query_rows=128,threads=512,decoded_kv=True,joint_kv=False,square_warps=True)),
        ('attention-q64-256','hopper_prefill_attention','attention',dict(slots=8,chunk=512,capacity=48000,
                packed_loads=True,query_rows=64,threads=256,decoded_kv=True,joint_kv=False))):
        entry=root/(name+'.py');artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.'+module,factory,schedule,
                         dependencies=('tensor.compiler.entry',)))
        build_artifact(entry,artifact,target='sm_90a')
        if name=='decode':result['decode']=str(artifact)
        else:result['candidates'].append(dict(schedule=schedule,path=str(artifact),compiled=True))
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def decoded_micro_main(prepared_file:str,out:str='build/qwen35-h200-decoded-micro'):
    import json
    from pathlib import Path
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare_decoded_micro.remote(json.loads(Path(prepared_file).read_text()))
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    result=measure_micro.remote(ready);(root/'measured.json').write_text(json.dumps(result,indent=2)+'\n')


@app.function(image=image,cpu=16,memory=32768,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_selected(prepared,query_rows=128,threads=256,joint=True):
    """Replace only prefill attention, retaining the measured projection kernels."""
    import hashlib,json,os,shutil
    from pathlib import Path
    from tensor.compiler.entry import export_source
    from tensor.artifacts.format import read_artifact
    from benchmarks.qwen35.build import build_artifact
    os.chdir('/workspace');volume.reload()
    source=Path(prepared['hopper']);control=Path(prepared['prefill'])
    manifest=json.loads((source/'hopper.json').read_text())
    original=json.loads((control/'prefill.json').read_text())
    schedule=dict(query_rows=query_rows,threads=threads,joint_kv=joint,packed_loads=True)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_prefill_attention.py'))}
    key=hashlib.sha256(json.dumps(dict(source_sha256=hashlib.sha256((source/'hopper.json').read_bytes()).hexdigest(),
                schedule=schedule,hashes=hashes),sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/bandwidth-selected')/key;root.mkdir(parents=True,exist_ok=True)
    count=0
    for name,row in manifest['kernels'].items():
        path=(source/row['path']).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('source Hopper checksum mismatch')
        destination=root/row['path']
        if row['kind']!='attention':
            shutil.copy2(path,destination);continue
        old=original['kernels'][name];artifact=control/old['path']
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=old['sha256']:
            raise ValueError('control checksum mismatch')
        descriptor,_=read_artifact(artifact)
        if row['original_cubin_sha256']!=descriptor['files']['kernel.cubin']:
            raise ValueError('attention control cubin mismatch')
        entry=destination.with_suffix('.py')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_prefill_attention',
                'attention',dict(old['parameters'],**schedule),dependencies=('tensor.compiler.entry',)))
        destination.unlink(missing_ok=True);build_artifact(entry,destination,target='sm_90a')
        row['sha256']=hashlib.sha256(destination.read_bytes()).hexdigest();count+=1
    if not count:raise ValueError('prefill attention absent')
    manifest['attention_schedule']=schedule;manifest['attention_source_hashes']=hashes
    (root/'hopper.json').write_text(json.dumps(manifest,indent=2)+'\n')
    result=dict(prepared,hopper=str(root),source_identity=key,
                bandwidth_source_hashes=hashes,profile_only=False,profile_phases=False)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def replay_main(prepared_file:str,query_rows:int=128,threads:int=256,joint:bool=True,
                out:str='build/qwen35-h200-bandwidth-replay'):
    import json
    from datetime import datetime,timezone
    from pathlib import Path
    from benchmarks.qwen35.modal_compact import measure_compact
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare_selected.remote(json.loads(Path(prepared_file).read_text()),query_rows,threads,joint)
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-bandwidth-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


@app.local_entrypoint()
def bootstrap_main(out:str='build/qwen35-h200-bandwidth-bootstrap'):
    """Prepare the selected architecture in a new workspace without baseline jobs."""
    import json
    from pathlib import Path
    from benchmarks.qwen35.modal_h200 import prepare as prepare_base
    from benchmarks.qwen35.modal_compact import prepare_compact
    from benchmarks.qwen35.modal_hopper_long import prepare_long,prepare_dense_tiles
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    base=prepare_base.remote();(root/'base.json').write_text(json.dumps(base,indent=2)+'\n')
    compact=prepare_compact.remote(base)
    ready=prepare_long.remote(compact,64,'bf16-pairs-attention')
    ready=prepare_dense_tiles.remote(ready,False)
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')


@app.function(image=image,cpu=16,memory=32768,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_workspace(prepared,query_rows=128,threads=256,value_splits=1):
    import hashlib,json,os
    from pathlib import Path
    from benchmarks.qwen35.attention_workspace_producer import produce
    os.chdir('/workspace');volume.reload()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__),Path('benchmarks/qwen35/attention_workspace_producer.py'),
        Path('packages/tensor-llm/src/tensor_llm/qwen35/attention_workspace.py'),
        Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/attention_workspace.py'),
        Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_prefill_attention.py'))}
    key=hashlib.sha256(json.dumps(dict(prepared=prepared,hashes=hashes,
                    query_rows=query_rows,threads=threads,value_splits=value_splits),sort_keys=True).encode()).hexdigest()[:16]
    root=produce(prepared,Path('/cache/attention-workspace')/key,query_rows,threads,value_splits)
    result=dict(prepared,attention_workspace=str(root),source_identity=key,
                bandwidth_source_hashes=hashes,profile_only=False,profile_phases=False)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def workspace_main(prepared_file:str,query_rows:int=128,threads:int=256,value_splits:int=1,
                   out:str='build/qwen35-h200-attention-workspace'):
    import json
    from datetime import datetime,timezone
    from pathlib import Path
    from benchmarks.qwen35.modal_compact import measure_compact
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare_workspace.remote(json.loads(Path(prepared_file).read_text()),query_rows,threads,value_splits)
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-workspace-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
