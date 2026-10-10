"""Qualify and measure transient BF16 scratch for split-context verification."""
import modal
from benchmarks.qwen35.modal_h200 import image,volume

app=modal.App('tensor-qwen35-h200-verification-workspace')


@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def measure(prepared):
    import hashlib,json,os,subprocess,sys
    from pathlib import Path
    import numpy as np
    import tensor
    from benchmarks.qwen35.kernel_timing import time_calls
    os.chdir('/workspace');volume.reload()
    result=dict(scope='isolated verification attention; not complete model throughput',
        prepared=prepared,cases=[],source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__),Path('benchmarks/qwen35/kernel_timing.py'),
                Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_attention.py'),
                Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/attention_workspace.py'),
                Path('packages/tensor-llm/tests/test_qwen_hopper.py'))})
    command=[sys.executable,'-m','pytest','packages/tensor-llm/tests/test_qwen_hopper.py',
             '-k','query_tiles and True','-q','-o','addopts=']
    test=subprocess.run(command,env=dict(os.environ,TENSOR_QWEN_CUDA='1'),
                        text=True,capture_output=True,timeout=600)
    result['kernel_checks']=dict(command=command,exit_code=test.returncode,log=test.stdout+test.stderr)
    print(result['kernel_checks']['log'],flush=True)
    root=Path(prepared['workspace']);manifest=json.loads((root/'attention-workspace.json').read_text())
    for row in manifest['kernels'].values():
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('workspace artifact checksum mismatch')
    if test.returncode:raise RuntimeError('verification workspace primitive checks failed')
    rng=np.random.default_rng(29083);s,c,cap,d=8,64,48000,256;rows=s*c;splits=16
    with tensor.Device() as dev:
        result['device']=dev.info
        if dev.info['name']!='NVIDIA H200':raise RuntimeError('H200 required')
        q=dev.from_numpy(rng.standard_normal((rows,16,d),dtype='float32'),dtype='bfloat16')
        cache_arrays=[rng.integers(0,127,(s,2,cap,d),dtype='uint8') for _ in range(2)]
        scale_arrays=[rng.uniform(.001,.02,(s,2,cap,2)).astype('float32') for _ in range(2)]
        parts=dev.empty((s,2,splits,c*8,d));stats=dev.empty((s,2,splits,c*8,2))
        scratch=[dev.empty((s,2,cap,d),'bfloat16') for _ in range(2)]
        control=dev.load(prepared['control'])
        decoder=dev.load(root/manifest['kernels']['decode']['path'])
        candidate=dev.load(root/manifest['kernels']['attention']['path'])
        for context in (0,32000,47936):
            lengths=np.array([64,63,37,0,1,9,64,48],'int32')
            pos=dev.from_numpy(np.full(s,context,'int32'));lens=dev.from_numpy(lengths)
            # Poison all future cache rows. A valid result must never use them.
            caches=[];scales=[]
            for original in cache_arrays:
                a=original.copy()
                for slot,count in enumerate(lengths):a[slot,:,context+count:]=127
                caches.append(dev.from_numpy(a))
            for original in scale_arrays:
                a=original.copy()
                for slot,count in enumerate(lengths):a[slot,:,context+count:]=np.nan
                scales.append(dev.from_numpy(a))
            control_args=[q,*caches,*scales,pos,lens,parts,stats]
            decode_args=[*caches,*scales,pos,lens,*scratch]
            candidate_args=[q,*scratch,pos,lens,parts,stats]
            control.launch(*control_args);expected_parts=parts.to_numpy();expected_stats=stats.to_numpy()
            decoder.launch(*decode_args);candidate.launch(*candidate_args)
            actual_parts=parts.to_numpy();actual_stats=stats.to_numpy()
            row=dict(context=context,lengths=lengths.tolist(),
                bitwise_equal=bool(np.array_equal(expected_parts,actual_parts)
                    and np.array_equal(expected_stats,actual_stats)),
                finite=bool(np.isfinite(actual_parts).all()),
                relative_rms=float(np.linalg.norm(actual_parts-expected_parts)/max(np.linalg.norm(expected_parts),1e-20)),
                control_milliseconds=time_calls(dev,[(control,control_args)]),
                candidate_milliseconds=time_calls(dev,[(decoder,decode_args),(candidate,candidate_args)]),
                timing_includes_workspace_decode=True,workspace_bytes=sum(b.nbytes for b in scratch))
            result['cases'].append(row);print('Verification workspace',row,flush=True)
            for buffer in (*caches,*scales,pos,lens):buffer.release()
    (root/'micro.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def main(prepared_file:str,out:str='build/qwen35-h200-verification-workspace-micro.json'):
    import json
    from pathlib import Path
    result=measure.remote(json.loads(Path(prepared_file).read_text()))
    Path(out).write_text(json.dumps(result,indent=2)+'\n')
