"""Build bounded native batch profiles for the unchanged 32K/16K stress load."""
from pathlib import Path
import modal
from benchmarks.qwen35.modal_h200 import image as base_image, volume, ROOT, REVISION

app = modal.App('tensor-qwen35-h200-batch-scaling')
image = base_image.add_local_file(
    ROOT/'docs/research/data/qwen35-native-h200/scaling/workload-c64.json.gz',
    '/workspace/workload-c64.json.gz')


@app.function(image=image, cpu=4, memory=16384, timeout=7200,
              volumes={'/cache': volume}, scaledown_window=2)
def prepare(slots):
    import hashlib, json, os, time
    from benchmarks.qwen35.profile_producer import produce as decoder
    from benchmarks.qwen35.prefill_producer import produce as prefill
    from benchmarks.qwen35.compact_producer import produce as compact
    from benchmarks.qwen35.hopper_producer import produce as hopper
    from benchmarks.qwen35.mtp_producer import produce as mtp
    from benchmarks.qwen35.mtp_prefill_producer import produce as mtp_prefill
    from benchmarks.qwen35.spec_producer import produce as verifier
    from benchmarks.qwen35.spec_attention_producer import produce as attention
    from benchmarks.qwen35.spec_linear_producer import produce as linear
    from benchmarks.qwen35.hopper_verify_producer import produce as experts, produce_dense
    from benchmarks.qwen35.recompute_producer import produce as recompute
    from benchmarks.qwen35.attention_workspace_producer import produce as workspace, produce_verification
    from benchmarks.qwen35.modal_dense_m128 import prepare as dense_m128
    os.chdir('/workspace'); volume.reload()
    if slots not in (8,16,32,64): raise ValueError('bounded batch sweep required')
    sources=[*Path('src/tensor').rglob('*.py'),
             *Path('packages/tensor-llm/src').rglob('*.py'),
             *Path('benchmarks/qwen35').glob('*.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    identity=hashlib.sha256(json.dumps(dict(slots=slots,sources=hashes),sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/batch-scaling')/identity
    ready=root/'prepared.json'
    if ready.is_file(): return json.loads(ready.read_text())
    root.mkdir(parents=True,exist_ok=True); started=time.perf_counter()
    checkpoint=Path('/cache/models')/REVISION
    # Keep the total prefill rows at 16K rather than multiplying live activations.
    chunk=16384//slots; window=128; paths={}
    paths['decoder']=decoder(checkpoint,root/'decode-profile',slots=slots,context=48000,kv_dtype='fp8',target='sm_90')
    paths['draft']=mtp(checkpoint,paths['decoder'],root/'mtp')
    original=prefill(checkpoint,root/'prefill',slots=slots,chunk=chunk,context=48000,
                     kv_dtype='fp8',block_m=64,packed_kv=True,target='sm_90')
    selected=compact(original,root/'compact')
    wide=hopper(selected,root/'hopper',expert_mma='bf16-pairs',hopper_attention=True)
    paths['draft_prefill']=mtp_prefill(checkpoint,original,root/'mtp-prefill')
    raw=prefill(checkpoint,root/'prefill-verify',slots=slots,chunk=window,context=48000,
                kv_dtype='fp8',block_m=64,packed_kv=True,target='sm_90')
    frozen=verifier(raw,root/'verify-snapshots')
    split=attention(frozen,root/'verify-attention',key_rows=32,query_tokens=8,kernel_target='sm_90a')
    split=linear(split,root/'verify-linear',expert_block_m=32)
    split=experts(split,root/'verify-experts',block_m=64,columns=128,paired=False)
    control=produce_dense(split,root/'verify-control')
    produce_verification(control,control/'attention-workspace')
    manifest=json.loads((control/'prefill.json').read_text());manifest['attention_workspace']='attention-workspace'
    (control/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n')
    paths['verify']=recompute(control,root/'verify')
    repair=mtp_prefill(checkpoint,raw,root/'repair')
    paths['repair']=attention(repair,root/'repair-attention',key_rows=32,query_tokens=8,kernel_target='sm_90a')
    result=dict(slots=slots,prefill_chunk=chunk,verification_chunk=window,state_recompute=True,
                base=dict(checkpoint=str(checkpoint),paths={k:str(v) for k,v in paths.items()}),
                prefill=str(selected),hopper=str(wide),source_identity=identity,
                source_hashes=hashes,resident_speculative_graphs=True,adaptive_verification=False)
    result['attention_workspace']=str(workspace(result,root/'attention-workspace'))
    result=dense_m128.local(result)
    result['preparation_seconds']=time.perf_counter()-started
    ready.write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def prepare_main(out: str='build/qwen35-h200-akbar-scaling'):
    import json
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    handles={slots:prepare.spawn(slots) for slots in (8,16,32,64)}
    for slots,handle in handles.items():
        result=handle.get()
        (root/f'prepared-c{slots}.json').write_text(json.dumps(result,indent=2)+'\n')
        print('Prepared batch',slots,result['source_identity'],flush=True)


@app.function(image=image,cpu=4,memory=16384,timeout=7200,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_short(prepared):
    """Add a four-position fallback without changing the large verifier."""
    import hashlib,json,os,shutil,time
    from benchmarks.qwen35.prefill_producer import produce as prefill
    from benchmarks.qwen35.spec_producer import produce as verifier
    from benchmarks.qwen35.spec_attention_producer import produce as attention
    from benchmarks.qwen35.spec_linear_producer import produce as linear
    from benchmarks.qwen35.hopper_verify_producer import produce as experts,produce_dense
    from benchmarks.qwen35.mtp_prefill_producer import produce as repair
    from tensor_llm.qwen35.speculative.graph_pool import implementation_hashes
    os.chdir('/workspace');volume.reload()
    if prepared.get('adaptive_verification'):raise ValueError('parent already has an adaptive pool')
    sources=[*Path('src/tensor').rglob('*.py'),*Path('packages/tensor-llm/src').rglob('*.py'),
             *Path('benchmarks/qwen35').glob('*.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    identity=hashlib.sha256(json.dumps(dict(parent=prepared,sources=hashes),sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/batch-short')/identity;ready=root/'prepared.json'
    if ready.exists():return json.loads(ready.read_text())
    root.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    slots=prepared['slots'];paths=dict(prepared['base']['paths']);control=Path(paths['verify'])
    selected=root/'verify';shutil.copytree(control,selected,dirs_exist_ok=True)
    raw=prefill(prepared['base']['checkpoint'],root/'prefill4',slots=slots,chunk=4,
        context=48000,kv_dtype='fp8',block_m=64,packed_kv=True,target='sm_90')
    small=verifier(raw,root/'snapshots4')
    small=attention(small,root/'attention4',key_rows=32,query_tokens=8,kernel_target='sm_90a',pad_queries=True)
    small=linear(small,root/'linear4',expert_block_m=32)
    small=experts(small,root/'experts4',block_m=64,columns=128,paired=False)
    produce_dense(small,selected/'profiles/4/verify')
    draft=repair(prepared['base']['checkpoint'],raw,root/'repair4')
    attention(draft,selected/'profiles/4/repair',key_rows=32,query_tokens=8,kernel_target='sm_90a',pad_queries=True)
    manifest=json.loads((selected/'prefill.json').read_text())
    manifest['adaptive_verification']=dict(implementation=implementation_hashes(),profiles=[dict(
        window=4,verify='profiles/4/verify',repair='profiles/4/repair')])
    (selected/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n')
    paths['verify']=str(selected)
    result=dict(prepared,base=dict(prepared['base'],paths=paths),source_identity=identity,
        adaptive_verification=True,adaptive_windows=[4,128],adaptive_control_bundle=str(control),
        adaptive_source_hashes=hashes,resident_speculative_graphs=slots<=32,
        adaptation_reason='C16 fixed-window tail: 15 requests completed in 84-95 s; final request extended cohort to 410.67 s',
        short_profile_preparation_seconds=time.perf_counter()-started)
    ready.write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result


@app.local_entrypoint()
def short_main(prepared_dir: str='build/qwen35-h200-akbar-scaling',slots: int=32):
    import json
    root=Path(prepared_dir);path=root/f'prepared-c{slots}.json'
    parent=json.loads(path.read_text());(root/f'prepared-c{slots}-fixed.json').write_text(json.dumps(parent,indent=2)+'\n')
    path.write_text(json.dumps(prepare_short.remote(parent),indent=2)+'\n')


@app.function(image=image, gpu='H200', cpu=8, memory=65536, timeout=3600,
              volumes={'/cache':volume}, scaledown_window=2)
def qualify_scaling():
    import os,sys
    from benchmarks.qwen35.qualification_cache import qualify
    os.chdir('/workspace');volume.reload()
    command=[sys.executable,'-m','pytest','packages/tensor-llm/tests/test_qwen_kernels.py',
             'packages/tensor-llm/tests/test_qwen_batch_scaling.py',
             'packages/tensor-llm/tests/test_qwen_recompute.py',
             'packages/tensor-llm/tests/test_qwen_spec.py',
             'packages/tensor-llm/tests/test_qwen_graph_pool.py','-q','-o','addopts=']
    result,proof=qualify(command);volume.commit()
    print(result.stdout,result.stderr,flush=True)
    if result.returncode:raise RuntimeError('physical batch scaling checks failed')
    return dict(proof=proof,command=command,exit_code=result.returncode,log=result.stdout+result.stderr)


@app.local_entrypoint()
def tests_main(out: str='build/qwen35-h200-akbar-scaling/kernel-proof.json'):
    import json
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(qualify_scaling.remote(),indent=2)+'\n')
