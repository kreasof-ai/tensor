"""Prepare two graph widths over one model for short MTP fallback batches."""
import modal
from benchmarks.qwen35.modal_h200 import image,volume
from benchmarks.qwen35.modal_compact import app as replay_app,measure_compact

app=modal.App('tensor-qwen35-h200-verifier-pool')
app.include(replay_app)


@app.function(image=image,cpu=16,memory=32768,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def prepare(prepared):
    import hashlib,json,os,shutil
    from pathlib import Path
    from tensor_llm.qwen35.speculative.graph_pool import implementation_hashes
    from benchmarks.qwen35.spec_attention_producer import produce as attention
    from benchmarks.qwen35.hopper_verify_producer import produce_dense,produce as experts
    os.chdir('/workspace');volume.reload()
    sources=[Path(__file__),Path('benchmarks/qwen35/spec_run.py'),
             *Path('packages/tensor-llm/src/tensor_llm/qwen35').rglob('*.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    identity=hashlib.sha256(json.dumps(dict(prepared=prepared,sources=hashes),sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/verifier-pool')/identity;root.mkdir(parents=True,exist_ok=True)
    paths=dict(prepared['base']['paths']);source=Path(paths['verify']);selected=root/'verify'
    # Copy the parent and its embedded attention workspace with their manifests.
    manifest=json.loads((source/'prefill.json').read_text())
    if 'adaptive_verification' in manifest:raise ValueError('parent is already pooled')
    for row in manifest['kernels'].values():
        path=(source/row['path']).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('parent artifact checksum mismatch')
    shutil.copytree(source,selected,dirs_exist_ok=True)
    base=Path('/cache/bundles/85cbb54120e2258b')
    small=attention(base/'verify8-selected',root/'attention8',key_rows=32,query_tokens=8,kernel_target='sm_90a')
    expert_schedule=manifest['hopper_expert_schedule']
    paired=expert_schedule.get('paired',expert_schedule['shared_mma_dtype']=='float16')
    small=experts(small,root/'experts8',block_m=expert_schedule['block_m'],
                  columns=expert_schedule['columns'],paired=paired)
    produce_dense(small,selected/'profiles/8/verify')
    attention(base/'repair8-attention',selected/'profiles/8/repair',key_rows=32,query_tokens=8,kernel_target='sm_90a')
    manifest['adaptive_verification']=dict(implementation=implementation_hashes(),profiles=[dict(
        window=8,verify='profiles/8/verify',repair='profiles/8/repair')])
    (selected/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n')
    control=paths['verify'];paths['verify']=str(selected)
    result=dict(prepared,base=dict(prepared['base'],paths=paths),adaptive_verification=True,
                adaptive_windows=[8,manifest['chunk']],adaptive_control_bundle=control,
                adaptive_expert_schedule=expert_schedule,
                source_identity=identity,adaptive_source_hashes=hashes)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit();return result


@app.local_entrypoint()
def main(prepared_file:str,out:str='build/qwen35-h200-verifier-pool',prepare_only:bool=False):
    from datetime import datetime,timezone
    import json
    from pathlib import Path
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare.remote(json.loads(Path(prepared_file).read_text()))
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    if prepare_only:return
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-pool-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
