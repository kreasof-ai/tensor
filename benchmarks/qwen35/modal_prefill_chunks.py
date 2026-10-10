"""Prepare wider prefill chunks while retaining the frozen 512-row control."""
import modal
from benchmarks.qwen35.modal_h200 import image,volume
from benchmarks.qwen35.modal_compact import app as replay_app,measure_compact

app=modal.App('tensor-qwen35-h200-prefill-chunks')
app.include(replay_app)


@app.function(image=image,cpu=16,memory=98304,timeout=1200,
              volumes={'/cache':volume},scaledown_window=2)
def prepare(prepared,chunk):
    import hashlib,json,os
    from pathlib import Path
    from benchmarks.qwen35.prefill_producer import produce as prefill
    from benchmarks.qwen35.compact_producer import produce as compact
    from benchmarks.qwen35.hopper_producer import produce as hopper
    from benchmarks.qwen35.mtp_prefill_producer import produce as mtp
    from benchmarks.qwen35.attention_workspace_producer import produce as workspace
    os.chdir('/workspace');volume.reload()
    if chunk not in (1024,2048):raise ValueError('bounded wider prefill chunk required')
    sources=[Path(__file__),*Path('packages/tensor-llm/src/tensor_llm/qwen35').rglob('*.py'),
             *Path('benchmarks/qwen35').glob('*producer.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    identity=hashlib.sha256(json.dumps(dict(prepared=prepared,chunk=chunk,sources=hashes),sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/prefill-chunks')/identity;root.mkdir(parents=True,exist_ok=True)
    checkpoint=prepared['base']['checkpoint']
    original=prefill(checkpoint,root/'original',slots=8,chunk=chunk,context=48000,
                     kv_dtype='fp8',block_m=64,packed_kv=True,target='sm_90')
    selected=compact(original,root/'compact')
    wide=hopper(selected,root/'hopper',expert_mma='bf16-pairs',hopper_attention=True)
    paths=dict(prepared['base']['paths'])
    paths['draft_prefill']=str(mtp(checkpoint,original,root/'draft-prefill'))
    result=dict(prepared,base=dict(prepared['base'],paths=paths),prefill=str(selected),
                hopper=str(wide),source_identity=identity,prefill_chunk=chunk,
                prefill_chunk_source_hashes=hashes,frozen_control_chunk=512)
    result['attention_workspace']=str(workspace(result,root/'attention-workspace'))
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def main(prepared_file:str,chunk:int=1024,prepare_only:bool=False,
         out:str='build/qwen35-h200-prefill-chunk'):
    from datetime import datetime,timezone
    import json
    from pathlib import Path
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare.remote(json.loads(Path(prepared_file).read_text()),chunk)
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    if prepare_only:return
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-chunk'+str(chunk)+'-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
