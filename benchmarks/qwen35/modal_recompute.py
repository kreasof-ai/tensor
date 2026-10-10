"""Physical numerical checks for accepted-prefix recurrent recomputation."""
import modal
from benchmarks.qwen35.modal_h200 import image,volume
from benchmarks.qwen35.modal_compact import app as replay_app,measure_compact

app=modal.App('tensor-qwen35-h200-recompute')
app.include(replay_app)


@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def qualify():
    import hashlib,os,subprocess,sys
    from pathlib import Path
    os.chdir('/workspace');volume.reload()
    command=[sys.executable,'-m','pytest','packages/tensor-llm/tests/test_qwen_recompute.py',
             '-q','-o','addopts=']
    test=subprocess.run(command,env=dict(os.environ,TENSOR_QWEN_CUDA='1'),text=True,
                        capture_output=True,timeout=450)
    print(test.stdout,test.stderr,flush=True)
    return dict(exit_code=test.returncode,command=command,output=test.stdout+test.stderr,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/recompute.py'),
            Path('packages/tensor-llm/tests/test_qwen_recompute.py'))})


@app.local_entrypoint()
def tests_main(out:str='build/qwen35-h200-recompute-tests.json'):
    import json
    from pathlib import Path
    result=qualify.remote();Path(out).write_text(json.dumps(result,indent=2)+'\n')
    if result['exit_code']:raise RuntimeError('accepted-prefix recomputation checks failed')


@app.function(image=image,cpu=16,memory=32768,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def prepare(prepared):
    import hashlib,json,os
    from pathlib import Path
    from benchmarks.qwen35.recompute_producer import produce
    os.chdir('/workspace');volume.reload()
    sources=[Path(__file__),Path('benchmarks/qwen35/recompute_producer.py'),
             *Path('packages/tensor-llm/src/tensor_llm/qwen35').rglob('*.py')]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    identity=hashlib.sha256(json.dumps(dict(prepared=prepared,sources=hashes),sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/recompute')/identity;root.mkdir(parents=True,exist_ok=True)
    paths=dict(prepared['base']['paths']);control=paths['verify']
    if prepared.get('verification_chunk')==128:
        from benchmarks.qwen35.hopper_verify_producer import produce_dense
        from benchmarks.qwen35.attention_workspace_producer import produce_verification
        control=produce_dense(control,root/'control')
        produce_verification(control,control/'attention-workspace')
        manifest=json.loads((control/'prefill.json').read_text())
        manifest['attention_workspace']='attention-workspace'
        (control/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n')
        control=str(control)
    paths['verify']=str(produce(control,root/'verify'))
    result=dict(prepared,base=dict(prepared['base'],paths=paths),source_identity=identity,
                state_recompute=True,recompute_control_bundle=control,recompute_source_hashes=hashes)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def main(prepared_file:str,out:str='build/qwen35-h200-recompute',prepare_only:bool=False):
    from datetime import datetime,timezone
    import json
    from pathlib import Path
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare.remote(json.loads(Path(prepared_file).read_text()))
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    if prepare_only:return
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-recompute-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
