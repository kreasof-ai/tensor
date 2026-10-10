"""Check packed FP8 operands before timing the bounded projection candidates."""
import modal
from benchmarks.qwen35.modal_h200 import image,volume
from benchmarks.qwen35.modal_expert_search import search

app=modal.App('tensor-qwen35-h200-projection-bandwidth')


@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def measure(prepared):
    import hashlib,os,subprocess,sys
    from pathlib import Path
    os.chdir('/workspace');volume.reload()
    command=[sys.executable,'-m','pytest','packages/tensor-llm/tests/test_qwen_hopper.py',
             '-k','packed','-q','-o','addopts=']
    test=subprocess.run(command,env=dict(os.environ,TENSOR_QWEN_CUDA='1'),text=True,
                        capture_output=True,timeout=300)
    print(test.stdout,test.stderr,flush=True)
    if test.returncode:raise RuntimeError('packed projection checks failed')
    for case in prepared['cases']:
        for row in [dict(path=case['control'],sha256=case['control_sha256']),*case['candidates']]:
            if hashlib.sha256(Path(row['path']).read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('projection micro artifact checksum mismatch')
    Path(prepared['root']).mkdir(parents=True,exist_ok=True)
    result=search.local(prepared,normal_inputs=True)
    result['kernel_checks']=dict(command=command,exit_code=test.returncode,log=test.stdout+test.stderr)
    return result


@app.local_entrypoint()
def main(prepared_file:str,out:str='build/qwen35-h200-projection-bandwidth.json'):
    import json
    from pathlib import Path
    result=measure.remote(json.loads(Path(prepared_file).read_text()))
    Path(out).write_text(json.dumps(result,indent=2)+'\n')
