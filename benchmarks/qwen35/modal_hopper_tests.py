"""Run Hopper projections, long-query causality and rollback oracle checks."""
import modal
from benchmarks.qwen35.modal_h200 import image,volume
app=modal.App('tensor-qwen35-h200-kernel-tests')


@app.function(image=image,gpu='H200',cpu=16,memory=98304,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def diagnose_projection(prepared):
    import ctypes as ct
    import json,os
    from pathlib import Path
    import numpy as np
    import tensor
    from tensor_llm import Qwen35Batch,Qwen35Prefill
    from tensor.runtime.abi import BoundCall
    os.chdir('/workspace');volume.reload()
    base=prepared['base'];bundle=Path(prepared['hopper'])
    h=json.loads((bundle/'hopper.json').read_text())
    workload=json.loads(Path('workload.json').read_text())
    result=[]
    with tensor.Device() as d,Qwen35Batch(base['checkpoint'],base['paths']['decoder'],d,
                                        progress=lambda s:print(s,flush=True)) as model:
        executor=Qwen35Prefill(model,base['paths']['prefill']);model.reset()
        try:
            tokens=np.asarray([r['prompt_token_ids'][:512] for r in workload['requests']],'int32')
            for name,data in [('tokens',tokens),('lengths',np.full(8,512,'int32'))]:
                d.driver.call('cuMemcpyHtoD_v2',executor.buffers[name].pointer,
                              ct.c_void_p(data.ctypes.data),data.nbytes)
            model.active=np.ones(8,'int32');model._write('active',model.active)
            reverse={id(k):key for key,k in executor.kernels.items()}
            checked=set()
            for kernel,bound in executor.plan:
                d._launch(kernel,bound)
                key=reverse.get(id(kernel))
                row=h['kernels'].get(key)
                if row is None or row['kind']!='fp8_linear' or key in checked:continue
                checked.add(key)
                selected=d.load(bundle/row['path'])
                values=dict(zip([a['name'] for a in kernel.manifest['abi']],bound.storage))
                expected=values['out'].to_numpy()
                output=d.empty(values['out'].shape,'float32');values['out']=output
                try:
                    args=tuple(values[a['name']] for a in selected.manifest['arguments'])
                    storage,symbols,launch=selected._bind(args,{},include_outputs=True)
                    d._launch(selected,BoundCall(d,selected.manifest,storage,symbols,launch,validated=True))
                    actual=output.to_numpy()
                    record=dict(key=key,shape=list(actual.shape),bitwise_equal=bool(np.array_equal(actual,expected)),
                        relative_rms=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
                        maximum_absolute_error=float(np.max(np.abs(actual-expected))),
                        changed_elements=int(np.count_nonzero(actual!=expected)))
                    result.append(record);print('Real dense comparison',record,flush=True)
                finally:output.release();selected.release()
                if len(result)>=3:break
        finally:executor.close()
    return result


@app.function(image=image,gpu='H200',cpu=16,memory=32768,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def qualify(selection=''):
    import os,subprocess,sys
    os.chdir('/workspace')
    command=[sys.executable,'-m','pytest',
        'packages/tensor-llm/tests/test_qwen_hopper.py',
        'packages/tensor-llm/tests/test_qwen_spec.py::test_recurrent_snapshots_restore_every_rejection_depth[32]',
        '-q','-o','addopts=']
    if selection:command.extend(['-k',selection])
    result=subprocess.run(command,env=dict(os.environ,TENSOR_QWEN_CUDA='1'),text=True,capture_output=True)
    print(result.stdout,result.stderr,flush=True);volume.commit()
    return dict(exit_code=result.returncode,output=result.stdout+result.stderr)


@app.local_entrypoint()
def main(out:str='build/qwen35-h200-hopper-tests.json',projection_prepared_file:str='',selection:str=''):
    import json
    from pathlib import Path
    result=diagnose_projection.remote(json.loads(Path(projection_prepared_file).read_text())) if projection_prepared_file else qualify.remote(selection)
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(result,indent=2)+'\n')
    if not projection_prepared_file and result['exit_code']:raise RuntimeError('Hopper numerical checks failed')
