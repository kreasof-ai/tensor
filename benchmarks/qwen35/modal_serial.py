"""Locate verification/serial divergence on identical real 32K prefixes."""
from benchmarks.qwen35.modal_hopper_long import app,image,volume,prepare_expert_tiles,prepare_query_tiles


@app.function(image=image,gpu='H200',cpu=16,memory=98304,timeout=600,
              volumes={'/cache':volume},scaledown_window=2)
def diagnose(prepared,steps=2,capture_layers=True):
    import json,os,re
    from pathlib import Path
    from types import SimpleNamespace
    import numpy as np
    import tensor
    from tensor_llm import Qwen35Batch,Qwen35Prefill,Qwen35Verifier
    from tensor_llm.qwen35.compact_prefill import install
    from tensor_llm.qwen35.hopper import install as hopper
    from benchmarks.qwen35.whole_tune import Snapshot
    from benchmarks.qwen35.spec_run import install_selected
    os.chdir('/workspace');volume.reload()
    if not 1<=steps<=prepared['verification_chunk']:
        raise ValueError('serial steps must fit the verification window')
    base=prepared['base'];paths=base['paths']
    prompts=np.asarray([r['prompt_token_ids'] for r in json.loads(Path('workload.json').read_text())['requests']],'int32')
    with tensor.Device() as device,Qwen35Batch(base['checkpoint'],paths['decoder'],device,
            progress=lambda s:print(s,flush=True)) as model:
        prefix=Qwen35Prefill(model,prepared['prefill']);install(prefix,prepared['prefill'])
        hopper(prefix,prepared['hopper']);model.reset()
        try:
            for start in range(0,32000,prefix.chunk):
                count=min(prefix.chunk,32000-start)
                block=np.zeros((8,prefix.chunk),'int32');block[:,:count]=prompts[:,start:start+count]
                pending=prefix.forward(block,np.full(8,count,'int32'))
        finally:prefix.close()
        snapshot=Snapshot(model);inputs=[];references=[];logits=[];greedy=[]
        for step in range(steps):
            inputs.append(pending.copy());captured={}
            pending,output=model.forward(pending,read_logits=True,
                debug=(lambda label,value:captured.__setitem__(label,value.copy())) if capture_layers else None)
            references.append(captured);logits.append(output);greedy.append(pending.copy())
        snapshot.restore(model)
        verifier=Qwen35Verifier(model,paths['verify']);install_selected(verifier,paths['verify'])
        graph=verifier.graph;actual={};layer=0
        weights={id(buffer):name for name,buffer in model.weights.items()}
        def submit():
            nonlocal layer
            for kernel,bound in verifier.plan:
                device._launch(kernel,bound)
                values=dict(zip((a['name'] for a in kernel.manifest['abi']),bound.storage))
                name=weights.get(id(values.get('w')))
                label=None
                if name:
                    match=re.fullmatch(r'model.language_model.layers.(\d+).(input_layernorm|post_attention_layernorm).weight',name)
                    if match:
                        layer=int(match[1]);label=(layer,'input' if match[2]=='input_layernorm' else 'post_attention')
                    elif name=='model.language_model.norm.weight':label=(40,'final')
                if values.get('out') is verifier.buffers['ffn']:label=(layer,'mlp')
                if label is not None:
                    buffer=verifier.buffers['ffn' if label[1]=='mlp' else 'normal']
                    actual[label]=buffer.to_numpy().reshape(8,verifier.chunk,-1)[:,:steps].copy()
        if capture_layers:verifier.graph=SimpleNamespace(launch=submit)
        try:
            tokens=np.zeros((8,verifier.chunk),'int32');tokens[:,:steps]=np.stack(inputs,axis=1)
            predictions,output=verifier.forward(tokens,np.full(8,steps,'int32'),read_logits=True)
        finally:verifier.graph=graph
        try:
            rows=[]
            for label in references[0]:
                expected=np.stack([r[label] for r in references],axis=1)
                value=actual[label]
                row=dict(layer=label[0],phase=label[1],relative_rms=[float(
                    np.linalg.norm(value[:,i]-expected[:,i])/max(np.linalg.norm(expected[:,i]),1e-20)) for i in range(steps)])
                rows.append(row)
                if max(row['relative_rms'])>.001:print('Serial divergence',row,flush=True)
            expected=np.stack(logits,axis=1);value=output.reshape(8,verifier.chunk,-1)[:,:steps]
            result=dict(device=device.info,prepared=prepared,prefix_positions=snapshot.position.tolist(),
                scope=('uncaptured per-layer' if capture_layers else 'captured graph')+' teacher-forced verification vs native serial',
                layers=rows,steps=steps,
                relative_logit_rms=float(np.linalg.norm(value-expected)/np.linalg.norm(expected)),
                greedy_matches=int((predictions[:,:steps]==np.stack(greedy,axis=1)).sum()),greedy_total=8*steps,
                model_throughput_qualified=False)
            result['passed']=bool(np.isfinite(value).all() and result['relative_logit_rms']<=.03
                and result['greedy_matches']==result['greedy_total'])
            print('Serial final',result['relative_logit_rms'],result['greedy_matches'],flush=True)
            return result
        finally:verifier.close()


@app.local_entrypoint()
def serial_main(prepared_file:str,out:str='build/qwen35-h200-serial-diagnosis.json',paired_experts:bool=False,key_rows:int=32,
                steps:int=2,capture_layers:bool=True,query_tokens:int=8,kernel_target:str='sm_90a'):
    import json
    from pathlib import Path
    prepared=json.loads(Path(prepared_file).read_text())
    if paired_experts or key_rows!=32 or query_tokens!=8 or kernel_target!='sm_90a':
        if paired_experts:prepared=prepare_expert_tiles.remote(prepared,64,128,False,0,True)
        prepared=prepare_query_tiles.remote(prepared,query_tokens,key_rows,kernel_target)
    result=diagnose.remote(prepared,steps,capture_layers)
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n')
