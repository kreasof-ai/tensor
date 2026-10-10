"""Build bounded long-window verification on H200 without changing C8 load."""
from benchmarks.qwen35.modal_hopper import app,image,volume,prepare_hopper
from benchmarks.qwen35.modal_compact import measure_compact


@app.function(image=image,cpu=16,memory=98304,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_dense_tiles(prepared,asynchronous=False):
    import hashlib,json,os
    from pathlib import Path
    from benchmarks.qwen35.hopper_verify_producer import produce_dense
    os.chdir('/workspace');volume.reload()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_dense.py'),
        Path('benchmarks/qwen35/hopper_verify_producer.py'))}
    identity=hashlib.sha256(json.dumps(dict(hashes=hashes,prepared=prepared,
        asynchronous=asynchronous),sort_keys=True).encode()).hexdigest()[:16]
    paths=prepared['base']['paths'].copy();root=Path('/cache/hopper-dense')/identity
    paths['verify']=str(produce_dense(paths['verify'],root,asynchronous=asynchronous))
    result=dict(prepared,base=dict(prepared['base'],paths=paths),
        dense_tile_source_hashes=hashes,source_identity=identity)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def dense_main(prepared_file:str,asynchronous:bool=False,out:str='build/qwen35-h200-verify-dense'):
    from datetime import datetime,timezone
    import json
    from pathlib import Path
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare_dense_tiles.remote(json.loads(Path(prepared_file).read_text()),asynchronous)
    ready.update(profile_phases=False,profile_only=False)
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-dense-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


@app.function(image=image,cpu=16,memory=98304,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_expert_tiles(prepared,block_m,columns,warp_mma,persistent_tiles=0,paired=False):
    import hashlib,json,os
    from pathlib import Path
    from benchmarks.qwen35.hopper_verify_producer import produce
    os.chdir('/workspace');volume.reload()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_experts.py'),
        Path('benchmarks/qwen35/hopper_verify_producer.py'))}
    identity=hashlib.sha256(json.dumps(dict(hashes=hashes,prepared=prepared,
        block_m=block_m,columns=columns,warp_mma=warp_mma,persistent_tiles=persistent_tiles,paired=paired),sort_keys=True).encode()).hexdigest()[:16]
    paths=prepared['base']['paths'].copy();root=Path('/cache/hopper-verify')/identity
    paths['verify']=str(produce(paths['verify'],root,block_m=block_m,
                               columns=columns,warp_mma=warp_mma,persistent_tiles=persistent_tiles,paired=paired))
    result=dict(prepared,base=dict(prepared['base'],paths=paths),
        expert_tile_source_hashes=hashes,source_identity=identity)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def expert_main(prepared_file:str,block_m:int=16,columns:int=64,warp_mma:bool=True,
                out:str='build/qwen35-h200-verify-tiles',persistent_tiles:int=0):
    from datetime import datetime,timezone
    import json
    from pathlib import Path
    if block_m not in (16,32,64) or columns not in (64,128):raise ValueError('unsupported expert tile')
    if not warp_mma and block_m!=64:raise ValueError('WGMMA requires 64 rows')
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    if persistent_tiles not in (0,64,128):raise ValueError('unsupported persistent grid')
    ready=prepare_expert_tiles.remote(json.loads(Path(prepared_file).read_text()),block_m,columns,warp_mma,persistent_tiles)
    ready.update(profile_phases=False,profile_only=False)
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-expert-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


@app.function(image=image,cpu=16,memory=98304,timeout=900,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_query_tiles(prepared,query_tokens,key_rows=32,kernel_target='sm_90a'):
    import hashlib,json,os
    from pathlib import Path
    from benchmarks.qwen35.spec_attention_producer import produce
    os.chdir('/workspace');volume.reload()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_attention.py'),
        Path('benchmarks/qwen35/spec_attention_producer.py'))}
    identity=hashlib.sha256(json.dumps(dict(hashes=hashes,prepared=prepared,
        query_tokens=query_tokens,key_rows=key_rows,kernel_target=kernel_target),sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/hopper-query')/identity
    paths=prepared['base']['paths'].copy()
    for name in ('verify','repair'):
        paths[name]=str(produce(paths[name],root/name,key_rows=key_rows,
            query_tokens=query_tokens,kernel_target=kernel_target))
    result=dict(prepared,base=dict(prepared['base'],paths=paths),
                query_tokens=query_tokens,key_rows=key_rows,attention_target=kernel_target,
                query_source_hashes=hashes,source_identity=identity)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def query_main(prepared_file:str,query_tokens:int=16,out:str='build/qwen35-h200-query'):
    from datetime import datetime,timezone
    import json
    from pathlib import Path
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    ready=prepare_query_tiles.remote(json.loads(Path(prepared_file).read_text()),query_tokens)
    ready.update(profile_phases=False,profile_only=False)
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-q'+str(query_tokens)+'-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')


@app.function(image=image,cpu=16,memory=98304,timeout=1800,
              volumes={'/cache':volume},scaledown_window=2)
def prepare_long(prepared,chunk,prefix_mma='mma-experts'):
    import hashlib,json,os
    from pathlib import Path
    from benchmarks.qwen35.prefill_producer import produce as prefill
    from benchmarks.qwen35.spec_producer import produce as verify
    from benchmarks.qwen35.spec_attention_producer import produce as attention
    from benchmarks.qwen35.spec_linear_producer import produce as linear
    from benchmarks.qwen35.hopper_verify_producer import produce as compact_verify
    from benchmarks.qwen35.mtp_prefill_producer import produce as repair
    os.chdir('/workspace');volume.reload()
    ready=prepare_hopper.local(prepared,'bf16-pairs',True) if prefix_mma=='bf16-pairs-attention' else prepare_hopper.local(prepared,prefix_mma)
    paths=ready['base']['paths'].copy();checkpoint=ready['base']['checkpoint']
    sources=[*Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels').glob('hopper_*.py'),
             Path('benchmarks/qwen35/hopper_verify_producer.py'),Path(__file__)]
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    identity=hashlib.sha256(json.dumps(dict(hashes=hashes,chunk=chunk,
        base_identity=ready['base']['source_identity'],prefix_identity=ready['source_identity']),sort_keys=True).encode()).hexdigest()[:16]
    root=Path('/cache/hopper-long')/identity;root.mkdir(parents=True,exist_ok=True)
    small=prefill(checkpoint,root/'prefill',chunk=chunk,block_m=64,
                  kv_dtype='fp8',packed_kv=True,target='sm_90')
    verified=verify(small,root/'verify')
    split=attention(verified,root/'attention',key_rows=32,query_tokens=8,kernel_target='sm_90a')
    selected=linear(split,root/'linear',expert_block_m=32)
    paths['verify']=str(compact_verify(selected,root/'compact'))
    repaired=repair(checkpoint,small,root/'repair')
    paths['repair']=str(attention(repaired,root/'repair-attention',key_rows=32,
                                  query_tokens=8,kernel_target='sm_90a'))
    result=dict(ready,base=dict(ready['base'],paths=paths),verification_chunk=chunk,
                source_identity=identity,long_source_hashes=hashes)
    (root/'prepared.json').write_text(json.dumps(result,indent=2)+'\n');volume.commit()
    return result


@app.local_entrypoint()
def long_main(prepared_file:str,chunk:int=32,out:str='build/qwen35-h200-long',prepared_is_ready:bool=False,prefix_mma:str='mma-experts'):
    from datetime import datetime,timezone
    import json
    from pathlib import Path
    if chunk not in (16,32,64,128):raise ValueError('bounded long window must be 16, 32, 64 or 128')
    root=Path(out);root.mkdir(parents=True,exist_ok=True)
    value=json.loads(Path(prepared_file).read_text())
    ready=value if prepared_is_ready else prepare_long.remote(value,chunk,prefix_mma)
    (root/'prepared.json').write_text(json.dumps(ready,indent=2)+'\n')
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(volume='tensor-qwen35-h200',run_id=run_id),indent=2)+'\n')
    result=measure_compact.remote(ready,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
