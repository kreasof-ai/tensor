"""Compile a separate, source-bound prefill attention workspace bundle."""
import hashlib,json
from pathlib import Path
from tensor.artifacts.format import read_artifact
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.attention_workspace import implementation_hashes
from .build import build_artifact


def _compile(out, original, old, module, schedule, **settings):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=dict(schema='tensor.qwen35-attention-workspace.v1',implementation=implementation_hashes(),
        slots=original['slots'],chunk=original['chunk'],capacity=original['context'],
        target='sm_90a',original_cubin_sha256=old['files']['kernel.cubin'],
        expected_attention_calls=10,authoritative_kv_dtype='fp8',workspace_dtype='bfloat16',
        **settings,kernels={})
    for name,producer,factory,p in (
        ('decode','attention_workspace','decode',dict(slots=original['slots'],capacity=original['context'])),
        ('attention',module,'partial' if module=='hopper_attention' else 'attention',schedule)):
        entry=out/(name+'.py');artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.'+producer,factory,p,
                                      dependencies=('tensor.compiler.entry',)))
        artifact.unlink(missing_ok=True);build_artifact(entry,artifact,target='sm_90a')
        result['kernels'][name]=dict(path=artifact.name,sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    (out/'attention-workspace.json').write_text(json.dumps(result,indent=2)+'\n')
    return out


def produce(prepared,out,query_rows=128,threads=256,value_splits=1):
    source=Path(prepared['hopper']);control=Path(prepared['prefill']);out=Path(out)
    manifest=json.loads((source/'hopper.json').read_text())
    original=json.loads((control/'prefill.json').read_text())
    attention=[(name,row) for name,row in manifest['kernels'].items() if row['kind']=='attention']
    if len(attention)!=1:raise ValueError('one prefill attention specialization required')
    name,row=attention[0];path=(source/row['path']).resolve()
    if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('source Hopper attention checksum mismatch')
    old,_=read_artifact(path)
    p=original['kernels'][name]['parameters']
    return _compile(out,original,old,'hopper_prefill_attention',dict(p,decoded_kv=True,
        query_rows=query_rows,threads=threads,value_splits=value_splits,packed_loads=True),
        query_rows=query_rows,threads=threads,value_splits=value_splits)


def produce_verification(source,out):
    source=Path(source)
    original=json.loads((source/'prefill.json').read_text())
    attention=[row for row in original['kernels'].values() if row['kind']=='split_attention']
    if len(attention)!=1:raise ValueError('one split attention specialization required')
    row=attention[0];path=(source/row['path']).resolve()
    if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise ValueError('source split attention checksum mismatch')
    old,_=read_artifact(path)
    selected=original['split_attention']
    if selected.get('kernel_target')!='sm_90a':raise ValueError('Hopper split attention required')
    schedule=dict(row['parameters'],key_rows=selected['key_rows'],
        query_tokens=selected['query_tokens'],pad_queries=selected.get('pad_queries',False),packed_loads=True,decoded_kv=True)
    return _compile(out,original,old,'hopper_attention',schedule,scope='verification',
        key_rows=selected['key_rows'],query_tokens=selected['query_tokens'])
