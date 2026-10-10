"""Compile pointer-ABI Hopper matrix kernels from a pinned control bundle."""
import hashlib
import json
from pathlib import Path
from tensor.artifacts.format import read_artifact
from tensor_llm.qwen35.hopper import implementation_hashes
from .build import build_artifact


def produce(source,out,*,expert_mma='control',hopper_attention=False,expert_columns=128,attention_load_rows=32):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out:raise ValueError('use a separate Hopper bundle')
    original=json.loads((source/'prefill.json').read_text())
    if original.get('target')!='sm_90':raise ValueError('Hopper control must be sm_90')
    out.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema='tensor.qwen35-hopper.v1',target='sm_90a',
        slots=original['slots'],chunk=original['chunk'],context=original['context'],
        implementation=implementation_hashes(),expert_mma=expert_mma,
        hopper_attention=hopper_attention,expert_columns=expert_columns,
        attention_load_rows=attention_load_rows,kernels={})
    if expert_columns not in (64,128):raise ValueError('expert columns must be 64 or 128')
    if attention_load_rows!=32:raise ValueError('only the padded 32-row attention candidate is supported')
    kinds={'fp8_linear','compact_fp8_experts','bf16_linear','attention','split_dense','split_experts'}
    if expert_mma!='control':kinds={'compact_fp8_experts','fp8_linear'}
    if expert_mma=='bf16-experts':kinds={'compact_fp8_experts'}
    if expert_mma=='mma-experts':kinds={'compact_fp8_experts'};manifest['target']='sm_90'
    if hopper_attention:
        if manifest['target']!='sm_90a':raise ValueError('attention requires sm_90a')
        kinds.add('attention')
    for key,record in original['kernels'].items():
        if record['kind'] not in kinds:continue
        artifact=source/record['path'];entry=artifact.with_suffix('.py')
        if not entry.is_file():raise ValueError('Hopper source missing: '+str(entry))
        if hashlib.sha256(artifact.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('Hopper control checksum mismatch')
        old,_=read_artifact(artifact)
        new_entry=out/entry.name
        if expert_mma!='control' and record['kind']=='compact_fp8_experts':
            from tensor.compiler.entry import export_source
            p=record['parameters']
            schedule=dict(rows=original['slots']*original['chunk'],k=p['k'],o=p['o'],
                routed_input=p['routed_input'],block_m=original['schedule']['block_m'],
                threads=128 if expert_columns==64 and expert_mma!='mma-experts' else 256,
                columns=64 if expert_mma=='mma-experts' else expert_columns,
                compact=True,packed_gather=True,
                stages=2 if expert_mma=='bf16-pipeline' else 1,bf16_mma=expert_mma.startswith('bf16'),mma_reduction=32 if expert_mma in ('bf16-32','bf16-pairs','bf16-async','bf16-pipeline') else 128,
                mma_reorder=expert_mma in ('bf16-pairs','bf16-async','bf16-pipeline'),async_mma=expert_mma in ('bf16-async','bf16-pipeline'))
            new_entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_experts',
                'expert_kernel',schedule,dependencies=('tensor.compiler.entry',)))
        elif expert_mma in ('bf16','bf16-32','bf16-pairs','bf16-async','bf16-pipeline') and record['kind']=='fp8_linear':
            from tensor.compiler.entry import export_source
            new_entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_dense','make_kernel',
                dict(record['parameters'],block_m=64,columns=128,threads=256,
                     packed_gather=True,
                     stages=2 if expert_mma=='bf16-pipeline' else 1,
                     mma_reduction=32 if expert_mma in ('bf16-32','bf16-pairs','bf16-async','bf16-pipeline') else 128,
                mma_reorder=expert_mma in ('bf16-pairs','bf16-async','bf16-pipeline'),async_mma=expert_mma in ('bf16-async','bf16-pipeline')),
                dependencies=('tensor.compiler.entry',)))
        elif hopper_attention and record['kind']=='attention':
            from tensor.compiler.entry import export_source
            new_entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_prefill_attention',
                'attention',dict(record['parameters'],packed_loads=True),
                dependencies=('tensor.compiler.entry',)))
        else:new_entry.write_bytes(entry.read_bytes())
        destination=new_entry.with_suffix('.tbin');destination.unlink(missing_ok=True)
        build_artifact(new_entry,destination,target=manifest['target'])
        manifest['kernels'][('_compact_'+key if record['kind']=='compact_fp8_experts' else key)]=dict(
            path=destination.name,sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
            original_cubin_sha256=old['files']['kernel.cubin'],kind=record['kind'])
        print('Hopper',record['kind'],record['parameters'],flush=True)
    (out/'hopper.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
