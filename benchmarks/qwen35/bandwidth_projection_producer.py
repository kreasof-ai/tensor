"""Reduce projection CTA registers without changing FP8 values or reductions."""
import hashlib,json,shutil
from pathlib import Path
from tensor.compiler.entry import export_source
from .build import build_artifact


def prefill(prepared,out,columns=64,packed_widen=False):
    source=Path(prepared['hopper']);control=Path(prepared['prefill']);out=Path(out)
    manifest=json.loads((source/'hopper.json').read_text())
    original=json.loads((control/'prefill.json').read_text())
    if columns not in (64,128):raise ValueError('unsupported projection width')
    if out.resolve()==source.resolve():raise ValueError('use a separate projection bundle')
    out.mkdir(parents=True,exist_ok=True)
    for name,row in manifest['kernels'].items():
        path=(source/row['path']).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('source Hopper checksum mismatch')
        destination=out/row['path']
        if row['kind'] not in ('compact_fp8_experts','fp8_linear'):
            shutil.copy2(path,destination);continue
        p=original['kernels'][name.removeprefix('_compact_')]['parameters']
        schedule=dict(block_m=64,columns=columns,threads=128 if columns==64 else 256,
            packed_gather=True,stages=1,mma_reduction=32,mma_reorder=True,async_mma=False,
            packed_widen=packed_widen)
        if row['kind']=='compact_fp8_experts':
            module,factory='hopper_experts','expert_kernel'
            schedule.update(rows=original['slots']*original['chunk'],k=p['k'],o=p['o'],
                routed_input=p['routed_input'],compact=True,bf16_mma=True)
        else:
            module,factory='hopper_dense','make_kernel';schedule.update(p)
        entry=destination.with_suffix('.py')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.'+module,factory,schedule,
                                      dependencies=('tensor.compiler.entry','tensor_llm.qwen35.kernels.fp8_operand')))
        destination.unlink(missing_ok=True);build_artifact(entry,destination,target='sm_90a')
        row['sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
        print('Prefill projection',module,schedule,flush=True)
    manifest['projection_columns']=columns
    manifest['expert_columns']=columns
    manifest['packed_widen']=packed_widen
    manifest['projection_producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'hopper.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out


def dense_verification(source,out,columns=64,packed_widen=False):
    source,out=Path(source),Path(out)
    manifest=json.loads((source/'prefill.json').read_text())
    if columns not in (64,128):raise ValueError('unsupported verification width')
    if out.resolve()==source.resolve():raise ValueError('use a separate verification bundle')
    out.mkdir(parents=True,exist_ok=True)
    for name,row in manifest['kernels'].items():
        path=(source/row['path']).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('source verification checksum mismatch')
        destination=out/row['path']
        if row['kind']!='split_dense':
            shutil.copy2(path,destination);continue
        schedule=dict(row['parameters'],block_m=64,columns=columns,threads=128 if columns==64 else 256,
            packed_gather=True,stages=1,mma_reduction=32,mma_reorder=True,async_mma=False,
            packed_widen=packed_widen)
        entry=destination.with_suffix('.py')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_dense','make_kernel',schedule,
                                      dependencies=('tensor.compiler.entry','tensor_llm.qwen35.kernels.fp8_operand')))
        destination.unlink(missing_ok=True);build_artifact(entry,destination,target='sm_90a')
        row['sha256']=hashlib.sha256(destination.read_bytes()).hexdigest()
        print('Verification dense',schedule,flush=True)
    manifest['bandwidth_dense_columns']=columns
    manifest['bandwidth_dense_packed_widen']=packed_widen
    manifest['bandwidth_producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
