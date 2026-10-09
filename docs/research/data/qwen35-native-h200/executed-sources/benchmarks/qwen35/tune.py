"""Producer-side kernel replacement with resident native checkpoint resources.

Replacement requires the same public buffer ABI and binds every existing
buffer again. Graph capture and kernel module ownership change together; model
weights and request states remain owned by the original batch executor.
"""
import hashlib
import json
from pathlib import Path

from .build import build_artifact
from tensor.compiler.entry import export_source


def mma_bundle(source_bundle,out,*,columns=64,threads=128,partitions=1,prequantized=False,stages=1,packed_copy=False):
    source_bundle,out=Path(source_bundle),Path(out)
    out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((source_bundle/'inference.json').read_text())
    quantizers={}
    for key,row in manifest['kernels'].items():
        kind,p=row['kind'],row['parameters']
        target=out/(key+'.tbin')
        if kind in ('fp8_linear','fp8_experts'):
            logical=out/(key+'-logical.tbin')
            old=row.get('logical',row)
            logical.write_bytes((source_bundle/old['path']).read_bytes())
            row['logical']=dict(path=logical.name,sha256=hashlib.sha256(logical.read_bytes()).hexdigest())
            actual_parts=min(partitions,p['k']//128)
            newp=dict(p,columns=columns,threads=threads,partitions=actual_parts,stages=stages,packed_copy=packed_copy)
            entry=out/(key+'.py')
            entry.write_text(export_source('tensor_llm.qwen35.kernels.matmul','make_kernel',kind+('_mma_prequantized' if prequantized else '_mma'),newp,
                dependencies=('tensor.compiler.entry','tensor.compiler.cuda_lowering')))
            build_artifact(entry,target,target=manifest['target'])
            row['schedule']=dict(family='mma',columns=columns,threads=threads,partitions=actual_parts,stages=stages,packed_copy=packed_copy)
            if prequantized:
                qp=dict(r=p['r'],k=p['k'])
                if kind=='fp8_experts' and p.get('routed_input'):qp['top']=p['top']
                qkey=hashlib.sha256(json.dumps(qp,sort_keys=True).encode()).hexdigest()[:16]
                if qkey not in quantizers:
                    qe=out/('quantize-'+qkey+'.py');qa=qe.with_suffix('.tbin')
                    qe.write_text(export_source('tensor_llm.qwen35.kernels.matmul','quantize_kernel',qp,
                        dependencies=('tensor.compiler.entry','tensor.compiler.cuda_lowering')))
                    build_artifact(qe,qa,target=manifest['target'])
                    quantizers[qkey]=dict(path=qa.name,sha256=hashlib.sha256(qa.read_bytes()).hexdigest())
                row['quantize']=quantizers[qkey]
            if actual_parts>1:
                mergep=dict(r=p['r'],o=p['o'],partitions=actual_parts)
                if kind=='fp8_experts':mergep['top']=p['top']
                merge_entry=out/(key+'-merge.py');merge_artifact=merge_entry.with_suffix('.tbin')
                merge_entry.write_text(export_source('tensor_llm.qwen35.kernels.matmul','merge_kernel',mergep,
                    dependencies=('tensor.compiler.entry',)))
                build_artifact(merge_entry,merge_artifact,target=manifest['target'])
                row['merge']=dict(path=merge_artifact.name,sha256=hashlib.sha256(merge_artifact.read_bytes()).hexdigest())
        else:
            target.write_bytes((source_bundle/row['path']).read_bytes())
        row['path']=target.name
        row['sha256']=hashlib.sha256(target.read_bytes()).hexdigest()
    # This tuning bundle binds the original consumer and records the producer
    # separately. Shipping it requires the final producer/consumer manifest.
    manifest['tuning_producer']=dict(module='tensor_llm.qwen35.kernels.matmul',
        sha256=hashlib.sha256(Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/matmul.py').read_bytes()).hexdigest())
    from tensor_llm.qwen35.decode import implementation_hashes
    manifest['implementation']=implementation_hashes()
    (out/'inference.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out



from tensor_llm.qwen35.pipeline import replace


def selected_bundle(source_bundle,out,search_directory):
    """Freeze measured projection candidates in a replayable AOT bundle."""
    import shutil
    source_bundle,out,search_directory=map(Path,(source_bundle,out,search_directory))
    shutil.copytree(source_bundle,out)
    manifest=json.loads((out/'inference.json').read_text())
    search=json.loads((search_directory/'search.json').read_text())
    for key,choice in search['winners'].items():
        if choice['status']!='passed':raise ValueError('unqualified projection winner')
        row=manifest['kernels'][key]
        if not row.get('quantize'):raise ValueError('selected projection requires a prequantized bundle')
        for role in ('artifact','merge'):
            descriptor=choice.get(role)
            if descriptor is None:
                if role=='merge':row.pop('merge',None)
                continue
            source=search_directory/descriptor['path']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=descriptor['sha256']:raise ValueError('search artifact checksum mismatch')
            target=out/(key+('-merge' if role=='merge' else '')+'.tbin')
            target.write_bytes(source.read_bytes())
            new=dict(path=target.name,sha256=descriptor['sha256'])
            if role=='artifact':row.update(new)
            else:row['merge']=new
        row['schedule']=choice['schedule']
    manifest['search_report_sha256']=hashlib.sha256((search_directory/'search.json').read_bytes()).hexdigest()
    (out/'inference.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
