"""Add routed and dense small-chunk FP8 Split-K projection candidates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .build import build_artifact, needs_build
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.artifacts import identity
from tensor_llm.qwen35.speculative.linear import implementation_hashes


def produce(prefill,out,*,expert_parts=4,dense_parts=8,expert_columns=128,grouped_rows=False,expert_block_m=16):
    prefill,out=Path(prefill).resolve(),Path(out).resolve()
    if out==prefill or out.is_relative_to(prefill):raise ValueError('use a separate output')
    manifest=json.loads((prefill/'prefill.json').read_text());out.mkdir(parents=True,exist_ok=True)
    logicals=list(manifest['kernels'].items());records={}
    for _,row in logicals:
        source=(prefill/row['path']).resolve()
        if not source.is_relative_to(prefill) or hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('source artifact checksum mismatch')
        shutil.copyfile(source,out/row['path'])
    def add(kind,p,module,factory,*args):
        key=identity(kind,p);entry=out/(key+'.py');artifact=entry.with_suffix('.tbin')
        source=export_source(module,factory,*args,dependencies=('tensor.compiler.entry',))
        if needs_build(entry, artifact, source, manifest.get('target','sm_89')):
            entry.write_text(source);artifact.unlink(missing_ok=True)
            build_artifact(entry,artifact,target=manifest.get('target','sm_89'))
        manifest['kernels'][key]=dict(kind=kind,parameters=p,path=artifact.name,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        print(kind,p,flush=True);return key
    for logical,row in logicals:
        kind,p=row['kind'],row['parameters']
        if kind=='bf16_linear' and p['o'] in (1,32,256):
            records[logical]=dict(matmul=add('small_bf16',p,'tensor_llm.qwen35.kernels.matmul','bf16_decode_kernel',p))
            continue
        if kind not in ('fp8_experts','fp8_linear'):continue
        grouped=kind=='fp8_experts';parts=min(expert_parts if grouped else dense_parts,p['k']//128)
        selected=dict(p,partitions=parts)
        if grouped and grouped_rows:
            if p['r']>16:raise ValueError('fixed row groups support up to sixteen verification rows')
            matmul=add('split_experts',selected,'tensor_llm.qwen35.kernels.matmul','make_kernel','fp8_experts_mma_prequantized',
                dict(p,columns=expert_columns,block_m=16,partitions=parts,stages=2,packed_copy=True))
        elif grouped:
            matmul=add('split_experts',selected,'tensor_llm.qwen35.kernels.speculative_linear','expert_kernel',
                dict(rows=p['r'],k=p['k'],o=p['o'],routed_input=p['routed_input'],
                     columns=expert_columns,block_m=expert_block_m,threads=256 if expert_block_m>=32 else 128,partitions=parts,stages=2))
        else:
            matmul=add('split_dense',selected,'tensor_llm.qwen35.kernels.matmul','make_kernel','fp8_linear_mma_prequantized',
                dict(p,columns=64,block_m=16 if p['r']<=16 else (64 if p['r']>=64 else 32),
                     threads=256 if p['r']>=64 else 128,packed_copy=True,partitions=parts,stages=2))
        merge_p=dict(r=p['r'],o=p['o'],partitions=parts)
        if grouped:merge_p['top']=8
        merge=add('split_linear_merge',merge_p,'tensor_llm.qwen35.kernels.matmul','merge_kernel',merge_p)
        records[logical]=dict(matmul=matmul,merge=merge,
                             partial_shape=[p['r'],*([8] if grouped else []),parts,p['o']])
        if grouped and grouped_rows:records[logical]['fixed_rows']=True
    if grouped_rows:
        p=dict(r=manifest['slots']*manifest['chunk'],top=8)
        key=add('fixed_routes',p,'tensor_llm.qwen35.kernels.decode','make_kernel','moe_groups',p)
        manifest['fixed_routes']=key
    manifest.update(split_linear=records,split_linear_implementation=implementation_hashes(),
        split_linear_schedule=dict(expert_parts=expert_parts,dense_parts=dense_parts,expert_columns=expert_columns,grouped_rows=grouped_rows,expert_block_m=expert_block_m))
    (out/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n');return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--expert-parts',type=int,default=4);p.add_argument('--dense-parts',type=int,default=8)
    p.add_argument('--expert-columns',type=int,default=128)
    p.add_argument('--grouped-rows',action='store_true')
    p.add_argument('--expert-block-m',type=int,choices=(16,32,64),default=16)
    a=p.parse_args();produce(a.prefill,a.out,expert_parts=a.expert_parts,dense_parts=a.dense_parts,expert_columns=a.expert_columns,grouped_rows=a.grouped_rows,expert_block_m=a.expert_block_m)
