"""Compact Hopper expert projections for selected Split-K verification."""
import hashlib,json,shutil
from pathlib import Path
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.artifacts import identity
from tensor_llm.qwen35.compact_prefill import implementation_hashes
from .build import build_artifact


def produce_dense(source,out,*,asynchronous=False):
    """Replace selected dense Split-K cubins without changing their partitions."""
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out:raise ValueError('use a separate dense verification bundle')
    out.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((source/'prefill.json').read_text())
    for record in manifest['kernels'].values():
        path=(source/record['path']).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('source artifact checksum mismatch')
        shutil.copy2(path,out/record['path'])
    count=0
    for key,record in manifest['kernels'].items():
        if record['kind']!='split_dense':continue
        schedule=dict(record['parameters'],block_m=64,columns=128,threads=256,
            stages=1,packed_gather=True,mma_reduction=32,mma_reorder=True,async_mma=asynchronous)
        entry=out/(key+'.py');artifact=out/record['path']
        entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_dense','make_kernel',schedule,
            dependencies=('tensor.compiler.entry',)))
        artifact.unlink();build_artifact(entry,artifact,target='sm_90a')
        record['sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest();count+=1
        print('Hopper verification dense',schedule,flush=True)
    if not count:raise ValueError('selected dense Split-K projections absent')
    manifest['hopper_dense_schedule']=dict(asynchronous=asynchronous,shared_mma_dtype='float16',
        reduction=32,reordered=True,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out


def produce(source,out,*,block_m=64,columns=128,warp_mma=False,persistent_tiles=0,paired=False):
    source,out=Path(source).resolve(),Path(out).resolve()
    if source==out:raise ValueError('use a separate compact verification bundle')
    out.mkdir(parents=True,exist_ok=True)
    for p in source.iterdir():
        if p.is_file():shutil.copy2(p,out/p.name)
    manifest=json.loads((source/'prefill.json').read_text());rows=manifest['slots']*manifest['chunk']
    record=dict(max_tiles=(rows*8+block_m-1)//block_m+255,block_m=block_m,projections={})
    def add(kind,p,module,factory,schedule,target):
        key=identity(kind,p);entry=out/(key+'.py');artifact=entry.with_suffix('.tbin')
        entry.write_text(export_source(module,factory,schedule,dependencies=('tensor.compiler.entry',)))
        artifact.unlink(missing_ok=True);build_artifact(entry,artifact,target=target)
        manifest['kernels'][key]=dict(kind=kind,parameters=p,path=artifact.name,
                                      sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        return key
    p=dict(rows=rows,block_m=block_m)
    record['tile_map']=add('expert_tile_map',p,'tensor_llm.qwen35.kernels.compact_experts',
                          'tile_map_kernel',p,'sm_90')
    for logical,row in list(manifest['kernels'].items()):
        if row['kind']!='split_experts':continue
        p=row['parameters']
        schedule=dict(rows=rows,k=p['k'],o=p['o'],routed_input=p['routed_input'],
            block_m=block_m,threads=128 if warp_mma or columns==64 else 256,columns=columns,partitions=p['partitions'],
            compact=True,packed_gather=True,bf16_mma=not warp_mma,stages=1,persistent_tiles=persistent_tiles)
        if paired:
            if warp_mma:raise ValueError('paired WGMMA conflicts with warp MMA')
            schedule.update(mma_reduction=32,mma_reorder=True)
        record['projections'][logical]=add('hopper_split_experts',p,
            'tensor_llm.qwen35.kernels.hopper_experts','expert_kernel',schedule,'sm_90' if warp_mma else 'sm_90a')
    if not record['projections']:raise ValueError('selected Split-K experts absent')
    manifest['compact_experts']=record;manifest['compact_expert_implementation']=implementation_hashes()
    manifest['hopper_expert_schedule']=dict(block_m=block_m,columns=columns,weight_storage='fp8',persistent_tiles=persistent_tiles,
        shared_mma_dtype='fp8' if warp_mma else ('float16' if paired else 'bfloat16'),paired=paired,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n');return out
