"""Compile chunked prefill artifacts without a consumer inference framework."""
from pathlib import Path
import hashlib,json
from .build import build_artifact, needs_build
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.artifacts import identity, requirements
from tensor_llm.qwen35.checkpoint import Qwen35Checkpoint


def produce(checkpoint,out,*,slots=8,chunk=128,context=48000,kv_dtype='bfloat16',block_m=16,packed_kv=False,target='sm_89'):
    if kv_dtype not in ('bfloat16','fp8') or (packed_kv and kv_dtype!='fp8'):
        raise ValueError('packed FP8 KV requires the FP8 cache profile')
    c=Qwen35Checkpoint(checkpoint);rows=slots*chunk
    out=Path(out);out.mkdir(parents=True,exist_ok=True);records={}
    def add(kind,p,module,factory,*args):
        key=identity(kind,p);entry=out/(key+'.py');artifact=entry.with_suffix('.tbin')
        text=export_source(module,factory,*args,dependencies=('tensor.compiler.entry','tensor.compiler.cuda_lowering','tensor_llm.qwen35.kernels.fp8_kv'))
        if needs_build(entry, artifact, text, target):
            artifact.unlink(missing_ok=True);entry.write_text(text)
            build_artifact(entry,artifact,target=target)
        records[key]=dict(kind=kind,parameters=p,path=artifact.name,sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        print('prefill',kind,p,flush=True)
    for kind in ('controls','gdn_conv','gdn_scan','expert_routes','advance'):
        p=dict(slots=slots,chunk=chunk)
        add(kind,p,'tensor_llm.qwen35.kernels.prefill','make_kernel',kind,p)
    p=dict(slots=slots,chunk=chunk,width=2048)
    add('last_rows',p,'tensor_llm.qwen35.kernels.prefill','make_kernel','last_rows',p)
    for kind in ('attention','attention_kv'):
        p=dict(slots=slots,chunk=chunk,capacity=context)
        if kv_dtype=='fp8':p['kv_dtype']='fp8'
        if kind=='attention_kv':p.update(eps=c.config.epsilon,theta=c.config.rope_theta)
        schedule = dict(p,packed_loads=True,query_rows=128,key_rows=16) if kind=='attention' and packed_kv else p
        add(kind,p,'tensor_llm.qwen35.kernels.prefill','make_kernel',kind,schedule)
    needed=requirements(c.config,rows,context)
    skipped={'gdn_conv','gdn_recurrent','attention_kv','attention_partial','attention_merge','argmax','moe_groups'}
    for kind,p in needed.values():
        if kind in skipped or (kind=='bf16_linear' and p['o']==c.config.vocab):continue
        if kind=='fp8_linear':
            add(kind,p,'tensor_llm.qwen35.kernels.matmul','make_kernel','fp8_linear_mma_prequantized',
                dict(p,columns=64,block_m=block_m,threads=256 if block_m>=32 else 128,packed_copy=True,stages=2))
        elif kind=='fp8_experts':
            add(kind,p,'tensor_llm.qwen35.kernels.prefill','expert_kernel',
                dict(rows=rows,k=p['k'],o=p['o'],routed_input=p['routed_input'],block_m=block_m,threads=256 if block_m>=32 else 128))
        elif kind=='bf16_linear':
            add(kind,p,'tensor_llm.qwen35.kernels.matmul','bf16_kernel',p)
        else:add(kind,p,'tensor_llm.qwen35.kernels.decode','make_kernel',kind,p)
    for k,top in ((2048,None),(4096,None),(512,None),(512,8)):
        p=dict(r=rows,k=k)
        if top:p['top']=top
        add('quantize',p,'tensor_llm.qwen35.kernels.matmul','quantize_kernel',p)
    manifest=dict(schema='tensor.qwen35-prefill.v1',slots=slots,chunk=chunk,context=context,kv_dtype=kv_dtype,target=target,
        schedule=dict(block_m=block_m,packed_kv=packed_kv),kernels=records)
    (out/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--chunk',type=int,default=128)
    p.add_argument('--kv-dtype',choices=('bfloat16','fp8'),default='bfloat16')
    p.add_argument('--block-m',type=int,choices=(16,32,64,128),default=16)
    p.add_argument('--packed-kv',action='store_true')
    p.add_argument('--target',default='sm_89')
    a=p.parse_args();produce(a.checkpoint,a.out,chunk=a.chunk,kv_dtype=a.kv_dtype,block_m=a.block_m,packed_kv=a.packed_kv,target=a.target)
