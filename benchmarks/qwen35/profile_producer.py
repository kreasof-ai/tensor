"""Produce the native C8 decoder profile selected by complete-state measurements.

The profile remains experimental: kernel correctness and model quality have
separate gates, and the 600 tok/s full replay target has not been achieved.
"""
import argparse,hashlib,json
from pathlib import Path
from tensor.compiler.entry import export_source
from tensor.compiler.build import build_artifact
from tensor_llm.qwen35.decode import implementation_hashes
from .producer import produce as produce_base
from .tune import mma_bundle


def produce(checkpoint,out,*,kv_dtype='bfloat16',context=48000,slots=8):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    base=produce_base(checkpoint,out/'base',slots=slots,context=context,kv_dtype=kv_dtype)
    decoder=mma_bundle(base,out/'decoder',columns=64,partitions=8,
        prequantized=True,stages=2,packed_copy=True)
    path=decoder/'inference.json';manifest=json.loads(path.read_text())
    def compile_row(key,row,module,factory,*args):
        entry=decoder/(key+'.py');artifact=decoder/row['path'];artifact.unlink()
        entry.write_text(export_source(module,factory,*args,
            dependencies=('tensor.compiler.entry','tensor.compiler.cuda_lowering')))
        build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
        row['sha256']=hashlib.sha256(artifact.read_bytes()).hexdigest()
    for key,row in manifest['kernels'].items():
        kind,p=row['kind'],row['parameters']
        if kind=='bf16_linear':
            if p['o'] in (1,32,256):
                compile_row(key,row,'tensor_llm.qwen35.kernels.matmul','bf16_decode_kernel',p)
            else:
                compile_row(key,row,'tensor_llm.qwen35.kernels.matmul','bf16_head_kernel',dict(p,columns=128,depth=128))
        elif kind=='attention_partial' and kv_dtype=='fp8':
            compile_row(key,row,'tensor_llm.qwen35.kernels.fp8_kv','make_kernel',kind,dict(p,packed_loads=True))
            row['schedule']=dict(family='packed_fp8_kv')
        elif kind=='fp8_experts':
            parts=min(4,p['k']//128)
            schedule=dict(columns=128,threads=128,partitions=parts,stages=2,packed_copy=True)
            compile_row(key,row,'tensor_llm.qwen35.kernels.matmul','make_kernel','fp8_experts_mma_prequantized',dict(p,**schedule))
            row['schedule']=dict(family='mma',**schedule)
            entry=decoder/(key+'-merge.py');artifact=entry.with_suffix('.tbin');artifact.unlink()
            entry.write_text(export_source('tensor_llm.qwen35.kernels.matmul','merge_kernel',dict(r=slots,top=8,o=p['o'],partitions=parts),
                dependencies=('tensor.compiler.entry',)))
            build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
            row['merge']=dict(path=artifact.name,sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    manifest['implementation']=implementation_hashes()
    manifest['profile_producer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest.update(model_throughput_qualified=False,full_stress_target_reached=False)
    path.write_text(json.dumps(manifest,indent=2)+'\n');return decoder


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kv-dtype',choices=('bfloat16','fp8'),default='bfloat16')
    a=p.parse_args();print(produce(a.checkpoint,a.out,kv_dtype=a.kv_dtype))
