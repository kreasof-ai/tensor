"""Add split-context small-chunk attention to a selected verification/repair bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from tensor.compiler.build import build_artifact
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.artifacts import identity
from tensor_llm.qwen35.speculative.attention import implementation_hashes


def produce(prefill,out,*,splits=16,key_rows=64):
    prefill,out=Path(prefill).resolve(),Path(out).resolve()
    if out==prefill or out.is_relative_to(prefill):raise ValueError('use a separate output')
    manifest=json.loads((prefill/'prefill.json').read_text())
    if manifest.get('kv_dtype')!='fp8':raise ValueError('split kernel requires FP8 KV')
    out.mkdir(parents=True,exist_ok=True)
    for row in manifest['kernels'].values():
        source=(prefill/row['path']).resolve()
        if not source.is_relative_to(prefill) or hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('source artifact checksum mismatch')
        shutil.copyfile(source,out/row['path'])
    p={k:manifest[k] for k in ('slots','chunk','context')};p['capacity']=p.pop('context');p['splits']=splits
    for kind,factory in (('split_attention','partial'),('split_merge','merge')):
        key=identity(kind,p);entry=out/(key+'.py');artifact=entry.with_suffix('.tbin')
        schedule=dict(p,packed_loads=True,key_rows=key_rows)
        source=export_source('tensor_llm.qwen35.kernels.speculative_attention',factory,schedule,dependencies=('tensor.compiler.entry',))
        if not artifact.is_file() or not entry.is_file() or entry.read_text()!=source:
            entry.write_text(source);artifact.unlink(missing_ok=True)
            build_artifact(entry,artifact,target='sm_89',compiler='nvrtc',nvrtc_home='build/nvrtc-12.9')
        manifest['kernels'][key]=dict(kind=kind,parameters=p,path=artifact.name,
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        print(kind,schedule,flush=True)
    manifest.update(split_attention=dict(splits=splits,key_rows=key_rows),
                    split_attention_implementation=implementation_hashes())
    (out/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n');return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--splits',type=int,default=16);p.add_argument('--key-rows',type=int,default=64)
    a=p.parse_args();produce(a.prefill,a.out,splits=a.splits,key_rows=a.key_rows)
