"""Archive a matched quantized-model repeat and its runtime-only ablation."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from tensor.artifacts.format import read_artifact
from tensor_llm.provenance import implementation_hashes


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(root,reference,out):
    root=Path(root)
    comparison=json.loads((root/'comparison/report.json').read_text())
    runtime=json.loads((root/'runtime-ablation/report.json').read_text())
    if comparison['status']!='passed' or runtime['status']!='passed':
        raise ValueError('requires two passed measurements')
    if comparison['model_sha256']!=runtime['model_sha256']:
        raise ValueError('measurements must use the same checkpoint')
    if runtime['bundles']['before']!=runtime['bundles']['searched']:
        raise ValueError('runtime ablation requires identical kernel bundles')
    shaders={}
    for name,bundle in comparison['bundles'].items():
        if bundle['implementation']!=implementation_hashes('webgpu'):
            raise ValueError('implementation changed after measurement')
        shaders[name]={}
        for key,record in bundle['kernels'].items():
            artifact=root/name/record['artifact']
            if digest(artifact)!=record['sha256']:raise ValueError('artifact changed')
            manifest,files=read_artifact(artifact)
            shaders[name][key]=dict(sha256=hashlib.sha256(files['kernel.wgsl']).hexdigest(),
                text=files['kernel.wgsl'].decode(),
                workgroup_storage_bytes=manifest['webgpu']['workgroup_storage_bytes'])
    bitwise={row['name']:np.array_equal(
        np.load(root/f'runtime-ablation/{i}-tensor_before.npy',allow_pickle=False),
        np.load(root/f'runtime-ablation/{i}-tensor_searched.npy',allow_pickle=False))
        for i,row in enumerate(runtime['validation'])}
    if not all(bitwise.values()):raise ValueError('runtime-only logits changed')
    chunk_bitwise={row['name']:np.array_equal(
        np.load(root/f'comparison/{i}-baseline32.npy',allow_pickle=False),
        np.load(root/f'comparison/{i}-adaptive.npy',allow_pickle=False))
        for i,row in enumerate(comparison['validation'])}
    sources={p:dict(sha256=digest(p),text=Path(p).read_text()) for p in
        (__file__,'benchmarks/lfm2/runtime_compare.py','benchmarks/lfm2/decode_search_compare.py',
         'benchmarks/lfm2/webgpu_profile.py','src/tensor/providers/webgpu.py')}
    report=dict(schema='tensor.lfm2-quantized-revisit.v1',status='passed',
        repository_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        comparison=comparison,runtime_ablation=runtime,
        runtime_logits_bitwise_equal=bitwise,chunk_logits_bitwise_equal=chunk_bitwise,
        bundle_shaders=shaders,sources=sources,
        native_release=json.loads((Path(reference)/'release.json').read_text()),
        native_logs={p.name:p.read_text(encoding='utf-8') for p in (root/'comparison').glob('llama*.log')},
        profiling={p.name:json.loads(p.read_text()) for p in root.glob('profile-*.json')},
        test_logs={p.name:p.read_text() for p in root.parent.glob('lfm2-2.6b-revisit*tests.log')},
        validation_arrays={str(p.relative_to(root)):digest(p) for folder in ('comparison','runtime-ablation')
                           for p in (root/folder).glob('*.npy')})
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2)+'\n')
    print('Archived',out.stat().st_size,'bytes;',sum(bitwise.values()),'bitwise runtime fixtures.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.root,args.reference,args.out)
