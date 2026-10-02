"""Build explicit accumulation-mode controls for the fixed GEMM comparison suite."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from benchmarks.inference.clblast_comparison import ROOT, cases
from scripts.validation.webgpu_validation import source_hashes, specialize


def build(directory, mode):
    import tensor
    from tensor.artifacts.format import read_artifact
    directory.mkdir(parents=True,exist_ok=False)
    sha=lambda data:hashlib.sha256(data).hexdigest()
    suite=dict(schema='tensor.clblast-comparison-suite.v1', accumulation_mode=mode, cases=[],
               consumer_source_sha256=source_hashes(ROOT/'src/tensor'),
               compiler_source_sha256={name:sha((ROOT/'src/tensor/compiler'/name).read_bytes())
                                       for name in ('webgpu.py','webgpu_lowering.py')},
               builder_source_sha256=sha(Path(__file__).read_bytes()),
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
    for row in cases():
        constants=dict(M=row['m'],N=row['n'],K=row['k'],DTYPE=row['dtype'],OUTPUT_DTYPE=row['dtype'],
                       TRANSPOSE_B=True,USE_BIAS=row['mode']=='linear',RELU=row['mode']=='linear')
        text=specialize((ROOT/'examples/webgpu_gemm.py').read_text(),constants)
        text=text.replace('kernel = linear',f'kernel = linear.with_attr("tensor.webgpu.gemm_accumulation", "{mode}")')
        source=directory/(row['name']+'.py');source.write_text(text)
        artifact=source.with_suffix('.tbin')
        tensor.build(source,artifact,provider='webgpu',cache_dir=directory/'cache')
        manifest,files=read_artifact(artifact)
        suite['cases'].append(dict(**row,artifact=artifact.name,artifact_sha256=sha(artifact.read_bytes()),
                                   wgsl_sha256=sha(files['kernel.wgsl']),launch=manifest['launch'],
                                   workgroup_storage_bytes=manifest['webgpu']['workgroup_storage_bytes']))
        print('built',mode,row['name'],flush=True)
    (directory/'suite.json').write_text(json.dumps(suite,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--mode',choices=('auto','register','shared'),required=True)
    args=parser.parse_args();build(args.directory,args.mode)
