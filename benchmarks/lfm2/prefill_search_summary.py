"""Archive full-model throughput, precision checks, and replay provenance."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,subprocess
from pathlib import Path
from tensor.artifacts.format import read_artifact


def run(root,out):
    root=Path(root);report=json.loads((root/'comparison/report.json').read_text())
    if report['status']!='passed' or len(report['benchmarks'])!=3:raise ValueError('requires completed throughput measurement')
    manifests=report['tensor_bundles'];baseline=manifests['tensor_baseline'];searched=manifests['tensor']
    decode=[]
    for key,record in baseline['kernels'].items():
        if record['parameters'].get('r',1)!=1:continue
        other=searched['kernels'].get(key)
        if other is None:raise AssertionError('decode coverage changed')
        _,left=read_artifact(root/'baseline'/record['artifact'])
        _,right=read_artifact(root/'searched'/other['artifact'])
        if left['kernel.wgsl']!=right['kernel.wgsl']:raise AssertionError('decode shader changed')
        decode.append({'key':key,'shader_sha256':hashlib.sha256(left['kernel.wgsl']).hexdigest()})
    sources={}
    for filename in ('packages/tensor-llm/src/tensor_llm/lfm2/model.py','packages/tensor-llm/src/tensor_llm/lfm2/kernels/webgpu.py',
                     'src/tensor/compiler/webgpu_lowering.py','benchmarks/lfm2/producer.py',
                     'benchmarks/lfm2/tinygrad_compare.py','benchmarks/lfm2/tinygrad_reference.py',
                     'benchmarks/lfm2/tinygrad_schedule_replay.py','benchmarks/lfm2/prefill_schedule_check.py',
                     'benchmarks/lfm2/prefill_search_summary.py','benchmarks/lfm2/vulkan_reference.py'):
        path=Path(filename);sources[filename]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'text':path.read_text()}
    numerical={}
    for name in report['benchmarks'][0]['samples']:
        rows=[row[name] for row in report['validation']]
        numerical[name]={'fixtures':len(rows),'max_relative_rms':max(row['relative_rms'] for row in rows),
                         'min_cosine':min(row['cosine'] for row in rows),
                         'matching_argmax':sum(row['argmax'][0]==row['argmax'][1] for row in rows),
                         'bitwise_reset':True}
    result={'schema':'tensor.lfm2-prefill-search.v1','status':'passed',
            'repository_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'full_model':report,'numerical_summary':numerical,
            'fused_ffn_check':json.loads((root/'fused-check/report.json').read_text()),
            'unchanged_decode_shaders':decode,'sources':sources}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'numerical':numerical,'benchmarks':[{key:value for key,value in row.items() if key!='samples'}
                                                       for row in report['benchmarks']],
                      'decode_shaders':len(decode)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();run(a.root,a.out)
