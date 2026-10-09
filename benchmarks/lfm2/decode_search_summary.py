"""Archive decode search, independent replay, and accepted model throughput."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,subprocess
from pathlib import Path
from tensor.artifacts.format import read_artifact


def run(root,out):
    root=Path(root);load=lambda path:json.loads((root/path).read_text())
    search=load('gemv/report.json');recheck=load('recheck/report.json');attention=load('attention/report.json');model=load('comparison/report.json')
    if search['status']!='finished' or recheck['status']!='passed' or attention['status']!='finished' or model['status']!='passed':
        raise ValueError('requires completed and validated measurements')
    profiles={name:load('profile-'+name+'.json') for name in ('before','after')}
    source_paths=['packages/tensor-llm/src/tensor_llm/lfm2/model.py','packages/tensor-llm/src/tensor_llm/lfm2/kernels/webgpu.py',
                  'src/tensor/compiler/webgpu_lowering.py','src/tensor/compiler/search.py','src/tensor/compiler/webgpu_schedules.py','benchmarks/lfm2/producer.py',
                  'benchmarks/lfm2/webgpu_profile.py','benchmarks/lfm2/decode_kernel_search.py',
                  'benchmarks/lfm2/decode_attention_search.py','benchmarks/lfm2/decode_search_recheck.py',
                  'benchmarks/lfm2/decode_search_compare.py','benchmarks/lfm2/decode_search_summary.py',
                  'tests/compiler/test_webgpu_decode_schedule.py','tests/providers/test_webgpu_lowering.py']
    sources={p:{'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'text':Path(p).read_text()} for p in source_paths}
    for key,path in (('tensor_llm.lfm2.model',source_paths[0]),('tensor_llm.lfm2.kernels.webgpu',source_paths[1])):
        if model['bundles']['searched']['implementation'][key]!=sources[path]['sha256']:raise ValueError('model source changed since measurement')
    shaders={}
    for name,directory in (('before','baseline'),('searched','searched')):
        shaders[name]={}
        for key,record in model['bundles'][name]['kernels'].items():
            artifact=root/directory/record['artifact'];manifest,files=read_artifact(artifact)
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('bundle checksum changed')
            shaders[name][key]={'sha256':hashlib.sha256(files['kernel.wgsl']).hexdigest(),'text':files['kernel.wgsl'].decode()}
    numerical={name:{'max_relative_rms':max(row[name]['relative_rms'] for row in model['validation']),
                     'min_cosine':min(row[name]['cosine'] for row in model['validation']),
                     'matching_argmax':sum(row[name]['argmax'][0]==row[name]['argmax'][1] for row in model['validation'])}
               for name in ('tensor_before','tensor_searched','llama_cpp')}
    result={'schema':'tensor.lfm2-decode-search.v1','status':'passed',
            'repository_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'projection_search':search,'projection_recheck':recheck,'attention_search':attention,
            'full_model':model,'profiles':profiles,'numerical_summary':numerical,'sources':sources,'bundle_shaders':shaders,
            'validation':{'cpu_discovery_tests':9,'cpu_decode_tests':8,'gpu_tail_and_fallback_tests':4,
                          'gpu_test_log':(root/'gemv-tests.log').read_text()}}
    out=Path(out);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'numerical':numerical,'candidates':sum(len(row['candidates']) for row in search['records']),
                      'benchmarks':[{k:v for k,v in row.items() if k!='samples'} for row in model['benchmarks']]},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
