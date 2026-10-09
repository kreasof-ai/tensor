"""Archive the completed searches and independent three-framework measurements."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,re,statistics
from pathlib import Path
from tensor.artifacts.format import read_artifact
from benchmarks.lfm2.tensor_projection_search import seeds

def native_samples(log,graph_repeats=1):
    records={};active=None;batch=None
    for line in log.splitlines():
        if line.startswith('PROFILE_BEGIN '):
            active=line.split()[1];records[active]=[];batch=None
        elif line.startswith('PROFILE_BATCH '):
            batch=int(line.split()[2]);records[active].append([])
        elif line.startswith('PROFILE_END '):active=None;batch=None
        elif active is not None and batch is not None and (match:=re.match(r'Total time: ([0-9.eE+-]+) us\.',line)):
            records[active][-1].append(float(match[1])*1e-6/graph_repeats)
    for suffix,batches in records.items():
        if len(batches)!=8 or any(len(batch)!=(20 if graph_repeats==1 else 1) for batch in batches):
            raise ValueError(f'incomplete native timestamp batches: {suffix}, {[len(b) for b in batches]}')
    return records

def run(search_root,recheck_root,native_root,json_out,native_default_root=None):
    root=Path(search_root);recheck_root=Path(recheck_root);native_root=Path(native_root)
    search=json.loads((root/'report.json').read_text());recheck=json.loads((recheck_root/'report.json').read_text())
    native=json.loads((native_root/'report.json').read_text());profile_log=(native_root/'run.log').read_text()
    if search['status']!='finished' or recheck['status']!='passed' or native['status']!='passed':
        raise ValueError('requires completed search and successful oracle checks')
    for row in search['records']:
        row['baseline']=next(c for c in row['candidates'] if c['config']==seeds()[0] and c['status']=='passed')
    samples=native_samples(profile_log,native['records'][0]['graph_repeats']);programs={};comparison=[]
    for row in recheck['records']:
        suffix=row['weight'];variants=row['variants'];winner=variants[row['tensor_winner']]
        for name,value in variants.items():
            if name.startswith('tensor-'):
                artifact=Path(value['artifact']);manifest,files=read_artifact(artifact)
                programs[f'{suffix}/{name}']={'manifest':manifest,'wgsl':files['kernel.wgsl'].decode(),
                                              'tile_program':artifact.with_suffix('.py').read_text()}
        programs[f'{suffix}/tinygrad']={'opencl':(recheck_root/f'{suffix}-tinygrad.cl').read_text()}
        gpu=samples[suffix][1:];native_variant=variants['llama.cpp']
        native_variant['gpu_samples_seconds']=gpu;native_variant['median_gpu_seconds']=statistics.median(statistics.median(batch) for batch in gpu)
        comparison.append({'weight':suffix,'shape':row['shape'],'tensor_winner':row['tensor_winner'],
                           'tensor_gpu_seconds':winner['median_batched_gpu_seconds'],
                           'tensor_completed_seconds':winner['median_completed_seconds'],
                           'baseline_gpu_seconds':variants['tensor-baseline']['median_batched_gpu_seconds'],
                           'baseline_completed_seconds':variants['tensor-baseline']['median_completed_seconds'],
                           'tinygrad_gpu_seconds':variants['tinygrad']['median_batched_gpu_seconds'],
                           'tinygrad_kernel_seconds':variants['tinygrad']['median_batched_kernel_seconds'],
                           'tinygrad_completed_seconds':variants['tinygrad']['median_completed_seconds'],
                           'llama_gpu_seconds':native_variant['median_gpu_seconds'],
                           'llama_completed_seconds':native_variant['median_completed_seconds']})
    source_paths=['src/tensor/compiler/webgpu_lowering.py','src/tensor/compiler/search.py','src/tensor/compiler/webgpu_schedules.py',
                  'packages/tensor-llm/src/tensor_llm/lfm2/kernels/webgpu.py',
                  'benchmarks/lfm2/tensor_projection_search.py','benchmarks/lfm2/projection_search_recheck.py',
                  'benchmarks/lfm2/llama_projection_compare.py','benchmarks/lfm2/projection_search_summary.py',
                  'tests/compiler/test_webgpu_search.py','tests/providers/test_webgpu_lowering.py']
    sources={p:{'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest(),'text':Path(p).read_text()} for p in source_paths}
    archive={'schema':'tensor.projection-search-comparison.v1','status':'passed','comparison':comparison,
             'tensor_search':search,'fresh_recheck':recheck,'native_profile':native,'native_profile_log':profile_log,
             'native_timestamp_samples':samples,'programs':programs,'sources':sources,
             'tests':{'discovery_and_legality_passed':9,'existing_cpu_lowering_passed':18,'native_partitioned_gpu_passed':12,
                      'native_partitioned_gpu_log':(root.parent/'tail-tests.log').read_text()},
             'limitations':['Hot isolated projections, not full-model throughput.',
                            'Tensor Vulkan/WGSL, tinygrad OpenCL, llama.cpp native Vulkan on the same physical GPU.',
                            'Tensor/tinygrad batch GPU intervals include inter-dispatch gaps; native GPU op timestamps collected separately on a 20-node graph with its logger and distinct outputs.',
                            'Common host FP16 rounding before the fresh comparison; Tensor search checked original unrounded F32 inputs.',
                            'Search resumed twice to broaden schedules and preserve family exploration; the aggregate active budget remains 30 minutes.',
                            'No automatic inference-bundle default change.']}
    if native_default_root:
        diagnostic_root=Path(native_default_root);diagnostic=json.loads((diagnostic_root/'report.json').read_text())
        log=(diagnostic_root/'run.log').read_text();gpu=native_samples(log,20)
        archive['native_default_diagnostic']={'report':diagnostic,'profile_log':log,'gpu_samples_seconds':gpu,
            'median_gpu_seconds':{name:statistics.median(statistics.median(b) for b in batches[1:]) for name,batches in gpu.items()},
            'interpretation':'Default FP16 accumulation fails the shared FP32-product oracle; retained as a different-precision performance diagnostic.'}
    Path(json_out).parent.mkdir(parents=True,exist_ok=True);Path(json_out).write_text(json.dumps(archive,indent=2)+'\n')
    print(json.dumps(comparison,indent=2));print('archive bytes',Path(json_out).stat().st_size)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('search-root','recheck-root','native-root','json-out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--native-default-root',type=Path)
    a=p.parse_args();run(a.search_root,a.recheck_root,a.native_root,a.json_out,a.native_default_root)
