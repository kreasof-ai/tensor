"""Compare frontend CUDA bundles with the frozen native implementation."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from benchmarks.lfm2.cuda_formats import before_runner
from benchmarks.lfm2.cuda_transfer import compare, validate, digest

NATIVE_COMMIT='0cffba8584b1a4da051202591c5a04b57d7233f1'


def audit(bundle):
    """Verify compiled provenance, profile application and inspectable IR."""
    from tensor.artifacts.format import read_artifact
    from tensor.compiler.search import ScheduleProfile
    from tensor_llm.lfm2.kernels.cuda import source
    bundle=Path(bundle);data=json.loads((bundle/'inference.json').read_text())
    profile=ScheduleProfile({k:v for k,v in data['schedule_profile'].items() if k!='sha256'})
    assert profile.sha256==data['schedule_profile']['sha256']
    lowering=Path(__file__).resolve().parents[2]/'src/tensor/compiler/cuda_lowering.py'
    records=[]
    for value in data['kernels'].values():
        kind,p,schedule=value['kind'],value['parameters'],value['schedule']
        assert schedule==profile.select(kind,p,provider='cuda',target=data['target'])
        artifact=bundle/value['artifact'];manifest,_=read_artifact(artifact)
        assert digest(artifact)==value['sha256']
        assert manifest['source_sha256']==hashlib.sha256(source(kind,{**p,**schedule}).encode()).hexdigest()
        assert manifest['compiler']['lowering_sha256']==digest(lowering)
        with zipfile.ZipFile(artifact) as archive:ir=json.loads(archive.read('kernel.tirx.json'))
        nodes=ir['nodes'];externs=[]
        for node in nodes:
            if node['type']=='tirx.Call' and nodes[node['data']['op']].get('data')=='tirx.call_extern':
                argument=nodes[node['data']['args']]['data'][0]
                externs.append(nodes[nodes[argument]['data']['value']]['data'])
        assert set(externs)<={'tensor_load_u16','tensor_pack_f16x2'}
        records.append(dict(kind=kind,parameters=p,schedule=schedule,hardware_externs=sorted(set(externs)),
                            frontend_loops=sum(n['type']=='tirx.For' for n in nodes),artifact_sha256=value['sha256']))
    return dict(status='passed',profile_sha256=profile.sha256,kernels=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('validate','compare','audit'))
    p.add_argument('--models',type=Path,default=Path('build/lfm2-models'))
    p.add_argument('--root',type=Path,default=Path('build/lfm2-compiler-cleanup'))
    p.add_argument('--reference',type=Path,default=Path('build/lfm2-cuda-transfer/native-b11310/lfm2-reference'))
    p.add_argument('--formats',nargs='+',choices=('F16','Q4_0','Q4_K_M'),default=['F16','Q4_0','Q4_K_M'])
    p.add_argument('--depths',type=int,nargs='+')
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--generated',type=int,default=64)
    args=p.parse_args()
    native=before_runner(args.root/'tensor_llm_native',commit=NATIVE_COMMIT) if args.action=='compare' else None
    for fmt in args.formats:
        model=args.models/f'LFM2.5-2.6B-{fmt}.gguf'
        if args.action=='audit':
            report=audit(args.root/(fmt+'-frontend'))
            (args.root/(fmt+'-audit.json')).write_text(json.dumps(report,indent=2)+'\n')
        elif args.action=='validate':
            validate(model,args.root/(fmt+'-frontend'),args.root/(fmt+'-validation'),
                     **({'depths':tuple(args.depths)} if args.depths else {}))
        else:
            out=args.root/(fmt+'-comparison')
            report=compare(model,dict(native_cuda=args.root/(fmt+'-native'),frontend=args.root/(fmt+'-frontend')),
                           args.reference,out,depths=tuple(args.depths or (32,512,8192)),repeats=args.repeats,
                           generated=args.generated,runner_types={'native_cuda':native})
            report['native_cuda_commit']=NATIVE_COMMIT
            report['native_cuda_source_sha256']={file.name:digest(file) for file in sorted((args.root/'tensor_llm_native').glob('*.py'))}
            report['harness_sha256']['cuda_frontend_compare.py']=digest(__file__)
            (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
