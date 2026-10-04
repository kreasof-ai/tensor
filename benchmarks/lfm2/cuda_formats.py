"""Matched three-format CUDA comparison, retaining a frozen before runner."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from benchmarks.lfm2.cuda_transfer import compare, validate, digest

BEFORE_COMMIT='4c6872674756454982af8c87c6a1d77b3fc5ef02'


def compile_templates(out):
    """Compile every selected 2.6B prefill schedule without a GPU or checkpoint."""
    from benchmarks.lfm2.cuda_format_search import compile_source
    from tensor_llm.cuda_kernels import CUDA_PREFILL, source
    specs=[]
    for kind,r,k,o,q in CUDA_PREFILL:
        specs.append((kind,dict(r=r,k=k,o=o,type=q)))
    for q in (1,2,12,14):
        specs.append(('linear_add',dict(r=1,k=256,o=7,type=q)))
        specs.append(('ffn',dict(r=1,k=2048,o=10752,type=q)))
    specs.append(('attention_grouped',dict(r=1,h=32,kh=8,d=64,cap=8576,splits=16)))
    records=[]
    for kind,parameters in specs:
        artifact=compile_source(source(kind,parameters),out)
        records.append(dict(kind=kind,parameters=parameters,artifact=artifact.name))
        print('compiled',kind,parameters,flush=True)
    (Path(out)/'coverage.json').write_text(json.dumps(records,indent=2)+'\n')


def before_runner(directory):
    path=Path(directory).resolve()
    repo=Path(__file__).resolve().parents[2]
    for file in path.glob('*.py'):
        original=subprocess.check_output(['git','-C',str(repo),'show',
                    f'{BEFORE_COMMIT}:packages/tensor-llm/src/tensor_llm/{file.name}'])
        if file.read_bytes()!=original:raise ValueError('frozen before source differs from '+BEFORE_COMMIT+': '+file.name)
    spec=importlib.util.spec_from_file_location('tensor_llm_before',path/'__init__.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);_sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module.LFM2


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('compare','validate-before','compile-templates'))
    p.add_argument('--models',type=Path,default=Path('build/lfm2-models'))
    p.add_argument('--root',type=Path,default=Path('build/lfm2-cuda-formats'))
    p.add_argument('--formats',nargs='+',choices=('F16','Q4_0','Q4_K_M'),default=['F16','Q4_0','Q4_K_M'])
    p.add_argument('--reference',type=Path,default=Path('build/lfm2-cuda-transfer/native-b11310/lfm2-reference'))
    p.add_argument('--depths',type=int,nargs='+')
    p.add_argument('--repeats',type=int,default=5)
    args=p.parse_args()
    if args.action=='compile-templates':
        compile_templates(args.root/'template-checks')
        raise SystemExit(0)
    before=before_runner(args.root/'tensor_llm_before')
    for fmt in args.formats:
        model=args.models/f'LFM2.5-2.6B-{fmt}.gguf'
        if args.action=='validate-before':
            validate(model,args.root/(fmt+'-before'),args.root/(fmt+'-before-validation'),runner_type=before,
                     **({'depths':tuple(args.depths)} if args.depths else {}))
        else:
            out=args.root/(fmt+'-comparison')
            report=compare(model,dict(before=args.root/(fmt+'-before'),optimized=args.root/(fmt+'-optimized')),
                           args.reference,out,depths=tuple(args.depths or (32,128,512,2048,8192)),repeats=args.repeats,
                           runner_types={'before':before})
            report['before_commit']=BEFORE_COMMIT
            report['before_source_sha256']={p.name:digest(p) for p in sorted((args.root/'tensor_llm_before').glob('*.py'))}
            report['harness_sha256']['cuda_formats.py']=digest(__file__)
            (out/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
