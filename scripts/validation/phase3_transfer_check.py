"""Verify a remote CI module closure and execute installed exports without compilers."""
from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import zipfile

from scripts.validation.phase1_transfer_check import exercise, NoCompilerImports
from scripts.validation.phase2_transfer_check import check as check_phase2
ROOT=Path(__file__).resolve().parents[2]


def check(archive,ci_record,platform):
    import sys
    sys.meta_path.insert(0,NoCompilerImports())
    baseline=check_phase2(archive,ci_record,platform)
    from tensor.artifacts.modules import Project, add, install, pack, _archive
    from tensor.cli import main
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='tensor-phase3-acceptance-') as directory:
        root=Path(directory)
        with zipfile.ZipFile(archive) as bundle:
            producer=json.loads(bundle.read('build/phase3-transfer/phase3-producer.json'))
            packaged=bundle.read('build/phase3-transfer/ops.tpack')
        assert hashlib.sha256(packaged).hexdigest()==producer['package_sha256']
        assert producer['revision']==baseline['producer_revision'] and producer['git_dirty'] is False
        assert producer['deterministic'] is True
        package=root/'relocated.tpack';package.write_bytes(packaged)
        graph,data=_archive(package)
        assert graph['packages']==producer['modules']
        for name in ('elementwise','gemm_relu','dynamic_affine','dynamic_gemm','scalar_offset'):
            module=data['tensor-base' if name=='elementwise' else 'tensor-ops']
            source=module.files[f'src/{name}.py']
            expected=subprocess.check_output(['git','show',f"{producer['revision']}:examples/{name}.py"],cwd=ROOT)
            assert source in (expected,expected.replace(b'\n',b'\r\n'))
        project=root/'app';project.mkdir()
        (project/'tensor.json').write_text(json.dumps({'formatVersion':1,'name':'consumer','version':'0.1.0','tensorAbi':1,'exports':{}}))
        cache=root/'modules'
        add(package,project,cache_dir=cache)
        package.unlink()  # frozen install must recover the closure from content-addressed snapshots
        install(project,cache_dir=cache,frozen=True)
        installed=Project(project,cache_dir=cache)
        artifacts,selections={},{}
        for name in ('elementwise','gemm_relu','dynamic_affine','dynamic_gemm','scalar_offset'):
            resolved=installed.module('tensor-base' if name=='elementwise' else 'tensor-ops').resolve(name,target='sm_86')
            assert resolved['selection']=='packaged'
            artifacts[name]=Path(resolved['path'])
            selections[name]={k:v for k,v in resolved.items() if k!='path'}
        runtime=exercise(artifacts)
        inputs=root/'inputs';inputs.mkdir()
        a=np.arange(129,dtype='float32');b=np.ones(129,dtype='float32')
        np.save(inputs/'a.npy',a);np.save(inputs/'b.npy',b)
        common=['--project',str(project),'--module-cache',str(cache),'--input',f'a={inputs}/a.npy','--input',f'b={inputs}/b.npy','--scalar','scale=2.5']
        assert main(['run','tensor-ops::dynamic_affine',*common,'--out-dir',str(root/'outputs')])==0
        np.testing.assert_array_equal(np.load(root/'outputs/c.npy'),2.5*a+b)
        assert main(['bench','tensor-ops::dynamic_affine',*common,'--warmup','1','--iters','3'])==0
        assert not list((cache/'artifacts').glob('*.tbin'))
        return {'status':'passed','ci_run':baseline['ci_run'],'producer_revision':producer['revision'],
                'consumer_revision':baseline['consumer_revision'],'consumer_dirty':baseline['consumer_dirty'],
                'two_hosts':True,'producer':producer,'package_sha256':producer['package_sha256'],
                'archive_sha256':baseline['archive_sha256'],'consumer_matches_producer_wheel':baseline['consumer_matches_producer_wheel'],
                'runtime':runtime,'selections':selections,'module_cli_run':True,'module_cli_bench':True,
                'relocated':True,'frozen_install_without_source_package':True,'local_compilation':False,
                'phase2_baseline':{'cases':len(baseline['records']),'diagnostics':len(baseline['diagnostics'])}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive',type=Path)
    parser.add_argument('--ci-record',type=Path,required=True)
    parser.add_argument('--platform',choices=('linux','windows'),required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();report=check(args.archive,args.ci_record,args.platform)
    with args.out.open('x',encoding='utf-8') as output:output.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'platform':args.platform,'cases':len(report['runtime']['records']),'diagnostics':len(report['runtime']['diagnostics'])}))
