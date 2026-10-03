"""Archive the independent F16 replay, variant sweep and adaptive acceptance."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,subprocess
from pathlib import Path
from tensor.artifacts.format import read_artifact
from tensor_llm.provenance import implementation_hashes


def run(root,out):
    root=Path(root)
    load=lambda p:json.loads((root/p).read_text())
    search=load('search/report.json');sweep=load('comparison/report.json');final=load('adaptive-comparison/report.json')
    if search['status']!='finished' or sweep['status']!='passed' or final['status']!='passed':raise ValueError('requires validated completed runs')
    shaders={}
    for name,bundle in final['bundles'].items():
        if bundle['implementation']!=implementation_hashes('webgpu'):raise ValueError('implementation changed after final replay')
        shaders[name]={}
        for key,record in bundle['kernels'].items():
            artifact=root/name/record['artifact']
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('artifact changed')
            manifest,files=read_artifact(artifact)
            shaders[name][key]=dict(sha256=hashlib.sha256(files['kernel.wgsl']).hexdigest(),text=files['kernel.wgsl'].decode(),
                workgroup_storage_bytes=manifest['webgpu']['workgroup_storage_bytes'])
    selection=[dict(kind=v['kind'],rows=v['rows'],k=v['k'],o=v['o'],label=v['best']['label'],
        control_seconds=v['control']['replay_median_seconds'],best_seconds=v['best']['replay_median_seconds'],
        retained=v['best']['replay_median_seconds']<v['control']['replay_median_seconds']*.97) for v in search['groups']]
    report=dict(schema='tensor.lfm2-prefill-chase.v1',status='passed',
        repository_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        search=search,variant_sweep=sweep,adaptive_comparison=final,selection=selection,bundle_shaders=shaders,
        profiling={p.name:json.loads(p.read_text()) for p in root.glob('profile-*.json')},
        validation_arrays={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'adaptive-comparison').glob('*.npy')},
        test_logs={p.name:p.read_text() for p in root.parent.glob('lfm2-prefill-chase*tests.log')})
    out=Path(out);out.write_text(json.dumps(report,indent=2)+'\n')
    print('Archived',out.stat().st_size,'bytes; selected',sum(v['retained'] for v in selection),'of',len(selection),'shape configurations.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.out)
