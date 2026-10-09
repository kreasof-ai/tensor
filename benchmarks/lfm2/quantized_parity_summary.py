"""Archive the packed-kernel parity experiment and validated final bundles."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
from tensor.artifacts.format import read_artifact
from tensor_llm.lfm2.provenance import implementation_hashes
from tensor_llm.lfm2.model import QUANT_DECODE,QUANT_PREFILL,QUANT_ATTENTION


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(root,reference,out):
    root=Path(root)
    load=lambda p:json.loads((root/p).read_text())
    comparison=load('comparison-final/report.json')
    if comparison['status']!='passed':raise ValueError('requires passed full-model comparison')
    submission=load('submission-ablation/report.json')
    if submission['status']!='passed':raise ValueError('requires passed identical-shader submission ablation')
    if submission['bundles']['before']!=submission['bundles']['searched']:
        raise ValueError('submission ablation requires identical bundle manifests')
    if submission['bundles']['searched']['implementation']!=implementation_hashes('webgpu'):
        raise ValueError('implementation changed after submission measurement')
    for i in range(len(submission['validation'])):
        before=np.load(root/'submission-ablation'/f'{i}-tensor_before.npy')
        after=np.load(root/'submission-ablation'/f'{i}-tensor_searched.npy')
        if not np.array_equal(before,after):raise ValueError('submission changed logits')
    searches={name:load(name+'/report.json') for name in
        ('decode-search','q6-dot-search','prefill-search','prefill-replay','attention-search')}
    if any(r['status']!='finished' for r in searches.values()):raise ValueError('requires completed discovery and replay')
    if searches['prefill-replay']['discovery_report_sha256']!=digest(root/'prefill-search/report.json'):
        raise ValueError('replay discovery provenance changed')
    shaders={}
    for name,bundle in comparison['bundles'].items():
        if bundle['implementation']!=implementation_hashes('webgpu'):
            raise ValueError('implementation changed after final measurement')
        shaders[name]={}
        for key,record in bundle['kernels'].items():
            path=root/name/record['artifact']
            if digest(path)!=record['sha256']:raise ValueError('bundle artifact changed')
            manifest,files=read_artifact(path)
            shaders[name][key]=dict(sha256=hashlib.sha256(files['kernel.wgsl']).hexdigest(),
                text=files['kernel.wgsl'].decode(),workgroup_storage_bytes=manifest['webgpu']['workgroup_storage_bytes'])
    selection=dict(decode=[dict(k=k,o=o,type=q,parameters=p) for (k,o,q),p in QUANT_DECODE.items()],
        prefill=[dict(kind=kind,rows=r,k=k,o=o,type=q,parameters=p) for (kind,r,k,o,q),p in QUANT_PREFILL.items()],
        attention=QUANT_ATTENTION)
    for v in selection['prefill']:
        g=next(g for g in searches['prefill-replay']['groups'] if
               (g['kind'],g['rows'],g['k'],g['o'])==(v['kind'],v['rows'],v['k'],v['o']))
        if v['parameters']['outer']!=g['best']['parameters']['outer']:
            raise ValueError('prefill selection differs from corrected replay')
        if g['best']['replay_median_seconds']>=g['control']['replay_median_seconds']*.97:
            raise ValueError('prefill selection failed replay improvement gate')
    report=dict(schema='tensor.lfm2-quantized-parity.v1',status='passed',
        repository_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        comparison=comparison,searches=searches,selection=selection,bundle_shaders=shaders,
        submission_ablation=submission,
        runtime_before=load('runtime-before.json'),
        kernel_only_comparison=load('kernel-only-comparison/report.json'),
        discovery_control_note='Initial prefill discovery omitted the wide production tile for columns >=5120. Corrected prefill-replay includes projection_tile and independently replays the control and discovery finalists. Retained improvements use corrected replay and full-model rates.',
        sources={p:dict(sha256=digest(p),text=Path(p).read_text()) for p in
            (__file__,'packages/tensor-llm/src/tensor_llm/lfm2/model.py','packages/tensor-llm/src/tensor_llm/lfm2/kernels/webgpu.py',
             'src/tensor/compiler/webgpu_lowering.py','benchmarks/lfm2/prefill_chase_search.py',
             'src/tensor/providers/webgpu.py','src/tensor/native/webgpu_plan.c',
             'benchmarks/lfm2/runtime_compare.py','benchmarks/lfm2/prefill_chase_compare.py',
             'benchmarks/lfm2/decode_fusion_search.py','benchmarks/lfm2/quantized_decode_search.py')},
        native_release=json.loads((Path(reference)/'release.json').read_text()),
        native_logs={p.name:p.read_text(encoding='utf-8') for p in (root/'comparison-final').glob('llama*.log')},
        profiling={p.name:json.loads(p.read_text()) for p in root.glob('profile-*.json')},
        validation_arrays={f'{directory}/{p.name}':digest(p) for directory in ('comparison-final','submission-ablation')
                           for p in (root/directory).glob('*.npy')},
        test_logs={p.name:p.read_text() for p in root.parent.glob('lfm2-2.6b-parity*tests.log')})
    from tensor.providers import _webgpu_native
    report['native_helper_binary']=dict(path=_webgpu_native.__file__,sha256=digest(_webgpu_native.__file__))
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print('Archived',out.stat().st_size,'bytes; final full-model comparison passed.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(a.root,a.reference,a.out)
