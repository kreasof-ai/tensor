"""Archive the 2.6B prefill experiment, including accuracy and target shortfalls."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))
import argparse,hashlib,json,subprocess
from pathlib import Path
from tensor.artifacts.format import read_artifact
from tensor_llm.lfm2.provenance import implementation_hashes
from tensor_llm.lfm2.model import Q16_PREFILL,HALF_PREFILL


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def run(root,reference,out):
    root=Path(root);load=lambda p:json.loads((root/p).read_text())
    comparison=load('comparison-final/report.json')
    if comparison['status']!='passed' or comparison['protocol']['repeats']<7:
        raise ValueError('requires passed seven-sample full-model acceptance')
    shaders={}
    for name,bundle in comparison['bundles'].items():
        if bundle['implementation']!=implementation_hashes('webgpu'):
            raise ValueError('implementation changed after final measurement')
        shaders[name]={}
        for key,record in bundle['kernels'].items():
            path=root/name/record['artifact']
            if digest(path)!=record['sha256']:raise ValueError('artifact changed')
            manifest,files=read_artifact(path)
            if manifest['compiler']['lowering_sha256']!=digest('src/tensor/compiler/webgpu_lowering.py'):
                raise ValueError('compiler changed after final measurement')
            shaders[name][key]=dict(sha256=hashlib.sha256(files['kernel.wgsl']).hexdigest(),
                text=files['kernel.wgsl'].decode(),workgroup_storage_bytes=manifest['webgpu']['workgroup_storage_bytes'])
    searches={}
    for path in sorted(root.glob('*/report.json')):
        report=json.loads(path.read_text())
        if report.get('groups'):
            searches[path.parent.name]=dict(report_sha256=digest(path),report=report)
    selected_bundle='mixed' if 'mixed' in comparison['bundles'] else 'q16'
    target={str(row['prompt_tokens']):row[selected_bundle]['prefill_tokens_per_second']>=1000
            for row in comparison['benchmarks'] if row['prompt_tokens']>=128}
    if set(target)!={'128','384'}:raise ValueError('requires both target prompt lengths')
    source_paths=[__file__,'benchmarks/lfm2/q16_prefill_search.py','benchmarks/lfm2/prefill_chase_search.py',
        'benchmarks/lfm2/prefill_chase_compare.py','benchmarks/lfm2/producer.py',
        'benchmarks/lfm2/webgpu_profile.py',
        'packages/tensor-llm/src/tensor_llm/lfm2/model.py','packages/tensor-llm/src/tensor_llm/lfm2/kernels/webgpu.py',
        'packages/tensor-llm/src/tensor_llm/common/gguf.py','src/tensor/compiler/webgpu_lowering.py',
        'src/tensor/providers/webgpu.py','src/tensor/native/webgpu_plan.c',
        'tests/integration/test_lfm2_q16_prefill.py','tests/integration/test_lfm2_prefill_chase.py',
        'tests/integration/test_lfm2_prefill_tail.py',
        'tests/providers/test_webgpu_outer_product.py']
    report=dict(schema='tensor.lfm2-prefill-1k.v1',status='passed',target_tokens_per_second=1000,
        target_achieved=all(target.values()),target_by_prompt_length=target,
        selected_bundle=selected_bundle,
        repository_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        comparison=comparison,searches=searches,bundle_shaders=shaders,
        selection=[dict(kind=kind,rows=r,k=k,o=o,type=q,parameters=p) for (kind,r,k,o,q),p in Q16_PREFILL.items()],
        half_selection=[dict(kind=kind,rows=r,k=k,o=o,type=q,parameters=p) for (kind,r,k,o,q),p in HALF_PREFILL.items()],
        sources={p:dict(sha256=digest(p),text=Path(p).read_text()) for p in source_paths},
        native_release=json.loads((Path(reference)/'release.json').read_text()),
        native_logs={p.name:p.read_text(encoding='utf-8') for p in (root/'comparison-final').glob('llama*.log')},
        pilot=load('q16-pilot/report.json'),
        pilot_protocol_note='The early pilot timing description predates the new arithmetic. Its q16 bundle parameters select two-component activations and exact F32 weight values. Final protocol explicitly records this.',
        validation_arrays={p.name:digest(p) for p in (root/'comparison-final').glob('*.npy')},
        profiling={p.name:json.loads(p.read_text()) for p in root.glob('profile-*.json')},
        test_logs={p.name:p.read_text() for p in root.parent.glob('lfm2-prefill-1k*tests.log')})
    report['search_protocol_notes']={
        'q16-search':'Initial incomplete discovery stopped at duplicate quantizer artifact creation; retained only as diagnostic evidence, not acceptance.',
        'q16-fixed-search':'Fixed low-component scale high/254; flag and source snapshot record this even though its early protocol text does not.',
        'q16-prepacked-search':'Integer candidates used 36-byte signed-byte caches; the original protocol describes packed controls. Cache bytes, flag and source snapshot record the expanded candidate format.',
        'half-search':'Initial protocol predates partial F16 accumulation. half_accum=true and source snapshot define short even/odd half FMA chains with F32 totals; kernel bounds include half rounding and underflow.',
        'half-wide-search':'Some early labels omit microtile dimensions and repeat. Parameters and distinct artifact filenames identify each candidate.',
        'half-staged-ffn':'Direct F16 shared staging was slower and was reverted; its source snapshot records the rejected implementation.'}
    if (root/'mixed-pilot/report.json').exists():report['mixed_pilot']=load('mixed-pilot/report.json')
    if (root/'q16-prepacked-pilot/report.json').exists():report['prepacked_pilot']=load('q16-prepacked-pilot/report.json')
    report['intermediate_comparisons']={name:load(path) for name,path in
        (('mixed_before_liveness','comparison-926.json'),('last_row','last-row-pilot/report.json'),
         ('tail_after_attention','tail-pilot/report.json'),('query_tail_before_layout','comparison-994.json')) if (root/path).exists()}
    report['runtime_specialization']=dict(profile='prefill_mixed',public_rows=[1,32,128],internal_rows=8,
        suffix=['attention','conv','conv'],causal_receptive_field_rows=5,
        description='Geometry/type-guarded suffix optimization: the final attention stores every K/V before cropping queries and residual rows to eight. Both causal convolution histories and final output remain live. Only the final FFN token is evaluated. Intermediate chunks end after all persistent state updates; final chunks still produce logits, including read=False calls.',
        selection_note='Rows-128 schedules come from measured search and replay. Internal rows-8 schedules are legal hand-selected tiles, validated by independent suffix/attention tests and full-model gates.')
    from tensor.providers import _webgpu_native
    report['native_helper_binary']=dict(path=_webgpu_native.__file__,sha256=digest(_webgpu_native.__file__))
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print('Archived',out.stat().st_size,'bytes; target',target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();run(a.root,a.reference,a.out)
