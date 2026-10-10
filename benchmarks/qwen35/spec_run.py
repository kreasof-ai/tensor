"""Run an exact native Qwen35 C8 stress cohort with verified MTP proposals."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import tensor
from tensor_llm import Qwen35Batch,Qwen35MTP,Qwen35Prefill,Qwen35Verifier
from tensor_llm.qwen35.mtp.prefill import Qwen35MTPPrefill
from tensor_llm.qwen35.speculative.attention import install as install_attention
from tensor_llm.qwen35.speculative.linear import install as install_linear
from .spec_benchmark import initialize,run
from .spec_telemetry import Monitor


def provenance(paths,workload):
    bundles={}
    for name,path in paths.items():
        manifest=path/('inference.json' if name in ('decoder','draft') else 'prefill.json')
        bundles[name]=dict(path=str(path.resolve()),manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                           manifest=json.loads(manifest.read_text()))
    import tensor_llm
    package=Path(tensor_llm.__file__).parent
    sources=[*sorted((package/'qwen35').rglob('*.py')),
             *sorted((package/'speculative').rglob('*.py')),
             *sorted((package/'common').rglob('*.py')),
             Path(__file__),Path(__file__).with_name('spec_benchmark.py')]
    return dict(bundles=bundles,workload_file_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})


def install_selected(executor,bundle):
    manifest=json.loads((Path(bundle)/'prefill.json').read_text())
    if 'split_attention' in manifest:install_attention(executor,bundle)
    if 'split_linear' in manifest:install_linear(executor,bundle)
    if 'compact_experts' in manifest:
        from tensor_llm.qwen35.compact_prefill import install as install_compact
        install_compact(executor,bundle)


def execute(target,draft,paths,workload,out,*,output_lookup=False,fallback_proposals=3):
    """Callable on a resident owner; no framework/compiler imports or weight reload."""
    paths={name:Path(path) for name,path in paths.items()};workload=Path(workload);out=Path(out)
    data=json.loads(workload.read_text());requests=data['requests']
    if len(requests)!=target.slots or len({len(r['prompt_token_ids']) for r in requests})!=1 or len({r['output_tokens'] for r in requests})!=1:
        raise ValueError('native cohort needs equal prompt/output lengths and one request per slot')
    tokens=np.asarray([r['prompt_token_ids'] for r in requests],dtype='int32').T
    out.mkdir(parents=True,exist_ok=True)
    (out/'provenance.json').write_text(json.dumps(provenance(paths,workload),indent=2)+'\n')
    verifier=repair=None
    with Monitor(out/'telemetry.jsonl') as monitor:
        start=time.perf_counter();prefill=Qwen35Prefill(target,paths['prefill'])
        try:
            draft_prefill=Qwen35MTPPrefill(draft,paths['draft_prefill'])
            try:initialize(target,draft,prefill,draft_prefill,tokens)
            finally:draft_prefill.close()
        finally:prefill.close()
        prefix_seconds=time.perf_counter()-start
        setup_start=time.perf_counter()
        try:
            verifier=Qwen35Verifier(target,paths['verify'])
            repair=Qwen35MTPPrefill(draft,paths['repair'])
            install_selected(verifier,paths['verify']);install_selected(repair,paths['repair'])
            setup_seconds=time.perf_counter()-setup_start
            report=run(target,draft,verifier,repair,out,output_tokens=requests[0]['output_tokens'],
                prefill_seconds=prefix_seconds+setup_seconds,output_lookup=output_lookup,
                fallback_proposals=fallback_proposals)
        finally:
            if repair:repair.close()
            if verifier:verifier.close()
    report.update(telemetry=monitor.summary(),workload_sha256=data['sha256'],
        workload_file_sha256=hashlib.sha256(workload.read_bytes()).hexdigest(),
        model_revision=data['tokenizer']['revision'],prompt_tokens_per_request=len(tokens),
        total_prompt_tokens=int(tokens.size),completed_requests=len(requests),prefix_seconds=prefix_seconds,
        execution_setup_seconds=setup_seconds,numerical_speed_gate=report['aggregate_output_tokens_per_second']>=600,
        timing='native cohort including prefix, execution setup, fill/drain and all speculative stages; excludes HTTP serialization',
        model_throughput_qualified=False,full_stress_target_reached=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('FULL NATIVE SPECULATIVE STRESS',report,flush=True);return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','decoder','draft','prefill','draft-prefill','verify','repair','workload','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--output-lookup',action='store_true');p.add_argument('--fallback-proposals',type=int,default=3)
    a=p.parse_args()
    paths={name:getattr(a,name) for name in ('decoder','draft','prefill','draft_prefill','verify','repair')}
    with tensor.Device() as device,Qwen35Batch(a.checkpoint,a.decoder,device,progress=lambda s:print(s,flush=True)) as target:
        with Qwen35MTP(target,a.draft) as draft:
            execute(target,draft,paths,a.workload,a.out,output_lookup=a.output_lookup,fallback_proposals=a.fallback_proposals)


if __name__=='__main__':main()
