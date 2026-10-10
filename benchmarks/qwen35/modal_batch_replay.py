"""Qualify larger resident batches against C8 groups, then replay HTTP stress."""
import modal
from pathlib import Path
from benchmarks.qwen35.modal_batch_scaling import image,volume
from benchmarks.qwen35.modal_h200 import MODEL,REVISION

app=modal.App('tensor-qwen35-h200-batch-replay')


@app.function(image=image,gpu='H200',cpu=8,memory=65536,timeout=7200,
              volumes={'/cache':volume},scaledown_window=2)
def measure(prepared,reference,run_id):
    import asyncio,hashlib,json,os,subprocess,sys,time
    import tensor
    import numpy as np
    from tensor_llm import Qwen35Batch,Qwen35MTP
    from benchmarks.qwen35.batch_scaling import workload,allocation_plan,observe,compare_records
    from benchmarks.qwen35.qualification_cache import qualify
    from benchmarks.llm_serving.runner import run as replay
    from benchmarks.qwen35.spec_quality import run as serial_quality
    from benchmarks.qwen35.spec_run import make_verifier,install_selected
    os.chdir('/workspace');volume.reload();slots=prepared['slots']
    out=Path('/cache/batch-runs')/run_id;out.mkdir(parents=True,exist_ok=False)
    data=workload(slots);tokens=np.asarray([r['prompt_token_ids'] for r in data['requests']],'int32')
    summary=dict(status='running',run_id=run_id,prepared=prepared,reference=reference,
        workload_sha256=data['sha256'],canonical_model_qualified=False,
        model_throughput_qualified=False,full_stress_target_reached=False,
        source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path('benchmarks/qwen35').glob('*.py'))})
    def save():
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');volume.commit()
    try:
        summary['allocation_plan']=allocation_plan(prepared)
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())*2**20
        if summary['allocation_plan']['required_with_reserve_bytes']>free:
            summary['status']='capacity-rejected';summary['free_device_bytes']=free
            return summary
        command=[sys.executable,'-m','pytest','packages/tensor-llm/tests/test_qwen_kernels.py',
            'packages/tensor-llm/tests/test_qwen_batch_scaling.py',
            'packages/tensor-llm/tests/test_qwen_recompute.py',
            'packages/tensor-llm/tests/test_qwen_spec.py',
            'packages/tensor-llm/tests/test_qwen_graph_pool.py','-q','-o','addopts=']
        test,summary['kernel_qualification']=qualify(command)
        (out/'kernel-tests.log').write_text(test.stdout+test.stderr);save()
        print(test.stdout,test.stderr,flush=True)
        if test.returncode:raise RuntimeError('batch kernel qualification failed')
        expected=[]
        with tensor.Device() as device:
            summary['device']=device.info
            if device.info['name']!='NVIDIA H200':raise RuntimeError('H200 allocation required')
            # One reference owner reused across groups; larger owner created only
            # after the reference closes, so weights are never duplicated in HBM.
            with Qwen35Batch(reference['base']['checkpoint'],reference['base']['paths']['decoder'],device,
                             progress=lambda value:print(value,flush=True)) as model:
                for offset in range(0,slots,8):
                    expected.extend(observe(model,reference,tokens[offset:offset+8]))
                    print('C8 reference group completed',offset//8+1,'/',slots//8,flush=True)
            with Qwen35Batch(prepared['base']['checkpoint'],prepared['base']['paths']['decoder'],device,
                             progress=lambda value:print(value,flush=True)) as model:
                actual=observe(model,prepared,tokens)
                comparison=compare_records(expected,actual)
                comparison['scope']='full 32K prefix, valid verifier outputs and seven rollback depths vs independent C8 groups'
                summary['batch_independence_quality']=comparison
                (out/'batch-quality.json').write_text(json.dumps(dict(report=comparison,reference=expected,candidate=actual),indent=2)+'\n')
                print('Batch independence quality',comparison,flush=True);save()
                if not comparison['passed']:raise RuntimeError('larger batch changes independent request results')
                if prepared.get('adaptive_verification'):
                    from benchmarks.qwen35.recompute_quality import run as pool_quality
                    summary['adaptive_quality']=pool_quality(model,prepared['adaptive_control_bundle'],
                        prepared['base']['paths']['verify'],out/'adaptive-quality',pooled=True,
                        lengths=np.resize(np.array([4,3,1,0,4,2,3,1],'int32'),slots))
                # observe restores the initialized real prefix for this unchanged gate.
                with Qwen35MTP(model,prepared['base']['paths']['draft']) as draft:
                    summary['serial_quality']=serial_quality(model,draft,prepared['base']['paths']['verify'],out/'serial-quality')
        save()
        paths=prepared['base']['paths'];name=f'tensor-h200-mtp-lookup-c{slots}'
        command=[sys.executable,'-m','benchmarks.qwen35.server','--checkpoint',prepared['base']['checkpoint'],
            '--bundle',paths['decoder'],'--prefill-bundle',prepared['prefill'],'--compact-experts',
            '--hopper-bundle',prepared['hopper'],'--attention-workspace-bundle',prepared['attention_workspace'],
            '--port','8013','--output-lookup','--fallback-proposals','3']
        if prepared['resident_speculative_graphs']:command.append('--resident-speculative-graphs')
        for option,key in [('draft-bundle','draft'),('draft-prefill-bundle','draft_prefill'),
                           ('verify-bundle','verify'),('repair-bundle','repair')]:command.extend(['--'+option,paths[key]])
        server=dict(name=name,engine='tensor',base_url='http://127.0.0.1:8013',model=MODEL,
            model_revision=REVISION,engine_version='0.1.0-native-qwen35-dev',weight_format='FP8-block128',
            kv_dtype='fp8',state_dtype='float32',tokenizer_name=MODEL,tokenizer_revision=REVISION,
            hardware='NVIDIA H200 x1',cpu_offload='none',prefix_cache=False,speculative=True,
            gpu_indices=['0'],command=command,model_throughput_qualified=False,full_stress_target_reached=False,
            settings=dict(max_model_len=48000,max_num_seqs=slots,prefill_chunk=prepared['prefill_chunk'],
                prefill_dense_block_m=128,verification_window=128,
                adaptive_verification=prepared.get('adaptive_verification',False),
                adaptive_windows=prepared.get('adaptive_windows',[128]),
                speculative_graphs_resident=prepared['resident_speculative_graphs'],prefill_attention_workspace_dtype='bfloat16',
                verification_attention_workspace_dtype='bfloat16',output_lookup=True,fallback_proposals=3,
                scheduler='fixed native cohort',recurrent_state_restore='accepted-prefix-recompute'))
        config=dict(schema='tensor.llm-serving-servers.v1',comparison_group='qwen35-h200-batch-scaling-32k16k-native-fp8kv-experimental',servers=[server])
        (out/'servers.json').write_text(json.dumps(config,indent=2)+'\n')
        print('Full client replay',slots,flush=True)
        summary['client_report']=asyncio.run(replay(config,data,out/name,concurrencies=(slots,),repeats=1,
            startup_timeout=600,timeout=2400,interval=1.))
        if summary['client_report']['status']!='completed':raise RuntimeError('batch replay incomplete')
        point=summary['client_report']['servers'][0]['points'][0]['summary']
        if point['failed'] or point['completed']!=slots or point['output_tokens']!=slots*16000:
            raise RuntimeError('incomplete stress output counts')
        summary['experimental_7000_observed']=point['output_tokens_per_second']>=7000
        summary['status']='measured-experimental'
        print('Completed batch',slots,point['output_tokens_per_second'],'tok/s',flush=True)
    except BaseException as error:
        summary.update(status='failed',error=f'{type(error).__name__}: {error}');raise
    finally:save()
    return summary


@app.local_entrypoint()
def main(prepared_dir: str='build/qwen35-h200-akbar-scaling',slots: int=16):
    import json
    from datetime import datetime,timezone
    root=Path(prepared_dir);reference=json.loads((root/'prepared-c8.json').read_text())
    prepared=json.loads((root/f'prepared-c{slots}.json').read_text())
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+f'-c{slots}-'+prepared['source_identity']
    dest=root/f'c{slots}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'run.json').exists():raise RuntimeError('batch already has a run record; refusing an automatic retry')
    (dest/'run.json').write_text(json.dumps(dict(run_id=run_id,volume='tensor-qwen35-h200'),indent=2)+'\n')
    result=measure.remote(prepared,reference,run_id)
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
