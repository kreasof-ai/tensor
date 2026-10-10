"""Build, qualify and replay compact H200 expert prefill against frozen control."""
import modal
from benchmarks.qwen35.modal_h200 import image, volume, VOLUME_NAME, MODEL, REVISION

app = modal.App('tensor-qwen35-h200-compact')


@app.function(image=image, cpu=16, memory=32768, timeout=600, volumes={'/cache': volume},
              scaledown_window=2)
def prepare_compact(prepared):
    import hashlib
    import json
    import os
    from pathlib import Path
    from benchmarks.qwen35.compact_producer import produce
    from tensor_llm.qwen35.compact_prefill import implementation_hashes
    os.chdir('/workspace')
    volume.reload()
    hashes = implementation_hashes()
    control=Path(prepared['paths']['prefill'])/'prefill.json'
    identity = hashlib.sha256(json.dumps(dict(implementation=hashes,
        control_manifest_sha256=hashlib.sha256(control.read_bytes()).hexdigest(),
        base_source_identity=prepared['source_identity']),sort_keys=True).encode()).hexdigest()[:16]
    bundle = produce(prepared['paths']['prefill'], Path('/cache/compact')/identity)
    result = dict(base=prepared, prefill=str(bundle), source_identity=identity, source_hashes=hashes)
    (bundle/'prepared.json').write_text(json.dumps(result, indent=2)+'\n')
    volume.commit()
    return result


@app.function(image=image, gpu='H200', cpu=16, memory=98304, timeout=1200,
              volumes={'/cache': volume}, scaledown_window=2)
def measure_compact(prepared, run_id):
    import asyncio
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    import time
    import numpy as np
    import tensor
    from tensor_llm import Qwen35Batch, Qwen35Prefill
    from tensor_llm.qwen35.compact_prefill import install
    from benchmarks.qwen35.whole_tune import Snapshot
    from benchmarks.llm_serving.runner import run as replay
    os.chdir('/workspace')
    volume.reload()
    out = Path('/cache/hopper-runs' if prepared.get('hopper') else '/cache/compact-runs')/run_id
    out.mkdir(parents=True, exist_ok=False)
    base = prepared['base']
    paths = {k: Path(v) for k,v in base['paths'].items()}
    workload = json.loads(Path('workload.json').read_text())
    tokens = np.asarray([r['prompt_token_ids'] for r in workload['requests']], 'int32')
    summary = dict(prepared=prepared, run_id=run_id, status='running',
                   model_throughput_qualified=False, full_stress_target_reached=False,
                   source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in sorted(Path('benchmarks/qwen35').glob('*.py'))})
    def save():
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        volume.commit()
    try:
        test_command=[sys.executable, '-m', 'pytest',
                    'packages/tensor-llm/tests/test_qwen_compact.py',
                    'packages/tensor-llm/tests/test_qwen_prefill.py::test_large_expert_tiles_preserve_hot_experts_and_tail_rows',
                    '-q', '-o', 'addopts=']
        if prepared.get('verification_chunk'):
            test_command.extend(['packages/tensor-llm/tests/test_qwen_hopper.py',
                f"packages/tensor-llm/tests/test_qwen_spec.py::test_recurrent_snapshots_restore_every_rejection_depth[{prepared['verification_chunk']}]"])
            hopper_manifest=json.loads((Path(prepared['hopper'])/'hopper.json').read_text()) if prepared.get('hopper') else {}
            if not hopper_manifest.get('hopper_attention'):
                test_command.extend(['-k','not prefill_attention'])
        if prepared.get('state_recompute'):
            test_command.append('packages/tensor-llm/tests/test_qwen_recompute.py')
        if prepared.get('profile_only'):
            from types import SimpleNamespace
            test=SimpleNamespace(returncode=0,stdout='Diagnostic only; primitive checks retained in prior replay.\n',stderr='')
        else:
            from benchmarks.qwen35.qualification_cache import qualify
            test,summary['kernel_qualification']=qualify(test_command)
        (out/'kernel-tests.log').write_text(test.stdout+test.stderr)
        print(test.stdout, test.stderr, flush=True)
        summary['kernel_tests_exit_code'] = None if prepared.get('profile_only') else test.returncode
        if test.returncode:
            raise RuntimeError('compact routing/expert primitive check failed')
        with tensor.Device() as device, Qwen35Batch(base['checkpoint'], paths['decoder'], device,
                                                   progress=lambda s: print(s, flush=True)) as model:
            summary['device'] = device.info
            if device.info['name']!='NVIDIA H200' or device.info['arch']!='sm_90':
                raise RuntimeError('H200 was not allocated')
            original = Qwen35Prefill(model, paths['prefill'])
            compact = Qwen35Prefill(model, prepared['prefill'])
            install(compact, prepared['prefill'])
            if prepared.get('hopper'):
                from tensor_llm.qwen35.hopper import install as install_hopper
                install_hopper(compact,prepared['hopper'])
            if prepared.get('attention_workspace'):
                from tensor_llm.qwen35.attention_workspace import install as install_workspace
                install_workspace(compact,prepared['attention_workspace'])
            summary['same_state_quality'] = []
            from benchmarks.qwen35.prefill_comparison import forward as control_forward
            def original_forward(batch,lengths):
                if compact.chunk==original.chunk:
                    return original.forward(batch,lengths,read_logits=True)
                # Compare a wider batch against successive frozen control
                # chunks from exactly the same initial model state.
                return control_forward(original,batch,lengths)
            def compare(label, batch, lengths):
                snapshot = Snapshot(model)
                result, logits = original_forward(batch,lengths)
                expected_state = [(b,a.copy()) for b,a in Snapshot(model).states]
                position = model.position.copy()
                snapshot.restore(model)
                repeated, repeated_logits = original_forward(batch,lengths)
                repeat_logit_error = float(np.linalg.norm(repeated_logits-logits)/max(np.linalg.norm(logits),1e-20))
                repeat_state_error = max(float(np.linalg.norm(b.to_numpy()-a)/max(np.linalg.norm(a),1e-20))
                                         for b,a in expected_state)
                print('Original prefill repeatability',label,repeat_logit_error,repeat_state_error,flush=True)
                snapshot.restore(model)
                actual, actual_logits = compact.forward(batch, lengths, read_logits=True)
                logit_error = float(np.linalg.norm(actual_logits-logits)/max(np.linalg.norm(logits),1e-20))
                state_error = max(float(np.linalg.norm(b.to_numpy()-a)/max(np.linalg.norm(a),1e-20))
                                  for b,a in expected_state)
                record = dict(label=label,control_chunk=original.chunk,candidate_chunk=compact.chunk,
                    relative_logit_rms=logit_error,
                    original_repeat_relative_logit_rms=repeat_logit_error,
                    original_repeat_maximum_relative_state_rms=repeat_state_error,
                    maximum_relative_state_rms=state_error, greedy_matches=int((actual==result).sum()),
                    greedy_total=model.slots, finite=bool(np.isfinite(actual_logits).all()),
                    positions_match=bool(np.array_equal(position,model.position)), threshold=1e-5)
                record['passed'] = (record['finite'] and record['positions_match']
                        and record['greedy_matches']==model.slots and logit_error<=1e-5 and state_error<=1e-5
                        and repeat_logit_error<=1e-5 and repeat_state_error<=1e-5)
                summary['same_state_quality'].append(record)
                print('Compact same-state quality', record, flush=True)
                snapshot.restore(model)
                if not record['passed']:
                    raise RuntimeError('compact prefill differs from original control')
            try:
                model.reset()
                check_lengths=np.array([compact.chunk,compact.chunk-1,compact.chunk//2+1,0,
                                        128,3,compact.chunk,compact.chunk-1],'int32')
                compare('early heterogeneous lengths', tokens[:,:compact.chunk],
                        check_lengths)
                model.reset()
                started = time.perf_counter()
                for offset in range(0,tokens.shape[1],compact.chunk):
                    count = min(compact.chunk,tokens.shape[1]-offset)
                    block = np.zeros((model.slots,compact.chunk),'int32')
                    block[:,:count] = tokens[:,offset:offset+count]
                    compact.forward(block,np.full(model.slots,count,'int32'))
                    if (offset//compact.chunk+1)%16==0:
                        print('Compact prefix',offset+count,flush=True)
                summary['compact_target_prefix_seconds'] = time.perf_counter()-started
                if compact.chunk!=original.chunk:
                    from benchmarks.qwen35.prefill_comparison import cache_prefix_hashes
                    selected_snapshot=Snapshot(model)
                    selected_logits=model.buffers['logits'].to_numpy()
                    selected_cache=cache_prefix_hashes(model)
                    model.reset()
                    for offset in range(0,tokens.shape[1],original.chunk):
                        count=min(original.chunk,tokens.shape[1]-offset)
                        block=np.zeros((model.slots,original.chunk),'int32')
                        block[:,:count]=tokens[:,offset:offset+count]
                        expected,expected_logits=original.forward(block,np.full(model.slots,count,'int32'),read_logits=True)
                    error=float(np.linalg.norm(selected_logits-expected_logits)/max(np.linalg.norm(expected_logits),1e-20))
                    state_error=max(float(np.linalg.norm(buffer.to_numpy()-value)/max(np.linalg.norm(value),1e-20))
                        for buffer,value in selected_snapshot.states)
                    record=dict(control_chunk=original.chunk,candidate_chunk=compact.chunk,
                        prompt_tokens_per_request=tokens.shape[1],relative_logit_rms=error,
                        maximum_relative_state_rms=state_error,
                        greedy_matches=int((selected_snapshot.tokens==expected).sum()),
                        greedy_total=model.slots,positions_match=bool(np.array_equal(model.position,selected_snapshot.position)),
                        valid_kv_prefix_bitwise_equal=selected_cache==cache_prefix_hashes(model),
                        finite=bool(np.isfinite(selected_logits).all()),threshold=1e-5)
                    record['passed']=(record['finite'] and error<=1e-5 and state_error<=1e-5
                        and record['greedy_matches']==model.slots and record['positions_match']
                        and record['valid_kv_prefix_bitwise_equal'])
                    summary['cross_chunk_prefix_quality']=record
                    print('Cross-chunk full-prefix quality',record,flush=True)
                    selected_snapshot.restore(model)
                    if not record['passed']:raise RuntimeError('wider prefill differs across the full frozen prefix')
                compare('32000 prefix heterogeneous lengths', tokens[:,-compact.chunk:],
                        check_lengths)
                if prepared.get('state_recompute'):
                    from benchmarks.qwen35.recompute_quality import run as recompute_quality
                    summary['recompute_quality']=recompute_quality(model,prepared['recompute_control_bundle'],
                        paths['verify'],out/'recompute-quality')
                if prepared.get('verification_chunk') and prepared.get('profile_phases',False):
                    from benchmarks.qwen35.modal_profile import profile_captured_plan
                    profile_snapshot=Snapshot(model)
                    summary['captured_prefill_profile']=profile_captured_plan(compact,prepared['prefill'],out/'captured-prefill-profile.json')
                    profile_snapshot.restore(model)
                if prepared.get('verification_chunk'):
                    from tensor_llm import Qwen35MTP
                    from benchmarks.qwen35.spec_quality import run as quality
                    with Qwen35MTP(model,paths['draft']) as draft:
                        summary['verification_serial_quality']=quality(model,draft,paths['verify'],out/'serial-quality')
                if prepared.get('verification_chunk') and prepared.get('profile_phases',False):
                    from tensor_llm import Qwen35Verifier
                    from benchmarks.qwen35.spec_run import install_selected,make_verifier
                    from benchmarks.qwen35.modal_profile import profile_captured_plan
                    from contextlib import closing
                    with closing(make_verifier(model,paths['verify'])) as verifier:
                        install_selected(verifier,paths['verify'])
                        proposals=np.repeat(model.buffers['tokens'].to_numpy()[:,None],verifier.chunk,axis=1)
                        verifier.forward(proposals,np.full(model.slots,verifier.chunk,'int32'))
                        summary['captured_verification_profile']=profile_captured_plan(
                            verifier,paths['verify'],out/'captured-verification-profile.json')
            finally:
                original.close(); compact.close()
        save()
        if prepared.get('profile_only'):
            summary['status']='diagnosed';return summary
        name = 'tensor-h200-compact-mtp-lookup-c8'
        command = [sys.executable,'-m','benchmarks.qwen35.server','--checkpoint',base['checkpoint'],
                   '--bundle',str(paths['decoder']),'--prefill-bundle',prepared['prefill'],
                   '--compact-experts','--port','8013','--output-lookup','--fallback-proposals',str(prepared.get('fallback_proposals',3))]
        if prepared.get('hopper'):
            command.extend(['--hopper-bundle',prepared['hopper']])
        if prepared.get('attention_workspace'):
            command.extend(['--attention-workspace-bundle',prepared['attention_workspace']])
        for option,key in (('draft-bundle','draft'),('draft-prefill-bundle','draft_prefill'),
                           ('verify-bundle','verify'),('repair-bundle','repair')):
            command.extend(['--'+option,str(paths[key])])
        server = dict(name=name,engine='tensor',base_url='http://127.0.0.1:8013',model=MODEL,
            model_revision=REVISION,engine_version='0.1.0-native-qwen35-dev',weight_format='FP8-block128',
            kv_dtype='fp8',state_dtype='float32',tokenizer_name=MODEL,tokenizer_revision=REVISION,
            hardware='NVIDIA H200 x1',cpu_offload='none',prefix_cache=False,speculative=True,
            gpu_indices=['0'],command=command,model_throughput_qualified=False,full_stress_target_reached=False,
            settings=dict(max_model_len=48000,max_num_seqs=8,prefill_chunk=json.loads((Path(prepared['prefill'])/'prefill.json').read_text())['chunk'],
                          compact_expert_tiles=True,hopper=bool(prepared.get('hopper')),
                          prefill_attention_workspace_dtype='bfloat16' if prepared.get('attention_workspace') else None,
                          prefill_attention_workspace_bytes=786432000 if prepared.get('attention_workspace') else 0,
                          verification_attention_workspace_dtype='bfloat16' if prepared.get('verification_attention_workspace') else None,
                          verification_attention_workspace_bytes=786432000 if prepared.get('verification_attention_workspace') else 0,
                          verification_window=prepared.get('verification_chunk',8),
                          recurrent_state_restore='accepted-prefix-recompute' if prepared.get('state_recompute') else 'snapshots',
                          output_lookup=True,fallback_proposals=prepared.get('fallback_proposals',3),scheduler='fixed native cohort'))
        config = dict(schema='tensor.llm-serving-servers.v1',
            comparison_group='qwen35-h200-compact-32k16k-c8-native-fp8kv-experimental',servers=[server])
        (out/'servers.json').write_text(json.dumps(config,indent=2)+'\n')
        # Retain the selected compact manifest and exact artifact hashes.
        (out/'compact-manifest.json').write_text((Path(prepared['prefill'])/'prefill.json').read_text())
        print('Full compact H200 client replay',flush=True)
        summary['client_report'] = asyncio.run(replay(config,workload,out/name,concurrencies=(8,),
            repeats=1,startup_timeout=240,timeout=900,interval=1.))
        if summary['client_report']['status']!='completed':
            raise RuntimeError('compact H200 replay incomplete')
        summary['status'] = 'measured-experimental'
    except BaseException as error:
        summary.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    return summary


@app.local_entrypoint()
def main(prepared_file: str, out: str = 'build/qwen35-h200-compact'):
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    root = Path(out)
    root.mkdir(parents=True,exist_ok=True)
    base = json.loads(Path(prepared_file).read_text())
    prepared = prepare_compact.remote(base)
    (root/'prepared.json').write_text(json.dumps(prepared,indent=2)+'\n')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+prepared['source_identity']
    (root/'run.json').write_text(json.dumps(dict(volume=VOLUME_NAME,run_id=run_id),indent=2)+'\n')
    result = measure_compact.remote(prepared,run_id)
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Retained compact replay:',VOLUME_NAME,'/compact-runs/'+run_id)
