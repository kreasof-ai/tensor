"""Bounded single-H200 qualification and C8 replay using pinned native artifacts.

Run: modal run benchmarks/qwen35/modal_h200.py --out build/qwen35-h200
Weights and compiled bundles persist in a Modal Volume. CPU preparation precedes
the GPU function; only the native engine runs, with no baseline engine jobs.
"""
import os
from pathlib import Path
import modal

ROOT = (Path(os.environ['TENSOR_WORKSPACE']) if 'TENSOR_WORKSPACE' in os.environ
        else Path(__file__).resolve().parents[2])
MODEL = 'Qwen/Qwen3.5-35B-A3B-FP8'
REVISION = '9d1823d2dee688a6b25e77009dc727688c44936e'
VOLUME_NAME = 'tensor-qwen35-h200'
WORKLOAD = Path(os.environ.get('TENSOR_H200_WORKLOAD',
                str(ROOT/'docs/research/data/qwen35-native-h200/workload.json')))
app = modal.App('tensor-qwen35-h200')
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (modal.Image.from_registry('nvidia/cuda:12.9.1-devel-ubuntu24.04', add_python='3.12')
         .pip_install('torch==2.8.0', index_url='https://download.pytorch.org/whl/cpu')
         .pip_install('numpy==1.26.4', 'regex==2026.9.10', 'tilelang==0.1.14',
                      'apache-tvm-ffi==0.1.12', 'huggingface_hub==0.36.0',
                      'tokenizers==0.22.2', 'aiohttp==3.14.3', 'pytest==9.1.1')
         .add_local_file(ROOT/'tools/bootstrap_nvrtc.py', '/tmp/bootstrap_nvrtc.py', copy=True)
         .run_commands('python /tmp/bootstrap_nvrtc.py --out /opt/tensor-nvrtc')
         .env({'PYTHONPATH': '/workspace:/workspace/src:/workspace/packages/tensor-llm/src',
               'TENSOR_WORKSPACE': '/workspace', 'TENSOR_NVRTC_HOME': '/opt/tensor-nvrtc',
               'TENSOR_CACHE_DIR': '/cache/compiler-cache',
               'TENSOR_QWEN_TARGET': 'sm_90', 'OMP_NUM_THREADS': '4'})
         .add_local_dir(ROOT/'src', '/workspace/src', ignore=['**/__pycache__/**'])
         .add_local_dir(ROOT/'packages/tensor-llm/src', '/workspace/packages/tensor-llm/src',
                        ignore=['**/__pycache__/**', '**/*.egg-info/**'])
         .add_local_dir(ROOT/'packages/tensor-llm/tests', '/workspace/packages/tensor-llm/tests',
                        ignore=['**/__pycache__/**'])
         .add_local_dir(ROOT/'benchmarks', '/workspace/benchmarks', ignore=['**/__pycache__/**'])
         .add_local_file(WORKLOAD, '/workspace/workload.json'))


@app.function(image=image, cpu=16, memory=98304, timeout=3600, volumes={'/cache': volume},
              scaledown_window=2)
def prepare():
    import hashlib
    import json
    import time
    from huggingface_hub import hf_hub_download, snapshot_download
    os.chdir('/workspace')
    model = Path('/cache/models')/REVISION
    index = hf_hub_download(MODEL, 'model.safetensors.index.json', revision=REVISION,
                           local_dir=model)
    shards = sorted(set(json.loads(Path(index).read_text())['weight_map'].values()))
    print('Downloading pinned official FP8 model', REVISION, flush=True)
    snapshot_download(MODEL, revision=REVISION, local_dir=model, max_workers=4,
                      allow_patterns=[*shards, 'config.json', 'model.safetensors.index.json',
                                      'tokenizer.json', 'tokenizer_config.json'])
    volume.commit()
    sources = sorted(Path('src').rglob('*.py'))
    sources += sorted(Path('packages/tensor-llm/src').rglob('*.py'))
    sources += sorted(Path('benchmarks/qwen35').glob('*.py'))
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    root = Path('/cache/bundles')/identity
    ready = root/'prepared.json'
    if ready.is_file():
        return json.loads(ready.read_text())
    root.mkdir(parents=True, exist_ok=True)
    from benchmarks.qwen35.profile_producer import produce as decoder
    from benchmarks.qwen35.prefill_producer import produce as prefill
    from benchmarks.qwen35.mtp_producer import produce as mtp
    from benchmarks.qwen35.mtp_prefill_producer import produce as mtp_prefill
    from benchmarks.qwen35.spec_producer import produce as verify
    from benchmarks.qwen35.spec_attention_producer import produce as attention
    from benchmarks.qwen35.spec_linear_producer import produce as linear
    started = time.perf_counter()
    print('Compiling native sm_90 artifacts', identity, flush=True)
    # Fresh source identities always use a separate cache directory.
    paths = {}
    paths['decoder'] = decoder(model, root/'decode-profile', kv_dtype='fp8', target='sm_90')
    volume.commit()
    paths['draft'] = mtp(model, paths['decoder'], root/'mtp')
    paths['prefill'] = prefill(model, root/'prefill512', chunk=512, block_m=64,
                               kv_dtype='fp8', packed_kv=True, target='sm_90')
    paths['draft_prefill'] = mtp_prefill(model, paths['prefill'], root/'mtp-prefill512')
    volume.commit()
    small = prefill(model, root/'prefill8', chunk=8, block_m=64,
                    kv_dtype='fp8', packed_kv=True, target='sm_90')
    verified = verify(small, root/'verify8')
    split = attention(verified, root/'verify8-attention', key_rows=32)
    paths['verify'] = linear(split, root/'verify8-selected', expert_block_m=32)
    repair = mtp_prefill(model, small, root/'repair8')
    paths['repair'] = attention(repair, root/'repair8-attention', key_rows=32)
    result = dict(checkpoint=str(model), paths={k: str(v) for k, v in paths.items()},
                  source_identity=identity, source_hashes=hashes, model=MODEL,
                  model_revision=REVISION, target='sm_90', volume=VOLUME_NAME,
                  preparation_seconds=time.perf_counter()-started)
    ready.write_text(json.dumps(result, indent=2)+'\n')
    volume.commit()
    return result


@app.function(image=image, gpu='H200', cpu=16, memory=98304, timeout=1800,
              volumes={'/cache': volume}, scaledown_window=2)
def measure(prepared, run_id):
    import asyncio
    from importlib.metadata import version
    import json
    import subprocess
    import sys
    import time
    volume.reload()
    os.chdir('/workspace')
    out = Path('/cache/runs')/run_id
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(status='running', run_id=run_id, prepared=prepared,
                   model_throughput_qualified=False, full_stress_target_reached=False)
    def save():
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        volume.commit()
    try:
        summary['nvidia_smi'] = subprocess.check_output(['nvidia-smi'], text=True)
        summary['versions'] = {n: version(n) for n in ('numpy', 'torch', 'tilelang',
                             'apache-tvm-ffi', 'huggingface_hub', 'aiohttp')}
        summary['versions']['modal'] = modal.__version__
        print('H200 numerical kernel checks', flush=True)
        env = dict(os.environ, TENSOR_QWEN_CUDA='1')
        test = subprocess.run([sys.executable, '-m', 'pytest', 'packages/tensor-llm/tests/test_qwen_spec.py',
                               '-q', '-o', 'addopts='], env=env, text=True, capture_output=True)
        (out/'kernel-tests.log').write_text(test.stdout+test.stderr)
        summary['kernel_tests_exit_code'] = test.returncode
        print(test.stdout[-2500:], test.stderr[-1000:], flush=True)
        if test.returncode:
            raise RuntimeError('H200 primitive qualification failed; see kernel-tests.log')
        import tensor
        import numpy as np
        from tensor_llm import Qwen35Batch, Qwen35MTP, Qwen35Prefill
        from tensor_llm.qwen35.mtp.prefill import Qwen35MTPPrefill
        from benchmarks.qwen35.spec_benchmark import initialize
        from benchmarks.qwen35.spec_quality import run as quality
        paths = {k: Path(v) for k, v in prepared['paths'].items()}
        workload = json.loads(Path('workload.json').read_text())
        tokens = np.asarray([r['prompt_token_ids'] for r in workload['requests']], 'int32').T
        with tensor.Device() as device:
            summary['device'] = device.info
            if device.info['name'] != 'NVIDIA H200' or device.info['arch'] != 'sm_90':
                raise RuntimeError('requested H200 hardware was not allocated')
            with Qwen35Batch(prepared['checkpoint'], paths['decoder'], device,
                             progress=lambda s: print(s, flush=True)) as target:
                with Qwen35MTP(target, paths['draft']) as draft:
                    prefix = Qwen35Prefill(target, paths['prefill'])
                    draft_prefix = Qwen35MTPPrefill(draft, paths['draft_prefill'])
                    try:
                        initialize(target, draft, prefix, draft_prefix, tokens)
                    finally:
                        draft_prefix.close()
                        prefix.close()
                    summary['verification_quality'] = quality(target, draft, paths['verify'], out/'serial-quality')
        save()
        # Run the existing client harness against a managed native HTTP process.
        from benchmarks.llm_serving.runner import run as replay
        from benchmarks.qwen35.spec_run import provenance
        (out/'native-provenance.json').write_text(json.dumps(provenance(paths, Path('workload.json')), indent=2)+'\n')
        summary['client_runs'] = {}
        for speculative in (False, True):
            name = 'tensor-h200-mtp-lookup-c8' if speculative else 'tensor-h200-ar-c8'
            command = [sys.executable, '-m', 'benchmarks.qwen35.server', '--checkpoint',
                       prepared['checkpoint'], '--bundle', str(paths['decoder']),
                       '--prefill-bundle', str(paths['prefill']), '--port', '8013']
            if speculative:
                for option, key in (('draft-bundle', 'draft'), ('draft-prefill-bundle', 'draft_prefill'),
                                    ('verify-bundle', 'verify'), ('repair-bundle', 'repair')):
                    command.extend(['--'+option, str(paths[key])])
                command.extend(['--output-lookup', '--fallback-proposals', '3'])
            server = dict(name=name, engine='tensor', base_url='http://127.0.0.1:8013', model=MODEL,
                          engine_version='0.1.0-native-qwen35-dev', model_revision=REVISION,
                          weight_format='FP8-block128', kv_dtype='fp8', state_dtype='float32',
                          tokenizer_name=MODEL, tokenizer_revision=REVISION, hardware='NVIDIA H200 x1',
                          cpu_offload='none', prefix_cache=False, speculative=speculative, gpu_indices=['0'],
                          command=command, model_throughput_qualified=False, full_stress_target_reached=False,
                          settings=dict(max_model_len=48000, max_num_seqs=8, prefill_chunk=512,
                                        output_lookup=speculative, scheduler='fixed native cohort'))
            config = dict(schema='tensor.llm-serving-servers.v1',
                          comparison_group='qwen35-h200-32k16k-c8-native-fp8kv-experimental', servers=[server])
            (out/(name+'-servers.json')).write_text(json.dumps(config, indent=2)+'\n')
            print('Full H200 client replay', name, flush=True)
            report = asyncio.run(replay(config, workload, out/name, concurrencies=(8,), repeats=1,
                                       startup_timeout=240, timeout=900, interval=1.0))
            summary['client_runs'][name] = report
            save()
            print('H200 client replay completed', name, report['status'], flush=True)
            if report['status'] != 'completed':
                raise RuntimeError('H200 client replay incomplete; retained failure disposition')
        summary['status'] = 'measured-experimental'
    except BaseException as error:
        summary.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        print('H200 run failed', summary['error'], flush=True)
        raise
    finally:
        save()
    return summary


@app.local_entrypoint()
def main(out: str = 'build/qwen35-h200', prepared_file: str = ''):
    from datetime import datetime, timezone
    import json
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    prepared = (json.loads(Path(prepared_file).read_text()) if prepared_file else prepare.remote())
    (root/'prepared.json').write_text(json.dumps(prepared, indent=2)+'\n')
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+prepared['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id, volume=VOLUME_NAME), indent=2)+'\n')
    result = measure.remote(prepared, run_id)
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Retained H200 results:', VOLUME_NAME, '/runs/'+run_id)
