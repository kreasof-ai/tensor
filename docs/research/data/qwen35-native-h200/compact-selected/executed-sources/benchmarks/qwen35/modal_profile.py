"""Bounded H200 kernel diagnosis using the previously prepared native bundles.

modal run benchmarks/qwen35/modal_profile.py --prepared-file build/qwen35-h200/prepared.json
Event-instrumented uncaptured plans diagnose operation costs; they are not
throughput measurements or correctness qualification.
"""
import modal
from benchmarks.qwen35.modal_h200 import image, volume, VOLUME_NAME

app = modal.App('tensor-qwen35-h200-profile')


def profile_plan(executor, bundle, out):
    import ctypes as ct
    import json
    from pathlib import Path
    import time
    device = executor.device
    driver = device.driver
    path = Path(bundle)
    manifest_path = path/('prefill.json' if (path/'prefill.json').is_file() else 'inference.json')
    manifest = json.loads(manifest_path.read_text())
    rows = manifest['kernels']
    reverse = {id(kernel): key for key, kernel in executor.kernels.items()}
    pairs = []
    started = time.perf_counter()
    try:
        for entry in executor.plan:
            kernel, bound = entry[:2]
            key = reverse.get(id(kernel), 'target-head')
            descriptor = rows.get(key, {})
            kind = descriptor.get('kind', 'quantize' if key.endswith('-quantize')
                                  else 'merge' if key.endswith('-merge') else key)
            begin, end = ct.c_void_p(), ct.c_void_p()
            driver.call('cuEventCreate', ct.byref(begin), 0)
            pairs.append((key, kind, descriptor.get('parameters'), begin, end))
            driver.call('cuEventCreate', ct.byref(end), 0)
            driver.call('cuEventRecord', begin, device.stream)
            device._launch(kernel, bound)
            driver.call('cuEventRecord', end, device.stream)
        device.synchronize()
        records, groups = [], {}
        for key, kind, parameters, begin, end in pairs:
            value = ct.c_float()
            driver.call('cuEventElapsedTime', ct.byref(value), begin, end)
            milliseconds = float(value.value)
            records.append(dict(key=key, kind=kind, parameters=parameters,
                                milliseconds=milliseconds))
            groups[kind] = groups.get(kind, 0.)+milliseconds
        result = dict(device=device.info, total_kernel_milliseconds=sum(groups.values()),
                      instrumented_wall_seconds=time.perf_counter()-started,
                      groups=groups, records=records,
                      timing='sum of per-kernel CUDA event intervals in an uncaptured plan',
                      model_throughput_qualified=False)
        Path(out).write_text(json.dumps(result, indent=2)+'\n')
        print(Path(out).name, sorted(groups.items(), key=lambda r: -r[1])[:8], flush=True)
        return result
    finally:
        for _, _, _, begin, end in pairs:
            for event in (begin, end):
                if event.value:
                    driver.call('cuEventDestroy_v2', event)


@app.function(image=image, gpu='H200', cpu=16, memory=98304, timeout=600,
              volumes={'/cache': volume}, scaledown_window=2)
def diagnose(prepared, run_id):
    import hashlib
    import json
    import os
    from pathlib import Path
    import time
    import numpy as np
    import tensor
    from tensor_llm import Qwen35Batch, Qwen35Prefill, Qwen35Verifier
    from benchmarks.qwen35.whole_tune import Snapshot
    from benchmarks.qwen35.spec_run import install_selected
    os.chdir('/workspace')
    volume.reload()
    out = Path('/cache/profiles')/run_id
    out.mkdir(parents=True, exist_ok=False)
    paths = {k: Path(v) for k, v in prepared['paths'].items()}
    workload = json.loads(Path('workload.json').read_text())
    tokens = np.asarray([r['prompt_token_ids'] for r in workload['requests']], 'int32')
    summary = dict(run_id=run_id, prepared=prepared, workload_sha256=workload['sha256'],
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   phases={}, model_throughput_qualified=False)
    try:
        with tensor.Device() as device, Qwen35Batch(prepared['checkpoint'], paths['decoder'], device,
                                                   progress=lambda s: print(s, flush=True)) as model:
            summary['device'] = device.info
            if device.info['name'] != 'NVIDIA H200' or device.info['arch'] != 'sm_90':
                raise RuntimeError('H200 was not allocated')
            with_prefill = Qwen35Prefill(model, paths['prefill'])
            try:
                first = tokens[:, :with_prefill.chunk]
                for repetition in range(3):
                    model.reset()
                    with_prefill.forward(first)
                    model.reset()
                    model.active = np.ones(model.slots, 'int32')
                    model._write('active', model.active)
                    name = f'prefill-context0-r{repetition}'
                    summary['phases'][name] = profile_plan(with_prefill, paths['prefill'], out/(name+'.json'))
                model.reset()
                started = time.perf_counter()
                for offset in range(0, tokens.shape[1], with_prefill.chunk):
                    count = min(with_prefill.chunk, tokens.shape[1]-offset)
                    batch = np.zeros((model.slots, with_prefill.chunk), 'int32')
                    batch[:, :count] = tokens[:, offset:offset+count]
                    with_prefill.forward(batch, np.full(model.slots, count, 'int32'))
                    if (offset//with_prefill.chunk+1)%16 == 0:
                        print('Profiling prefix', offset+count, flush=True)
                summary['target_prefix_seconds'] = time.perf_counter()-started
                snapshot = Snapshot(model)
                last = tokens[:, -with_prefill.chunk:]
                for repetition in range(3):
                    snapshot.restore(model)
                    with_prefill.forward(last)
                    snapshot.restore(model)
                    name = f'prefill-context32000-r{repetition}'
                    summary['phases'][name] = profile_plan(with_prefill, paths['prefill'], out/(name+'.json'))
                snapshot.restore(model)
            finally:
                with_prefill.close()
            for repetition in range(3):
                snapshot.restore(model)
                model.step()
                snapshot.restore(model)
                name = f'ar-context32000-r{repetition}'
                summary['phases'][name] = profile_plan(model, paths['decoder'], out/(name+'.json'))
            snapshot.restore(model)
            verifier = Qwen35Verifier(model, paths['verify'])
            try:
                install_selected(verifier, paths['verify'])
                # Explicit diagnostic inputs: repeat each slot's pending token.
                # This is not a claim about MTP or output-lookup acceptance.
                proposals = np.repeat(snapshot.tokens[:, None], verifier.chunk, axis=1)
                summary['verification_inputs'] = 'each slot pending token repeated across eight rows'
                for repetition in range(3):
                    snapshot.restore(model)
                    verifier.forward(proposals)
                    verifier.commit(np.full(model.slots, verifier.chunk, 'int32'))
                    snapshot.restore(model)
                    name = f'verify-context32000-r{repetition}'
                    summary['phases'][name] = profile_plan(verifier, paths['verify'], out/(name+'.json'))
            finally:
                verifier.close()
                snapshot.restore(model)
    finally:
        (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        volume.commit()
    return summary


@app.local_entrypoint()
def main(prepared_file: str, out: str = 'build/qwen35-h200-profile'):
    from datetime import datetime, timezone
    import json
    from pathlib import Path
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    prepared = json.loads(Path(prepared_file).read_text())
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+prepared['source_identity']
    (root/'run.json').write_text(json.dumps(dict(volume=VOLUME_NAME, run_id=run_id), indent=2)+'\n')
    result = diagnose.remote(prepared, run_id)
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Retained profile:', VOLUME_NAME, '/profiles/'+run_id)
