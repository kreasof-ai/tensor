"""Run the latency-scaling workloads on a standalone native WebGPU consumer.

Build: python benchmarks/inference/webgpu_scaling_benchmark.py --build build/webgpu-scaling
Consume: python benchmarks/inference/webgpu_scaling_benchmark.py --consume build/webgpu-scaling
         --compare docs/research/data/latency-scaling.json --out result.json
No CUDA or Torch is required. Input values differ from the recorded A10G run.
"""
from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions, version
import json
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import time

from scripts.validation.webgpu_validation import compiler_guard, source_hashes, specialize

ROOT = Path(__file__).resolve().parents[2]
LEGACY_CONSUMER_FILES = ('webgpu.py', 'webgpu_contract.py', 'abi.py', 'artifact.py',
                         'runtime.py', 'modules.py', 'providers.py')


def consumer_hashes(root, expected):
    # Retain strict provenance when replaying the frozen pre-package-layout suite.
    if set(expected) == set(LEGACY_CONSUMER_FILES):
        return {name: digest((root / name).read_bytes()) for name in LEGACY_CONSUMER_FILES}
    return source_hashes(root)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def cases():
    result = []
    for size in (129, 1048576, 4194304, 16777216, 67108864):
        result.append({'name': f'pointwise-{size}', 'profile': 'pointwise',
                       'shapes': [[size], [size]], 'dtype': 'float32',
                       'example': 'elementwise.py', 'constants': {'SIZE': size}})
    for size in (512, 1024, 2048, 4096):
        result.append({'name': f'gemm-{size}-{size}-{size}', 'profile': 'gemm',
                       'shapes': [[size, size], [size, size], [size]], 'dtype': 'float16',
                       'example': 'webgpu_gemm.py', 'constants': {
                           'M': size, 'N': size, 'K': size, 'DTYPE': 'float16',
                           'OUTPUT_DTYPE': 'float16', 'TRANSPOSE_B': True,
                           'USE_BIAS': True, 'RELU': True}})
    for length in (1024, 2048, 4096, 8192):
        for causal in (False, True):
            result.append({'name': f'sdpa-1-8-{length}-64-{causal}', 'profile': 'attention',
                           'shapes': [[1, 8, length, 64]] * 3, 'dtype': 'float16',
                           'example': 'flash_attention.py', 'constants': {
                               'BATCH': 1, 'HEADS': 8, 'SEQ_LEN': length, 'HEAD_DIM': 64,
                               'IS_CAUSAL': causal, 'BLOCK_M': 8, 'BLOCK_N': 16}})
    return result


def produce(directory):
    import tensor as tx
    from tensor.artifacts.format import read_artifact

    directory.mkdir(parents=True, exist_ok=False)
    suite = {'schema': 'tensor.webgpu-scaling-suite.v1', 'cases': [],
             'consumer_source_sha256': source_hashes(ROOT / 'src/tensor'),
             'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
             'producer': {'hostname': socket.gethostname(), 'tilelang': version('tilelang'),
                          'tvm_ffi': version('apache-tvm-ffi')},
             'compiler_source_sha256': {
                 name: digest((ROOT / 'src/tensor/compiler' / name).read_bytes())
                 for name in ('webgpu.py', 'webgpu_lowering.py')},
             'benchmark_source_sha256': digest(Path(__file__).read_bytes())}
    for case in cases():
        folder = directory / case['name']
        folder.mkdir()
        source = folder / 'source.py'
        source.write_text(specialize((ROOT / 'examples' / case['example']).read_text(), case['constants']))
        artifact = folder / 'kernel.tbin'
        start = time.perf_counter()
        built = tx.build(source, artifact, provider='webgpu', cache_dir=directory / 'cache')
        manifest, files = read_artifact(artifact)
        suite['cases'].append({**case, 'artifact': artifact.relative_to(directory).as_posix(),
                               'artifact_sha256': digest(artifact.read_bytes()),
                               'source_sha256': digest(source.read_bytes()),
                               'wgsl_sha256': digest(files['kernel.wgsl']),
                               'launch': manifest['launch'],
                               'workgroup_storage_bytes': manifest['webgpu']['workgroup_storage_bytes'],
                               'build_seconds': time.perf_counter() - start,
                               'build_cache_hit': built['cache_hit']})
        print(f"built {case['name']}", flush=True)
    (directory / 'suite.json').write_text(json.dumps(suite, indent=2) + '\n', encoding='utf-8')
    return suite


def inputs_reference(case, index):
    import numpy as np

    rng = np.random.default_rng(np.random.SeedSequence([42, index]))
    values = [rng.standard_normal(shape, dtype=np.float32).astype(case['dtype'], copy=False)
              for shape in case['shapes']]
    if case['profile'] == 'pointwise':
        reference = values[0] * 2
        reference += values[1]
        np.maximum(reference, 0, out=reference)
    elif case['profile'] == 'gemm':
        a, weight, bias = values
        reference = a.astype('float32') @ weight.astype('float32').T
        reference += bias.astype('float32')
        np.maximum(reference, 0, out=reference)
        reference = reference.astype('float16')
    else:
        q, k, v = [value.astype('float32') for value in values]
        reference = np.empty(q.shape, dtype='float16')
        length, depth = q.shape[-2:]
        keys = np.arange(length)[None, :]
        # Bound reference memory instead of materializing the full H*S*S matrix.
        for start in range(0, length, 64):
            stop = min(start + 64, length)
            scores = (q[..., start:stop, :] @ k.swapaxes(-1, -2)) * depth**-0.5
            if case['constants']['IS_CAUSAL']:
                scores = np.where(keys <= np.arange(start, stop)[:, None], scores, -np.inf)
            scores -= scores.max(axis=-1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= scores.sum(axis=-1, keepdims=True)
            reference[..., start:stop, :] = (scores @ v).astype('float16')
    return values, reference


def check_output(actual, reference):
    import numpy as np

    maximum = 0.0
    actual, reference = actual.reshape(-1), reference.reshape(-1)
    for start in range(0, actual.size, 1048576):
        a, b = actual[start:start+1048576], reference[start:start+1048576]
        np.testing.assert_allclose(a, b, atol=.002, rtol=.02)
        maximum = max(maximum, float(np.max(np.abs(a.astype('float32') - b.astype('float32')))))
    return maximum


def samples(kernel, uploaded, device):
    for _ in range(20):
        output = kernel(*uploaded)
        device.synchronize()
        output.release()
    observations = []
    for _ in range(9):
        device.synchronize()
        for _ in range(5):
            start = time.perf_counter()
            output = kernel(*uploaded)
            device.synchronize()
            observations.append((time.perf_counter() - start) * 1e6)
            output.release()
    return {'median_us': statistics.median(observations), 'samples_us': observations}


@compiler_guard()
def consume(directory, out, *, ordinal=0, compare=None, repeat_cases=()):
    import numpy as np
    import tensor as tx
    from wgpu.backends import wgpu_native

    suite_path = directory / 'suite.json'
    suite = json.loads(suite_path.read_text())
    if suite['schema'] != 'tensor.webgpu-scaling-suite.v1':
        raise ValueError('unknown scaling suite schema')
    if suite['consumer_source_sha256'] != consumer_hashes(
            Path(tx.__file__).parent, suite['consumer_source_sha256']):
        raise ValueError('consumer differs from suite producer; install the matching Tensor wheel')
    if [{key: case[key] for key in cases()[0]} for case in suite['cases']] != cases():
        raise ValueError('suite differs from the 17 latency-scaling workloads')
    names = {case['name'] for case in suite['cases']}
    if set(repeat_cases) - names:
        raise ValueError(f'unknown repeat cases: {set(repeat_cases) - names}')
    baseline = json.loads(compare.read_text()) if compare else None
    baseline_cases = {case['name']: case for case in baseline['cases']} if baseline else {}
    report = {'schema': 'tensor.webgpu-scaling-result.v1', 'status': 'running',
              'timestamp': datetime.now(timezone.utc).isoformat(),
              'platform': platform.platform(), 'python': platform.python_version(),
              'hostname': socket.gethostname(), 'source_commit': suite['source_commit'],
              'suite_sha256': digest(suite_path.read_bytes()),
              'consumer_source_sha256': suite['consumer_source_sha256'],
              'benchmark_source_sha256': digest(Path(__file__).read_bytes()),
              'compiler_import_guard': True,
              'packages': sorted(dist.metadata['Name'] for dist in distributions()),
              'versions': {**{name: version(name) for name in ('numpy', 'wgpu', 'tensor-workspace')},
                           'wgpu-native': wgpu_native.__version__},
              'methodology': {'warmup_calls': 20, 'batches': 9, 'calls_per_batch': 5,
                              'timing': 'one allocating call plus queue completion; release outside timer',
                              'excluded': 'uploads/downloads, CPU reference, build, pipeline creation',
                              'provider_order': 'one WebGPU provider; workloads in A10G sweep order',
                              'seed': 42, 'rng': 'NumPy PCG64, SeedSequence([42, case_index])',
                              'same_input_values_as_a10g': False,
                              'reference': 'NumPy FP32; attention uses 64-query chunks',
                              'max_buffer_size': 268435456},
              'cases': [], 'repeat': []}
    if baseline:
        report['comparison'] = {'gpu': baseline['gpu'], 'report_sha256': digest(compare.read_bytes()),
                                'same_operations_shapes_dtypes': True, 'same_input_values': False,
                                'separate_hosts_os_drivers': True}
    out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    with tx.Device(ordinal, provider='webgpu', max_buffer_size=268435456) as device:
        if device.info['adapter']['adapter_type'] != 'DiscreteGPU':
            raise ValueError('scaling comparison requires a physical discrete GPU')
        report['adapter'] = device.info
        save()
        order = [(index, case, False) for index, case in enumerate(suite['cases'])]
        order += [(index, case, True) for index, case in enumerate(suite['cases']) if case['name'] in repeat_cases]
        for index, case, repeat in order:
            print(f"{'repeat' if repeat else 'start'} {case['name']}", flush=True)
            artifact = directory / case['artifact']
            if digest(artifact.read_bytes()) != case['artifact_sha256']:
                raise ValueError(f"artifact checksum mismatch: {case['name']}")
            if baseline:
                matched = baseline_cases[case['name']]
                if case['shapes'] != [item['shape'] for item in matched['inputs']] or any(
                        item['dtype'] != 'torch.' + case['dtype'] for item in matched['inputs']):
                    raise ValueError(f"A10G workload mismatch: {case['name']}")
                evidence = next(item for item in matched['evidence'] if item.get('provider') == 'webgpu')
                if case['wgsl_sha256'] != evidence['wgsl_sha256']:
                    raise ValueError(f"WGSL differs from A10G: {case['name']}")
            start = time.perf_counter()
            values, reference = inputs_reference(case, index)
            reference_seconds = time.perf_counter() - start
            uploaded = []
            try:
                start = time.perf_counter()
                with device.load(artifact) as kernel:
                    pipeline_seconds = time.perf_counter() - start
                    uploaded = [device.from_numpy(value) for value in values]
                    hashes = [digest(memoryview(value).cast('B')) for value in values]
                    del values
                    output = kernel(*uploaded)
                    try:
                        actual = output.to_numpy()
                        maximum = check_output(actual, reference)
                        output_hash = digest(memoryview(actual).cast('B'))
                        del actual, reference
                    finally:
                        output.release()
                    timing = samples(kernel, uploaded, device)
                entry = {'name': case['name'], 'status': 'passed', 'shapes': case['shapes'],
                         'dtype': case['dtype'], 'inputs_sha256': hashes, 'output_sha256': output_hash,
                         'artifact_sha256': case['artifact_sha256'], 'wgsl_sha256': case['wgsl_sha256'],
                         'maximum_absolute_error': maximum, 'atol': .002, 'rtol': .02,
                         'reference_seconds': reference_seconds, 'pipeline_create_seconds': pipeline_seconds,
                         'serialized': timing}
                if baseline:
                    previous = matched['serialized']['median_us']['webgpu']
                    entry['a10g_comparison'] = {'same_wgsl': True, 'median_us': previous,
                                               'rx_over_a10g': timing['median_us'] / previous}
                report['repeat' if repeat else 'cases'].append(entry)
                save()
                print(f"passed {case['name']}: {timing['median_us']/1000:.3f} ms, max error {maximum:g}", flush=True)
            except Exception as error:
                report.update(status='failed', failed_case=case['name'], error=f'{type(error).__name__}: {error}')
                save()
                raise
            finally:
                for buffer in uploaded:
                    buffer.release()
    report['status'] = 'passed'
    save()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--build', type=Path)
    action.add_argument('--consume', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--repeat-case', action='append', default=[])
    args = parser.parse_args()
    if args.consume and not args.out:
        parser.error('--consume requires --out')
    result = produce(args.build) if args.build else consume(
        args.consume, args.out, ordinal=args.device, compare=args.compare, repeat_cases=args.repeat_case)
    print(json.dumps({'status': result.get('status', 'built'), 'cases': len(result['cases'])}))


if __name__ == '__main__':
    main()
