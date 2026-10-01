"""Repeat one scaling workload with the same correctness and allocating-call protocol."""
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

from benchmarks.inference.webgpu_scaling_benchmark import (
    cases, check_output, consumer_hashes, digest, inputs_reference, samples)
from scripts.validation.webgpu_validation import compiler_guard


@compiler_guard()
def run(directory, name):
    import tensor as tx
    suite_path = directory / 'suite.json'
    suite = json.loads(suite_path.read_text())
    if suite['schema'] != 'tensor.webgpu-scaling-suite.v1':
        raise ValueError('unknown suite schema')
    if suite['consumer_source_sha256'] != consumer_hashes(
            Path(tx.__file__).parent, suite['consumer_source_sha256']):
        raise ValueError('consumer differs from suite producer')
    expected = cases()
    index = next(i for i, row in enumerate(expected) if row['name'] == name)
    case = suite['cases'][index]
    if {key: case[key] for key in expected[index]} != expected[index]:
        raise ValueError('workload differs from scaling sweep')
    artifact = directory / case['artifact']
    if digest(artifact.read_bytes()) != case['artifact_sha256']:
        raise ValueError('artifact checksum mismatch')
    values, reference = inputs_reference(case, index)
    report = {'schema': 'tensor.webgpu-scaling-recheck.v1',
              'timestamp': datetime.now(timezone.utc).isoformat(), 'name': name,
              'suite_sha256': digest(suite_path.read_bytes()),
              'consumer_source_sha256': suite['consumer_source_sha256'],
              'benchmark_source_sha256': digest(Path(__file__).read_bytes()),
              'sampling_source_sha256': digest(Path(samples.__code__.co_filename).read_bytes()),
              'versions': {key: version(key) for key in ('numpy', 'wgpu', 'tensor-workspace')},
              'compiler_import_guard': True, 'warmup_calls': 20, 'samples_per_run': 45,
              'timing': 'one allocating call plus queue completion; release outside timer',
              'inputs_sha256': [digest(memoryview(value).cast('B')) for value in values],
              'wgsl_sha256': case['wgsl_sha256'], 'artifact_sha256': case['artifact_sha256']}
    with tx.Device(0, provider='webgpu', max_buffer_size=268435456) as device:
        report['adapter'] = device.info
        if device.info['adapter']['adapter_type'] != 'DiscreteGPU':
            raise ValueError('requires a physical discrete GPU')
        uploaded = [device.from_numpy(value) for value in values]
        try:
            with device.load(artifact) as kernel:
                output = kernel(*uploaded)
                try:
                    actual = output.to_numpy()
                    report['maximum_absolute_error'] = check_output(actual, reference)
                    report['output_sha256'] = digest(memoryview(actual).cast('B'))
                    report.update(atol=.002, rtol=.02)
                finally:
                    output.release()
                report['runs'] = [samples(kernel, uploaded, device) for _ in range(3)]
        finally:
            for buffer in uploaded:
                buffer.release()
    report['status'] = 'passed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', type=Path)
    parser.add_argument('--case', default='gemm-512-512-512')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    report = run(args.suite, args.case)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'medians_ms':
                      [row['median_us'] / 1000 for row in report['runs']]}))


if __name__ == '__main__':
    main()
