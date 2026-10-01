"""Compare two standalone scaling sweeps, allowing explicitly recorded shader changes."""
from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path

from benchmarks.inference.webgpu_scaling_benchmark import cases


def compare(before, after):
    expected = cases()
    for report in (before, after):
        if report['schema'] != 'tensor.webgpu-scaling-result.v1' or report['status'] != 'passed':
            raise ValueError('comparison requires successful standalone scaling reports')
        if [row['name'] for row in report['cases']] != [row['name'] for row in expected]:
            raise ValueError('comparison requires all 17 workloads in sweep order')
        for row, case in zip(report['cases'], expected):
            if row['shapes'] != case['shapes'] or row['dtype'] != case['dtype']:
                raise ValueError(f"workload mismatch: {row['name']}")
    for key in ('hostname', 'platform', 'python', 'versions', 'methodology',
                'benchmark_source_sha256', 'compiler_import_guard'):
        if before[key] != after[key]:
            raise ValueError(f'comparison conditions differ: {key}')
    for key in before['adapter'].keys() | after['adapter'].keys():
        if key != 'features' and before['adapter'].get(key) != after['adapter'].get(key):
            raise ValueError(f'adapter conditions differ: {key}')
    result = {'schema': 'tensor.webgpu-scaling-optimization.v1', 'status': 'passed',
              'adapter': after['adapter'], 'methodology': after['methodology'],
              'same_inputs': True, 'same_host_os_driver_versions': True,
              'shader_changes_allowed': True,
              'enabled_features_before': before['adapter']['features'],
              'enabled_features_after': after['adapter']['features'],
              'scope': 'allocating kernel calls; does not exercise prepared-plan native encoding',
              'cases': [], 'repeat': []}
    for group in ('cases', 'repeat'):
        if [row['name'] for row in before[group]] != [row['name'] for row in after[group]]:
            raise ValueError('repeat inventories differ')
        for old, new in zip(before[group], after[group]):
            for key in ('name', 'shapes', 'dtype', 'inputs_sha256', 'atol', 'rtol'):
                if old[key] != new[key]:
                    raise ValueError(f"comparison mismatch for {old['name']}: {key}")
            if any(row['status'] != 'passed' or len(row['serialized']['samples_us']) != 45
                   for row in (old, new)):
                raise ValueError('each workload must pass correctness and retain 45 samples')
            b, a = [row['serialized']['median_us'] for row in (old, new)]
            result[group].append({
                'name': old['name'], 'shapes': old['shapes'], 'dtype': old['dtype'],
                'before_median_us': b, 'after_median_us': a, 'speedup': b / a,
                'latency_reduction_percent': 100 * (1 - a / b),
                'same_wgsl': old['wgsl_sha256'] == new['wgsl_sha256'],
                'same_output': old['output_sha256'] == new['output_sha256'],
                'before_maximum_absolute_error': old['maximum_absolute_error'],
                'after_maximum_absolute_error': new['maximum_absolute_error'],
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = compare(*[json.loads(path.read_text()) for path in (args.before, args.after)])
    result['reports'] = {
        name: {'path': path.as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in (('before', args.before), ('after', args.after))}
    result['comparison_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for row in result['cases']:
        print(f"{row['name']}: {row['before_median_us']/1000:.3f} -> "
              f"{row['after_median_us']/1000:.3f} ms ({row['speedup']:.2f}x), "
              f"same WGSL={row['same_wgsl']}")


if __name__ == '__main__':
    main()
