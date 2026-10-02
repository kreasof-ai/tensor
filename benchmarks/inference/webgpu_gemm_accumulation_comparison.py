"""Validate a compiler-only GEMM ablation measured by the CLBlast harness."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.inference.clblast_comparison import cases


def compare(before, after, old_suite, new_suite):
    for report in (before, after):
        if report['schema'] != 'tensor.clblast-comparison.v1' or report['status'] != 'completed':
            raise ValueError('requires completed GEMM sweeps')
        if [{key: row[key] for key in cases()[0]} for row in report['cases']] != cases():
            raise ValueError('requires all 32 GEMM/linear workloads')
    for key in ('platform', 'python', 'webgpu', 'opencl', 'library_sha256', 'probe_sha256',
                'benchmark_source_sha256', 'consumer_source_sha256', 'compiler_import_guard',
                'versions', 'methodology', 'tuning_sha256'):
        if before[key] != after[key]:
            raise ValueError(f'comparison conditions differ: {key}')
    if not before['compiler_import_guard']:
        raise ValueError('requires compiler-free measurement')
    for suite in (old_suite, new_suite):
        if suite['schema'] != 'tensor.clblast-comparison-suite.v1':
            raise ValueError('unexpected producer suite')
        if [{key: row[key] for key in cases()[0]} for row in suite['cases']] != cases():
            raise ValueError('producer inventory differs')
    result = dict(schema='tensor.webgpu-gemm-accumulation-comparison.v1', status='passed',
                  methodology=after['methodology'], same_inputs=True, same_outputs=True,
                  adapter=after['webgpu'], cases=[])
    for old, new, old_build, new_build in zip(before['cases'], after['cases'], old_suite['cases'], new_suite['cases']):
        name = old['name']
        for key in ('name', 'inputs_sha256', 'precision_matched', 'tuning_configuration'):
            if old[key] != new[key]:
                raise ValueError(f'{name}: comparison mismatch: {key}')
        for key in ('launch', 'workgroup_storage_bytes'):
            if old_build[key] != new_build[key]:
                raise ValueError(f'{name}: ablation changed {key}')
        if old['tensor']['correctness']['output_sha256'] != new['tensor']['correctness']['output_sha256']:
            raise ValueError(f'{name}: Tensor outputs differ')
        row = {key: old[key] for key in cases()[0]}
        row.update(precision_matched=old['precision_matched'], same_output=True,
                   wgsl_changed=old_build['wgsl_sha256'] != new_build['wgsl_sha256'], timings={})
        for report_row in (old, new):
            if not report_row['tensor']['correctness']['passed']:
                raise ValueError(f'{name}: failed Tensor correctness')
            for provider in ('tensor','clblast'):
                correctness=report_row[provider]['correctness']
                if correctness['atol'] != .002 or correctness['rtol'] != .02:
                    raise ValueError(f'{name}: precision tolerance differs')
            for provider in ('tensor', 'clblast'):
                for mode, count in (('allocating', 45), ('preallocated', 45), ('gpu_interval', 20)):
                    timing = report_row[provider][mode]
                    samples = timing['samples_ms']
                    if len(samples) != count or any(value <= 0 for value in samples):
                        raise ValueError(f'{name}: incomplete timing samples')
                    if timing['median_ms'] != statistics.median(samples):
                        raise ValueError(f'{name}: inconsistent timing median')
            if row['precision_matched'] and not report_row['clblast']['correctness']['passed']:
                raise ValueError(f'{name}: matched FP32 CLBlast failed correctness')
        for mode in ('allocating', 'preallocated', 'gpu_interval'):
            b, a = old['tensor'][mode]['median_ms'], new['tensor'][mode]['median_ms']
            row['timings'][mode] = dict(before_ms=b, after_ms=a, speedup=b/a,
                                       clblast_after_ms=new['clblast'][mode]['median_ms'],
                                       tensor_over_clblast=a/new['clblast'][mode]['median_ms'])
        result['cases'].append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'before_suite', 'after_suite'):
        parser.add_argument(name, type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    paths = [getattr(args, key) for key in ('before', 'after', 'before_suite', 'after_suite')]
    reports = [json.loads(path.read_text()) for path in paths]
    for report, path in zip(reports[:2], paths[2:]):
        if report['suite_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError('report differs from suite checksum')
    result = compare(*reports)
    result['provenance'] = {key: dict(path=path.as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                            for key, path in zip(('before', 'after', 'before_suite', 'after_suite'), paths)}
    result['comparison_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for row in result['cases']:
        t = row['timings']['allocating']
        print(f"{row['name']}: {t['before_ms']:.3f} -> {t['after_ms']:.3f} ms ({t['speedup']:.2f}x)")


if __name__ == '__main__':
    main()
