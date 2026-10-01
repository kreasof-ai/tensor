"""Validate and summarize the complete stock/tuned CLBlast comparison evidence."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path
import statistics

from benchmarks.inference.clblast_comparison import cases


def summarize(stock, tuned, tuning):
    expected = cases()
    for report in (stock, tuned):
        if report['schema'] != 'tensor.clblast-comparison.v1' or report['status'] != 'completed':
            raise ValueError('requires complete comparison evidence')
        if not report['matched_fp32_all_passed']:
            raise ValueError('matched FP32 precision gate failed')
        if [{key: row[key] for key in expected[0]} for row in report['cases']] != expected:
            raise ValueError('workload inventory mismatch')
        for row in report['cases']:
            for provider in ('tensor', 'clblast'):
                for mode, count in (('allocating', 45), ('preallocated', 45), ('gpu_interval', 20)):
                    result = row[provider][mode]
                    if len(result['samples_ms']) != count or statistics.median(result['samples_ms']) != result['median_ms']:
                        raise ValueError('sampling protocol mismatch')
                if provider == 'tensor' or row['precision_matched']:
                    if not row[provider]['correctness']['passed']: raise ValueError('precision mismatch')
    for key in ('platform', 'python', 'opencl', 'webgpu', 'versions', 'methodology',
                'library_sha256', 'probe_sha256', 'suite_sha256',
                'benchmark_source_sha256', 'consumer_source_sha256'):
        if stock[key] != tuned[key]: raise ValueError(f'comparison provenance mismatch: {key}')
    if tuning['status'] != 'completed' or tuning['library_sha256'] != stock['library_sha256']:
        raise ValueError('tuning provenance mismatch')
    summary = {'schema': 'tensor.clblast-summary.v1', 'status': 'passed',
               'matched_fp32_profiles': 16, 'unmatched_fp16_profiles': 16,
               'tuning_candidates': len(tuning['candidates']),
               'tuning_candidates_passed': sum(row['status'] == 'passed' for row in tuning['candidates']),
               'clblast_fp16_failures_stock': sum(row['status'] == 'clblast_precision_failed' for row in stock['cases']),
               'clblast_fp16_failures_tuned': sum(row['status'] == 'clblast_precision_failed' for row in tuned['cases']),
               'fp32_peak_denominator_flops_per_second': 13.21e12,
               'peak_specification': 'same RX 6700 XT advertised FP32 denominator as latency-scaling.md',
               'cases': []}
    for old, new in zip(stock['cases'], tuned['cases']):
        if old['inputs_sha256'] != new['inputs_sha256']: raise ValueError('input mismatch')
        if old['tensor']['correctness']['output_sha256'] != new['tensor']['correctness']['output_sha256']:
            raise ValueError('Tensor repeat output mismatch')
        if not new['precision_matched']: continue
        row = {key: new[key] for key in ('name', 'm', 'n', 'k', 'dtype', 'mode')}
        flops = 2 * row['m'] * row['n'] * row['k']
        row.update(same_inputs=True, same_tensor_output=True, useful_gemm_flops=flops)
        for mode in ('allocating', 'preallocated', 'gpu_interval'):
            tensor, default, selected = (new['tensor'][mode]['median_ms'], old['clblast'][mode]['median_ms'],
                                         new['clblast'][mode]['median_ms'])
            row[mode] = dict(tensor_ms=tensor, clblast_stock_ms=default, clblast_tuned_ms=selected,
                             tensor_over_stock=tensor/default, tensor_over_tuned=tensor/selected,
                             tensor_tflops=flops/tensor/1e9, clblast_tuned_tflops=flops/selected/1e9,
                             tensor_useful_fp32_peak_percent=100*flops/(tensor/1000*13.21e12),
                             clblast_tuned_useful_fp32_peak_percent=100*flops/(selected/1000*13.21e12))
        summary['cases'].append(row)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('stock', 'tuned', 'tuning'): p.add_argument(name, type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    paths = [a.stock, a.tuned, a.tuning]
    result = summarize(*[json.loads(path.read_text()) for path in paths])
    result['reports_sha256'] = {path.as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    result['summary_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for row in result['cases']:
        t = row['allocating']
        print(f"{row['name']}: Tensor {t['tensor_ms']:.3f}, CLBlast stock {t['clblast_stock_ms']:.3f}, "
              f"tuned {t['clblast_tuned_ms']:.3f} ms ({t['tensor_over_tuned']:.2f}x)")


if __name__ == '__main__': main()
