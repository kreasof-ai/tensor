"""Derive useful-work compute and bandwidth utilization from RX scaling timings.

These are completed-call estimates, not GPU counter or occupancy measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path

SPEC_URL = 'https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6700-xt.html'


def derive(report):
    if report['status'] != 'passed' or report['adapter']['name'] != 'AMD Radeon RX 6700 XT':
        raise ValueError('requires successful RX 6700 XT scaling evidence')
    result = {
        'schema': 'tensor.webgpu-utilization.v1',
        'gpu': 'AMD Radeon RX 6700 XT',
        'peak_specifications': {'source': SPEC_URL, 'verified_date': '2026-10-01',
                                'fp32_flops_per_second': 13.21e12,
                                'fp16_flops_per_second': 26.43e12,
                                'gddr6_bytes_per_second': 384e9},
        'methodology': {
            'time': 'median completed allocating call; includes host submission and queue completion',
            'compute': 'useful floating-point arithmetic per call divided by time and advertised peak',
            'fp32_denominator': 'WGSL GEMM/attention use f32 FMA, including f32 conversion of f16 inputs',
            'fp16_denominator': 'alternate normalization to packed FP16 vector peak; not the arithmetic emitted by this lowering',
            'fma_flops': 2,
            'pointwise_flops': '2*N; multiply/add counted, ReLU comparison excluded',
            'gemm_flops': '2*M*N*K; bias/ReLU excluded',
            'noncausal_attention_flops': '4*B*H*S*S*D; QK and PV only',
            'causal_attention_flops': '2*B*H*S*(S+1)*D; useful triangular QK and PV only',
            'attention_exclusions': 'softmax, scaling, normalization, masked/padded extra tile work',
            'minimum_io_bytes': 'read each input and write each output once; FP32=4 bytes, FP16=2 bytes',
            'useful_mbu': 'minimum tensor IO bytes divided by completed-call time and 384e9 bytes/s',
            'actual_dram_mbu': 'unknown: no DRAM counters, repeated tile loads and cache behavior measured',
            'clock': 'advertised peak, not measured effective clocks',
            'scope': 'workload MFU approximation; not whole-model MFU or hardware busy/occupancy percentage',
        },
        'cases': [], 'repeat': [],
    }
    for group in ('cases', 'repeat'):
        for case in report[group]:
            seconds = case['serialized']['median_us'] / 1e6
            if case['name'].startswith('pointwise'):
                size = case['shapes'][0][0]
                flops, io_bytes = 2*size, 12*size
            elif case['name'].startswith('gemm'):
                m, k = case['shapes'][0]
                n = case['shapes'][1][0]
                flops, io_bytes = 2*m*n*k, 2*(m*k+n*k+m*n+n)
            else:
                b, h, s, d = case['shapes'][0]
                flops = 2*b*h*s*(s+1)*d if case['name'].endswith('True') else 4*b*h*s*s*d
                io_bytes = 8*b*h*s*d  # FP16 Q/K/V read, output written.
            throughput, bandwidth = flops/seconds, io_bytes/seconds
            result[group].append({
                'name': case['name'], 'median_seconds': seconds,
                'useful_flops': flops, 'minimum_tensor_io_bytes': io_bytes,
                'useful_tflops_per_second': throughput/1e12,
                'mfu_fp32_percent': 100*throughput/13.21e12,
                'mfu_fp16_peak_percent': 100*throughput/26.43e12,
                'useful_io_gb_per_second': bandwidth/1e9,
                'useful_io_mbu_percent': 100*bandwidth/384e9,
                'actual_dram_mbu_percent': None,
            })
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raw = args.report.read_bytes()
    result = derive(json.loads(raw))
    result['timing_report_sha256'] = hashlib.sha256(raw).hexdigest()
    result['derivation_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    for case in result['cases']:
        print(f"{case['name']}: FP32 MFU {case['mfu_fp32_percent']:.3f}%; useful-IO MBU {case['useful_io_mbu_percent']:.3f}%")


if __name__ == '__main__':
    main()
