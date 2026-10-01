"""Plot matched FP32 Tensor/CLBlast measurements; exclude unmatched HGEMM precision."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stock', type=Path)
    parser.add_argument('tuned', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    stock, tuned = [json.loads(path.read_text()) for path in (args.stock, args.tuned)]
    for report in (stock, tuned):
        if report['status'] != 'completed' or not report['matched_fp32_all_passed']:
            raise ValueError('requires completed and correct matched FP32 sweeps')
    original = {row['name']: row for row in stock['cases']}
    for row in tuned['cases']:
        if row['inputs_sha256'] != original[row['name']]['inputs_sha256']:
            raise ValueError('input mismatch')
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'tensor-clblast-rx6700xt'})
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for r, mode in enumerate(('gemm', 'linear')):
        panels = [
            ('Square matrices / allocating call', lambda row: row['m'] > 32, 'allocating'),
            ('Square matrices / GPU interval', lambda row: row['m'] > 32, 'gpu_interval'),
            ('FFN matrix shapes / allocating call', lambda row: row['m'] <= 32, 'allocating'),
        ]
        for ax, (title, select, timing) in zip(axes[r], panels):
            rows = [row for row in tuned['cases'] if row['dtype'] == 'float32' and row['mode'] == mode and select(row)]
            x = np.arange(len(rows))
            datasets = [
                ('Tensor WebGPU', '#d65a38', [row['tensor'][timing] for row in rows]),
                ('CLBlast stock', '#4c78a8', [original[row['name']]['clblast'][timing] for row in rows]),
                ('CLBlast bounded tuning', '#138b87', [row['clblast'][timing] for row in rows]),
            ]
            for i, (label, color, data) in enumerate(datasets):
                medians = np.array([entry['median_ms'] for entry in data])
                bounds = np.array([np.percentile(entry['samples_ms'], (25, 75)) for entry in data]).T
                error = np.maximum(0, np.vstack((medians-bounds[0], bounds[1]-medians)))
                ax.bar(x + (i-1)*.25, medians, width=.24, label=label, color=color,
                       yerr=error, capsize=2, error_kw={'linewidth': .8})
            labels = [str(row['m']) for row in rows] if rows[0]['m'] > 32 else [
                f"{row['m']}×{row['n']}×{row['k']}" for row in rows]
            ax.set_xticks(x, labels, rotation=20 if rows[0]['m'] <= 32 else 0)
            ax.set_yscale('log')
            ax.set_title(title)
            ax.grid(axis='y', which='major', alpha=.2)
            ax.set_axisbelow(True)
            ax.set_xlabel('M=N=K' if rows[0]['m'] > 32 else 'M×N×K (model shapes, random inputs)')
        axes[r, 0].set_ylabel(('Pure GEMM' if mode == 'gemm' else 'GEMM + bias/ReLU') + '\nLatency (ms, log scale)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.suptitle('RX 6700 XT: current generated kernels versus CLBlast 1.6.3', fontsize=16, y=.985)
    fig.text(.5, .925, 'Matched FP32 inputs, accumulation and correctness; 45 calls after 20 warmups\n'
             'GPU intervals: 20 samples; OpenCL spans preprocessing/epilogue and host submission idle; error bars = IQR',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .86))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=180)
    fig.savefig(args.out.with_suffix('.svg'))


if __name__ == '__main__': main()
