"""Plot the recorded A10G and standalone RX 6700 XT WebGPU scaling sweeps."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('a10g', type=Path)
    parser.add_argument('rx', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    a10g, rx = [json.loads(path.read_text()) for path in (args.a10g, args.rx)]
    if rx['status'] != 'passed' or len(rx['cases']) != 17:
        raise ValueError('plot requires the completed 17-workload sweep')
    baseline = {case['name']: case for case in a10g['cases']}
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'svg.hashsalt': 'tensor-wgpu-cross-gpu'})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
    groups = [
        ('Pointwise FP32', lambda name: name.startswith('pointwise'), 0),
        ('Linear + bias + ReLU FP16', lambda name: name.startswith('gemm'), 0),
        ('Attention noncausal FP16', lambda name: name.endswith('False'), 2),
        ('Attention causal FP16', lambda name: name.endswith('True'), 2),
    ]
    for ax, (label, select, dimension) in zip(axes, groups):
        rows = [case for case in rx['cases'] if select(case['name'])]
        sizes = [case['shapes'][0][dimension] for case in rows]
        for title, color, data in (
                ('A10G / Linux Vulkan', '#138b87', [baseline[c['name']]['serialized']['samples_us']['webgpu'] for c in rows]),
                ('RX 6700 XT / Windows Vulkan', '#d65a38', [c['serialized']['samples_us'] for c in rows])):
            values = np.array(data) / 1000
            ax.plot(sizes, np.median(values, axis=1), 'o-', label=title, color=color, markersize=4)
            ax.fill_between(sizes, np.percentile(values, 25, axis=1), np.percentile(values, 75, axis=1), color=color, alpha=.15)
        for repeat in rx['repeat']:
            if select(repeat['name']):
                ax.scatter(repeat['shapes'][0][dimension], repeat['serialized']['median_us']/1000,
                           facecolors='none', edgecolors='#d65a38', s=60, zorder=5)
        ax.set_title(label)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(sizes, ['129', '1M', '4M', '16M', '64M'] if dimension == 0 and label.startswith('Pointwise') else [str(s) for s in sizes])
        if label.startswith('Pointwise'):
            ax.tick_params(axis='x', labelrotation=30, labelsize=9)
        ax.set_xlabel('Elements' if label.startswith('Pointwise') else 'Square dimension' if label.startswith('Linear') else 'Sequence length (B1/H8/D64)')
        ax.grid(True, which='major', alpha=.2)
    axes[0].set_ylabel('Completed allocating call (ms, log scale)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .005))
    fig.suptitle('Same WebGPU workload sizes and byte-identical WGSL on two systems', fontsize=15, y=.985)
    fig.text(.5, .91, '45 calls after 20 warmups; bands = interquartile range; hollow markers = RX repeats\nDifferent input values, hosts, operating systems and drivers', ha='center', va='top', fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, .81))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=180)
    fig.savefig(args.out.with_suffix('.svg'))


if __name__ == '__main__':
    main()
