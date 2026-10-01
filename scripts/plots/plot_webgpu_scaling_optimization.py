"""Plot the same-machine before/after allocating-call scaling measurements."""
from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.inference.webgpu_scaling_comparison import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    before, after = [json.loads(path.read_text()) for path in (args.before, args.after)]
    compare(before, after)
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none',
                         'svg.hashsalt': 'tensor-wgpu-scaling-optimization'})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.8))
    groups = [
        ('Pointwise FP32', lambda name: name.startswith('pointwise'), 0),
        ('Linear + bias + ReLU FP16', lambda name: name.startswith('gemm'), 0),
        ('Attention noncausal FP16', lambda name: name.endswith('False'), 2),
        ('Attention causal FP16', lambda name: name.endswith('True'), 2),
    ]
    for ax, (label, select, dimension) in zip(axes, groups):
        rows = [row for row in after['cases'] if select(row['name'])]
        sizes = [row['shapes'][0][dimension] for row in rows]
        for title, color, report in (
                ('Frozen original shaders/runtime', '#d65a38', before),
                ('Current shaders/runtime', '#138b87', after)):
            values = np.array([row['serialized']['samples_us'] for row in report['cases']
                               if select(row['name'])]) / 1000
            ax.plot(sizes, np.median(values, axis=1), 'o-', label=title, color=color, markersize=4)
            ax.fill_between(sizes, np.percentile(values, 25, axis=1),
                            np.percentile(values, 75, axis=1), color=color, alpha=.15)
            for row in report['repeat']:
                if select(row['name']):
                    ax.scatter(row['shapes'][0][dimension], row['serialized']['median_us']/1000,
                               facecolors='none', edgecolors=color, s=60, zorder=5)
        ax.set_title(label)
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xticks(sizes, ['129', '1M', '4M', '16M', '64M'] if label.startswith('Pointwise')
                      else [str(size) for size in sizes])
        if label.startswith('Pointwise'):
            ax.tick_params(axis='x', labelrotation=30, labelsize=9)
        ax.set_xlabel('Elements' if label.startswith('Pointwise') else 'Square dimension'
                      if label.startswith('Linear') else 'Sequence length (B1/H8/D64)')
        ax.grid(True, which='major', alpha=.2)
    axes[0].set_ylabel('Completed allocating call (ms, log scale)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.5, .005))
    fig.suptitle('RX 6700 XT / Windows Vulkan: before and after compiler/runtime optimization',
                 fontsize=14, y=.985)
    fig.text(.5, .91, 'Identical inputs and workload shapes; 45 calls after 20 warmups\n'
             'Bands = interquartile range; hollow markers = repeats; prepared-plan encoding excluded',
             ha='center', va='top', fontsize=10)
    fig.tight_layout(rect=(0, .1, 1, .81))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out.with_suffix('.png'), dpi=180)
    fig.savefig(args.out.with_suffix('.svg'))


if __name__ == '__main__':
    main()
