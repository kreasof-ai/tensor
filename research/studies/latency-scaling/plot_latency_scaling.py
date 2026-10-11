"""Export the larger-shape latency measurements as a standalone research plot."""

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter, ScalarFormatter
import numpy as np


def family(case):
    name = case['name']
    if name.startswith('pointwise'):
        return 'Pointwise FP32', case['inputs'][0]['shape'][0]
    if name.startswith('gemm'):
        return 'Square linear FP16', case['inputs'][0]['shape'][0]
    return ('Attention causal' if name.endswith('True') else 'Attention non-causal',
            case['inputs'][0]['shape'][2])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--out', type=Path, required=True, help='Output stem for SVG and PNG')
    opts = parser.parse_args()
    report = json.loads(opts.report.read_text())
    webgpu = report.get('webgpu')
    if webgpu and (webgpu['software_adapter'] or webgpu['adapter']['adapter_type'] != 'DiscreteGPU' or
                   webgpu['name'] != report['gpu']):
        raise ValueError('The WebGPU comparison must use the same physical GPU as CUDA')
    groups = {}
    for case in report['cases']:
        label,size = family(case)
        groups.setdefault(label, []).append((size,case))
    repeats = report.get('repeat', {}).get('cases', [])
    labels = ('Pointwise FP32','Square linear FP16','Attention non-causal','Attention causal')
    providers = {
        'tensor_compile':('Tensor through torch.compile','#d54936'),
        'tensor_direct':('Direct Tensor C++','#222222'),
        'tilelang_matched':('TileLang, identical cubin','#3374b5'),
        'triton':('Triton compiled launcher','#8662ad'),
        'torch_eager':('Eager PyTorch','#36975b'),
    }
    ratio_providers = ['tilelang_matched','triton','torch_eager']
    if webgpu:
        if not all('webgpu' in case['serialized']['samples_us'] for case in report['cases']):
            raise ValueError('Every CUDA workload needs a measured WebGPU counterpart')
        providers['webgpu'] = ('Tensor wgpu, A10G Vulkan','#008b8b')
        ratio_providers.append('webgpu')
    plt.rcParams.update({'font.size':10, 'svg.fonttype':'none', 'svg.hashsalt':'tensor-latency-scaling'})
    fig, axes = plt.subplots(2,4,figsize=(19,10),gridspec_kw={'height_ratios':[1.35,1]})
    for index,label in enumerate(labels):
        rows = sorted(groups[label])
        sizes = [s for s,c in rows]
        top,bottom = axes[:,index]
        for key,(legend,color) in providers.items():
            medians = [c['serialized']['median_us'][key] for s,c in rows]
            low = [np.percentile(c['serialized']['samples_us'][key],25) for s,c in rows]
            high = [np.percentile(c['serialized']['samples_us'][key],75) for s,c in rows]
            top.plot(sizes,medians,'D--' if key == 'webgpu' else 'o-',
                     label=legend,color=color,linewidth=1.8,markersize=4)
            top.fill_between(sizes,low,high,color=color,alpha=.1,linewidth=0)
        for key in ratio_providers:
            legend,color = providers[key]
            ratios = [c['serialized']['median_us']['tensor_compile']/
                      c['serialized']['median_us'][key] for s,c in rows]
            bottom.plot(sizes,ratios,'D--' if key == 'webgpu' else 'o-',
                        label=legend,color=color,linewidth=1.8,markersize=4)
        for case in repeats:
            repeat_label,size = family(case)
            if repeat_label != label:
                continue
            values = case['serialized']['median_us']
            for key,(_,color) in providers.items():
                if key not in values:
                    continue
                top.scatter([size],[values[key]],facecolors='none',edgecolors=color,s=50,linewidths=1.2,zorder=4)
            for key in ratio_providers:
                if key not in values:
                    continue
                bottom.scatter([size],[values['tensor_compile']/values[key]],facecolors='none',
                               edgecolors=providers[key][1],s=50,linewidths=1.2,zorder=4)
        bottom.axhline(1,color='#777777',linestyle='--',linewidth=1)
        if webgpu:
            bottom.set_yscale('log')
            bottom.yaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value:g}'))
            bottom.yaxis.set_minor_formatter(NullFormatter())
        else:
            bottom.set_ylim(bottom=0)
        top.set_title(label, weight='bold', pad=14)
        top.set_yscale('log')
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        top.yaxis.set_major_formatter(formatter)
        top.yaxis.set_minor_formatter(NullFormatter())
        for axis in (top,bottom):
            axis.set_xscale('log',base=2)
            ticks = [129,1048576,67108864] if index==0 else sizes
            texts = ['129','1M','64M'] if index==0 else [str(s) for s in sizes]
            axis.set_xticks(ticks,texts)
            axis.grid(True,alpha=.2)
            axis.spines[['top','right']].set_visible(False)
        bottom.set_xlabel('Elements' if index==0 else 'M = N = K' if index==1 else 'Sequence length (B=1 H=8 D=64)')
    axes[0,0].set_ylabel('Single-call time until completion (µs)')
    axes[1,0].set_ylabel('Tensor through torch.compile / baseline\n(lower favors Tensor)')
    handles,legends = axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legends,loc='upper center',ncol=3 if webgpu else 5,bbox_to_anchor=(.5,.955),frameon=False)
    fig.suptitle('Latency scaling — NVIDIA A10G, CUDA and WebGPU/Vulkan' if webgpu else
                 'Larger-shape latency scaling — NVIDIA A10G',fontsize=17,weight='bold',y=.995)
    caption = ('Identical operations, shapes and inputs on A10G; allocating calls plus synchronization, 45 samples / 20 warmups. Bands: IQR; hollow markers: repeat.\n'
               'wgpu uses portable scalar GEMM and 8×16 attention tiles. Upload/download and cold pipeline creation excluded. Bottom ratios use a log scale. No autotuning.' if webgpu else
               'Warm allocating calls; synchronize after every invocation. Median of 45 observations; shaded bands show interquartile range.\n'
               'Hollow markers: targeted repeat. Fixed schedules, no autotuning. Bottom curves compare the full Tensor entry point with direct baselines.')
    fig.text(.5,.02,caption,
             ha='center',fontsize=10,color='#555555')
    fig.tight_layout(rect=(0,.075,1,.9),h_pad=2.5,w_pad=2)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.svg','.png'):
        output = opts.out.with_suffix(suffix)
        fig.savefig(output,dpi=180,bbox_inches='tight',
                    metadata={'Date':None} if suffix == '.svg' else None)
        if suffix == '.svg':
            output.write_text('\n'.join(line.rstrip() for line in output.read_text().splitlines())+'\n')
    plt.close(fig)


if __name__=='__main__':
    main()
