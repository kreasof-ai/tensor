"""Export the larger-shape latency measurements as a standalone research plot."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
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
    plt.rcParams.update({'font.size':10, 'svg.fonttype':'none'})
    fig, axes = plt.subplots(2,4,figsize=(17,8.5),gridspec_kw={'height_ratios':[1.35,1]})
    for index,label in enumerate(labels):
        rows = sorted(groups[label])
        sizes = [s for s,c in rows]
        top,bottom = axes[:,index]
        for key,(legend,color) in providers.items():
            medians = [c['serialized']['median_us'][key] for s,c in rows]
            low = [np.percentile(c['serialized']['samples_us'][key],25) for s,c in rows]
            high = [np.percentile(c['serialized']['samples_us'][key],75) for s,c in rows]
            top.plot(sizes,medians,'o-',label=legend,color=color,linewidth=1.8,markersize=4)
            top.fill_between(sizes,low,high,color=color,alpha=.1,linewidth=0)
        for key in ('tilelang_matched','triton','torch_eager'):
            legend,color = providers[key]
            ratios = [c['serialized']['median_us']['tensor_compile']/
                      c['serialized']['median_us'][key] for s,c in rows]
            bottom.plot(sizes,ratios,'o-',label=legend,color=color,linewidth=1.8,markersize=4)
        for case in repeats:
            repeat_label,size = family(case)
            if repeat_label != label:
                continue
            values = case['serialized']['median_us']
            for key,(_,color) in providers.items():
                top.scatter([size],[values[key]],facecolors='none',edgecolors=color,s=50,linewidths=1.2,zorder=4)
            for key in ('tilelang_matched','triton','torch_eager'):
                bottom.scatter([size],[values['tensor_compile']/values[key]],facecolors='none',
                               edgecolors=providers[key][1],s=50,linewidths=1.2,zorder=4)
        bottom.axhline(1,color='#777777',linestyle='--',linewidth=1)
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
    fig.legend(handles,legends,loc='upper center',ncol=5,bbox_to_anchor=(.5,.955),frameon=False)
    fig.suptitle('Larger-shape latency scaling — NVIDIA A10G',fontsize=17,weight='bold',y=.995)
    fig.text(.5,.02,'Warm allocating calls; synchronize after every invocation. Median of 45 observations; shaded bands show interquartile range.\n'
             'Hollow markers: targeted repeat. Fixed schedules, no autotuning. Bottom curves compare the full Tensor entry point with direct baselines.',
             ha='center',fontsize=10,color='#555555')
    fig.tight_layout(rect=(0,.075,1,.9),h_pad=2.5,w_pad=2)
    opts.out.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.svg','.png'):
        fig.savefig(opts.out.with_suffix(suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':
    main()
