"""Export the larger-shape latency measurements as a standalone research plot."""
import argparse
import json
from pathlib import Path
import re

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


def webgpu_points(report, label):
    """Keep software-adapter workloads separate from CUDA ratio measurements."""
    series = {}
    for case in report['cases']:
        name = case['name']
        if label == 'Pointwise FP32' and name.startswith('affine_'):
            size, group = int(name.removeprefix('affine_')), 'affine'
        elif label == 'Square linear FP16' and re.fullmatch(r'gemm_\d+', name):
            size, group = int(name.removeprefix('gemm_')), 'gemm'
        else:
            match = re.fullmatch(r'attention_d(64|128)_s(\d+)_(full|causal)', name)
            if not match or label != ('Attention causal' if match[3] == 'causal' else 'Attention non-causal'):
                continue
            size = int(match[2])
            # S=512 changes B/H from 2/2 to 1/1; do not connect across that change.
            group = f'D{match[1]}, B/H={"1/1" if size == 512 else "2/2"}'
        if case['status'] != 'passed':
            raise ValueError(f'Cannot plot failed WebGPU case {name}')
        median_us = case['timing']['median_launch_and_sync_seconds'] * 1e6
        series.setdefault(group, []).append((size, median_us))
    return {group: sorted(points) for group, points in series.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--webgpu-report', type=Path, help='Optional native wgpu validation report')
    parser.add_argument('--out', type=Path, required=True, help='Output stem for SVG and PNG')
    opts = parser.parse_args()
    report = json.loads(opts.report.read_text())
    webgpu = json.loads(opts.webgpu_report.read_text()) if opts.webgpu_report else None
    if webgpu and (not webgpu.get('software_adapter') or
                   webgpu['adapter']['adapter']['adapter_type'] != 'CPU' or
                   'llvmpipe' not in webgpu['adapter']['name'].lower()):
        raise ValueError('This comparison labels WebGPU as llvmpipe CPU; supply software-adapter evidence')
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
    plt.rcParams.update({'font.size':10, 'svg.fonttype':'none', 'svg.hashsalt':'tensor-latency-scaling'})
    fig, axes = plt.subplots(2,4,figsize=(19,10),gridspec_kw={'height_ratios':[1.35,1]})
    webgpu_styles = {
        'default': ('Tensor wgpu, llvmpipe CPU', '#008b8b', 'D'),
        'D128': ('wgpu CPU, D128 attention', '#b07b00', '^'),
    }
    webgpu_handles = {}
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
        software_series = webgpu_points(webgpu, label) if webgpu else {}
        for group, points in software_series.items():
            style = 'D128' if group.startswith('D128') else 'default'
            legend, color, marker = webgpu_styles[style]
            handle, = top.plot([s for s,t in points],[t for s,t in points],
                               linestyle='--',marker=marker,label=legend,color=color,
                               linewidth=1.8,markersize=5)
            webgpu_handles.setdefault(style, handle)
            if label.startswith('Attention') and group.endswith('1/1'):
                top.annotate('B1 H1', points[-1], xytext=(5,5 if style == 'D128' else -13),
                             textcoords='offset points',color=color,fontsize=8)
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
        title = 'Square GEMM / linear FP16' if label == 'Square linear FP16' else label
        top.set_title(title, weight='bold', pad=24 if webgpu else 14)
        if webgpu:
            note = ('CUDA ×2; wgpu ×2.5' if index == 0 else
                    'CUDA bias/ReLU; wgpu plain GEMM' if index == 1 else
                    'CUDA B1 H8 D64; wgpu B2 H2 / B1 H1')
            top.text(.5,1.02,note,transform=top.transAxes,ha='center',fontsize=8,color='#555555')
        top.set_yscale('log')
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        top.yaxis.set_major_formatter(formatter)
        top.yaxis.set_minor_formatter(NullFormatter())
        for axis in (top,bottom):
            axis.set_xscale('log',base=2)
            ticks = [129,1048576,67108864] if index==0 else sizes
            texts = ['129','1M','64M'] if index==0 else [str(s) for s in sizes]
            if axis is top and software_series:
                if index == 0:
                    ticks, texts = [1,129,4097,1048576,67108864], ['1','129','4097','1M','64M']
                else:
                    ticks = sorted(set(sizes) | {s for points in software_series.values() for s,t in points})
                    texts = [str(s) for s in ticks]
            axis.set_xticks(ticks,texts)
            if axis is top and index >= 2 and software_series:
                axis.tick_params(axis='x',labelrotation=45,labelsize=8)
            axis.grid(True,alpha=.2)
            axis.spines[['top','right']].set_visible(False)
        bottom.set_xlabel('Elements' if index==0 else 'M = N = K' if index==1 else 'Sequence length (CUDA B=1 H=8 D=64)')
    axes[0,0].set_ylabel('Single-call time until completion (µs)')
    axes[1,0].set_ylabel('CUDA Tensor through torch.compile / baseline\n(lower favors Tensor)')
    handles,legends = axes[0,0].get_legend_handles_labels()
    for style, handle in webgpu_handles.items():
        legend = webgpu_styles[style][0]
        if legend not in legends:
            handles.append(handle)
            legends.append(legend)
    fig.legend(handles,legends,loc='upper center',ncol=4 if webgpu else 5,bbox_to_anchor=(.5,.955),frameon=False)
    fig.suptitle('Latency scaling — NVIDIA A10G CUDA and llvmpipe CPU' if webgpu else
                 'Larger-shape latency scaling — NVIDIA A10G',fontsize=17,weight='bold',y=.995)
    caption = ('CUDA: allocating calls, 45 samples / 20 warmups; bands: IQR; hollow markers: repeat. '
               'wgpu CPU: preallocated calls, 5 samples / 2 warmups; no IQR recorded.\n'
               'All calls include synchronization. Attention wgpu curves connect only equal B/H profiles. '
               'Workloads/devices differ; bottom ratios are CUDA only. No autotuning.' if webgpu else
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
