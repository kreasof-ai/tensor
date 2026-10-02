"""Plot the whole-K accumulator ablation and its conservative default."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0,str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.inference.webgpu_gemm_accumulation_comparison import compare
from benchmarks.inference.webgpu_scaling_comparison import compare as compare_scaling


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('before','after','before_suite','after_suite','scaling_before','scaling_forced','scaling_after'):
        parser.add_argument(name,type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    load=lambda path:json.loads(path.read_text())
    b,a,bs,ass=[load(getattr(args,name)) for name in ('before','after','before_suite','after_suite')]
    compare(b,a,bs,ass)
    sb,sf,sa=[load(getattr(args,name)) for name in ('scaling_before','scaling_forced','scaling_after')]
    compare_scaling(sb,sa);compare_scaling(sb,sf)
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','svg.hashsalt':'tensor-whole-k'})
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for axis,mode in zip(axes[:2],('gemm','linear')):
        indices=[i for i,row in enumerate(a['cases']) if row['dtype']=='float32' and row['mode']==mode and row['m']>32]
        datasets=[('Before','#777777',[b['cases'][i]['tensor']['allocating']['samples_ms'] for i in indices]),
                  ('Default whole-K','#138b87',[a['cases'][i]['tensor']['allocating']['samples_ms'] for i in indices]),
                  ('CLBlast FP32','#4c78a8',[a['cases'][i]['clblast']['allocating']['samples_ms'] for i in indices])]
        draw(axis,[str(a['cases'][i]['m']) for i in indices],datasets)
        axis.set_title('FP32 '+('pure GEMM' if mode=='gemm' else 'linear + bias/ReLU'))
        axis.set_xlabel('M=N=K')
    indices=[i for i,row in enumerate(sa['cases']) if row['name'].startswith('gemm-')]
    datasets=[('Before','#777777',[np.array(sb['cases'][i]['serialized']['samples_us'])/1000 for i in indices]),
              ('Default whole-K','#138b87',[np.array(sa['cases'][i]['serialized']['samples_us'])/1000 for i in indices]),
              ('Forced whole-K','#d65a38',[np.array(sf['cases'][i]['serialized']['samples_us'])/1000 for i in indices])]
    draw(axes[2],[str(sa['cases'][i]['shapes'][0][0]) for i in indices],datasets)
    axes[2].set_title('FP16 linear / latency-scaling suite')
    axes[2].set_xlabel('M=N=K; FP32 accumulation')
    axes[0].set_ylabel('Completed allocating call (ms, log scale)')
    handles,labels=axes[0].get_legend_handles_labels()
    extra_handles,extra_labels=axes[2].get_legend_handles_labels()
    fig.legend(handles+[extra_handles[-1]],labels+[extra_labels[-1]],loc='lower center',ncol=4,frameon=False)
    fig.suptitle('RX 6700 XT / Vulkan: retain accumulators across the K loop',fontsize=14,y=.985)
    fig.text(.5,.91,'Tile sizes, launch and storage unchanged; identical inputs and bitwise outputs\n'
             '45 completed calls after 20 warmups; error bars = IQR; default enables eligible K ≥ 2048',
             ha='center',va='top',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.81))
    fig.savefig(args.out.with_suffix('.png'),dpi=180)
    fig.savefig(args.out.with_suffix('.svg'))


def draw(axis,labels,datasets):
    x=np.arange(len(labels))
    for i,(label,color,samples) in enumerate(datasets):
        median=np.median(samples,axis=1)
        bounds=np.percentile(samples,(25,75),axis=1)
        error=np.maximum(0,np.vstack((median-bounds[0],bounds[1]-median)))
        axis.bar(x+(i-1)*.25,median,width=.24,label=label,color=color,yerr=error,capsize=2)
    axis.set_xticks(x,labels);axis.set_yscale('log')
    axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)


if __name__=='__main__':main()
