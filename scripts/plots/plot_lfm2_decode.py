"""Plot retained full-model projection/attention ablations."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(directory,out):
    with plt.rc_context({'svg.hashsalt':'tensor-lfm2-decode','font.size':10}):
        fig,axes=plt.subplots(1,3,figsize=(13,4.7))
        for axis,kind in zip(axes,('f16','q4_0','q4_k_m')):
            report=json.loads((directory/f'lfm2-{kind}-decode-optimization-benchmark.json').read_text())
            cases=report['cases'];depths=[r['prompt_tokens'] for r in cases]
            for name,label,color in (('original','Original','#8295a4'),('half2','Half2 projections','#2166ac'),('split_fp32','Split-KV attention','#d6604d'),('split_half2','Split-KV + half2','#1b7837')):
                if name not in report['plans']:continue
                centers=[];low=[];high=[]
                for row in cases:
                    seconds=np.array([s['decode_seconds'] for s in row['samples'][name]])
                    center=row['decode_tokens']/np.median(seconds);values=row['decode_tokens']/seconds
                    centers.append(center);low.append(center-values.min());high.append(values.max()-center)
                axis.errorbar(depths,centers,yerr=[low,high],label=label,color=color,marker='o',capsize=3)
            axis.set_title(kind.upper());axis.set_xscale('log',base=2);axis.set_xticks(depths,labels=depths)
            axis.set_xlabel('Prompt tokens before 256-token decode');axis.set_ylabel('Decode tokens/s (higher is better)');axis.grid(alpha=.2)
        handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.07),ncol=4)
        fig.suptitle('LFM2.5-2.6B decode optimizations · A10G')
        fig.text(.025,.025,'Same packed weights and token IDs. Resident engines run sequentially in rotating order; five repeats after warmup. Bars span repeat min/max.',fontsize=8)
        fig.tight_layout(rect=(0,.16,1,.94));out.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.svg'),metadata={'Date':None});plt.close(fig)
        svg=out.with_suffix('.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('docs/research/data'))
    p.add_argument('--out',type=Path,default=Path('docs/research/data/lfm2-decode-throughput'))
    a=p.parse_args();plot(a.data,a.out)
