"""Plot the retained alternating FP32/half2 full-model decode experiment."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(directory,out):
    with plt.rc_context({'svg.hashsalt':'tensor-lfm2-fp16','font.size':10}):
        fig,axes=plt.subplots(1,3,figsize=(12,4.3))
        for axis,kind in zip(axes,('f16','q4_0','q4_k_m')):
            report=json.loads((directory/f'lfm2-{kind}-half2-paired.json').read_text())
            cases=report['cases'];depths=[r['prompt_tokens'] for r in cases]
            for name,label,color in (('fp32','Current FP32','#8295a4'),('fp16_half2','Experimental half2','#2166ac')):
                centers=[];low=[];high=[]
                for row in cases:
                    seconds=np.array([s['decode_seconds'] for s in row['samples'][name]])
                    center=row['decode_tokens']/np.median(seconds);values=row['decode_tokens']/seconds
                    centers.append(center);low.append(center-values.min());high.append(values.max()-center)
                axis.errorbar(depths,centers,yerr=[low,high],label=label,color=color,marker='o',capsize=3)
            axis.set_title(kind.upper());axis.set_xscale('log',base=2);axis.set_xticks(depths,labels=depths)
            axis.set_xlabel('Prompt tokens before 256-token decode');axis.set_ylabel('Decode tokens/s (higher is better)');axis.grid(alpha=.2)
        axes[0].legend();fig.suptitle('Packed-weight FP16 decode experiment · LFM2.5-2.6B · A10G')
        fig.text(.02,.025,'Same packed weights and token IDs. Alternating engine order; five repeats after warmup; host-visible logits. Bars span repeat min/max.',fontsize=8)
        fig.tight_layout(rect=(0,.07,1,.94));out.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.svg'),metadata={'Date':None});plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,default=Path('docs/research/data'))
    p.add_argument('--out',type=Path,default=Path('docs/research/data/lfm2-fp16-throughput'))
    a=p.parse_args();plot(a.data,a.out)
