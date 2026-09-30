"""Plot the retained matched full-model API throughput and repeat ranges."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(directory,destination):
    kinds=('f16','q4_0','q4_k_m')
    with plt.rc_context({'svg.hashsalt':'tensor-lfm2','font.size':10}):
        fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
        for column,kind in enumerate(kinds):
            report=json.loads((directory/f'lfm2-{kind}-benchmark.json').read_text())
            rows=report['cases'];depths=[row['prompt_tokens'] for row in rows]
            for axis,phase in zip(axes[:,column],('prefill','decode')):
                for engine,key,color in (('Tensor','samples','#2166ac'),('llama.cpp CUDA','llama_samples','#d6604d')):
                    values=[];low=[];high=[]
                    for row in rows:
                        count=row['prompt_tokens'] if phase=='prefill' else row['decode_tokens']
                        observations=np.array([count/sample[f'{phase}_seconds'] for sample in row[key]])
                        center=count/np.median([sample[f'{phase}_seconds'] for sample in row[key]])
                        values.append(center);low.append(center-observations.min());high.append(observations.max()-center)
                    axis.errorbar(depths,values,yerr=[low,high],label=engine,color=color,marker='o',capsize=3)
                axis.set_xscale('log',base=2);axis.set_xticks(depths,labels=depths);axis.grid(alpha=.2)
                axis.set_ylabel(f'{phase.capitalize()} tokens/s (higher is better)')
            axes[0,column].set_title(kind.upper());axes[1,column].set_xlabel('Prompt tokens before 256-token decode')
        axes[0,0].legend()
        fig.suptitle('LFM2.5-2.6B · one sequence · NVIDIA A10G',fontsize=14)
        fig.text(.02,.015,'Matched token IDs and GGUF files; host-visible logits, prefill chunks ≤128. Five repeats after one warmup; bars span repeat min/max.',fontsize=9)
        fig.tight_layout(rect=(0,.05,1,.96));destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination.with_suffix('.png'),dpi=180)
        fig.savefig(destination.with_suffix('.svg'),metadata={'Date':None});plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('docs/research/data'))
    p.add_argument('--out',type=Path,default=Path('docs/research/data/lfm2-throughput'))
    a=p.parse_args();plot(a.data,a.out)
