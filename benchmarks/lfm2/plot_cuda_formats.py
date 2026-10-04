"""Plot the retained completed-forward three-format CUDA comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(data,out):
    plt.rcParams['svg.hashsalt']='tensor-lfm2-cuda-formats'
    fig,axes=plt.subplots(2,3,figsize=(12,6),layout='constrained')
    colors={'before':'#7a7a7a','optimized':'#2476b8','llama.cpp':'#d17a16'}
    labels={'before':'Tensor before','optimized':'Tensor optimized','llama.cpp':'llama.cpp CUDA'}
    for column,fmt in enumerate(('F16','Q4_0','Q4_K_M')):
        report=json.loads((data/f'lfm2-cuda-formats-{fmt.lower()}-benchmark.json').read_text())
        lengths=[case['prompt_tokens'] for case in report['cases']]
        for row,phase in enumerate(('prefill','decode')):
            ax=axes[row,column]
            for name in colors:
                rates=[c['runners'][name][phase+'_tokens_per_second'] for c in report['cases']]
                ax.plot(lengths,rates,marker='o',color=colors[name],label=labels[name])
            ax.set_xscale('log',base=2);ax.set_xticks(lengths,[str(n) for n in lengths])
            ax.set_ylim(bottom=0);ax.grid(alpha=.2)
            if row==0:ax.set_title(fmt)
            if column==0:ax.set_ylabel(phase.capitalize()+' tokens / second')
            if row==1:ax.set_xlabel('Prefix tokens')
    handles,legend=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,legend,loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('LFM2.5-2.6B · NVIDIA A10G · completed host-logit calls')
    fig.savefig(out,metadata={'Date':None});plt.close(fig)
    if out.suffix=='.svg':out.write_text('\n'.join(line.rstrip() for line in out.read_text().splitlines())+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('docs/research/data'))
    parser.add_argument('--out',type=Path,default=Path('docs/research/data/lfm2-cuda-formats.svg'))
    args=parser.parse_args();plot(args.data,args.out)
