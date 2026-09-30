"""Plot retained complete-update timings; error bars span the five windows."""
from pathlib import Path
import argparse
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(source, destination):
    report=json.loads(Path(source).read_text())
    labels=['Tensor','Torch eager / explicit','Torch compiled / explicit',
            'Torch eager / SDPA','Torch compiled / SDPA',
            'Torch eager / SDPA + fused AdamW','Torch compiled / SDPA + fused AdamW']
    records=list(report['providers'].values())
    medians=[r['median_seconds_per_update']*1000 for r in records]
    low=[m-min(r['window_means_seconds'])*1000 for m,r in zip(medians,records)]
    high=[max(r['window_means_seconds'])*1000-m for m,r in zip(medians,records)]
    with plt.rc_context({'svg.hashsalt':'tensor-phase6','font.size':10}):
        fig,ax=plt.subplots(figsize=(10,5.2))
        ax.barh(labels,medians,xerr=[low,high],capsize=3,
                color=['#2166ac']+['#8295a4']*6)
        ax.invert_yaxis();ax.set_xlim(0,max(medians)*1.13)
        for i,value in enumerate(medians):ax.text(value+0.8,i,f'{value:.2f} ms',va='center')
        ax.set_xlabel('Milliseconds per complete training update (lower is better)')
        ax.set_title('nanoGPT: 12 layers, width 768, batch 2 × sequence 512 · A10G')
        ax.grid(axis='x',alpha=0.2);ax.set_axisbelow(True)
        fig.text(0.02,0.015,'Five ten-update windows, each starting from initial weights/state. Median latency; error bars span window min/max.',fontsize=8)
        fig.tight_layout(rect=(0,0.04,1,1))
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        fig.savefig(destination.with_suffix('.png'),dpi=180)
        fig.savefig(destination.with_suffix('.svg'),metadata={'Date':None})
        path=destination.with_suffix('.svg')
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
        plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('docs/research/data/phase6-nanogpt-benchmark.json'))
    parser.add_argument('--out',type=Path,default=Path('docs/research/data/phase6-nanogpt-latency'))
    args=parser.parse_args();plot(args.source,args.out)
