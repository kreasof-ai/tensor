"""Plot retained native 32K/16K cohorts with explicitly different capacity profiles.

This analysis is separate from the matched-engine comparison plot. It reads
client stream events rather than filling gaps in metrics polling.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def render(data, out, previews):
    root=Path(data);out=Path(out);previews=Path(previews)
    out.mkdir(parents=True,exist_ok=True);previews.mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in (16,32,64):
     p=root/f'c{c}/summary.json'
     if not p.exists():continue
     v=json.loads(p.read_text())
     if v['status']!='measured-experimental':continue
     point=v['client_report']['servers'][0]['points'][0]['summary']
     rows.append((c,point))
    if len(rows)!=3:raise RuntimeError('all three measured points required for the final plot')
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    x=list(range(3));colors=['#607d8b','#1976d2','#1565c0']
    for ax,field,title,unit in ((axes[0],'output_tokens_per_second','Whole replay throughput','output tok/s'),
                                (axes[1],'ttft_seconds','Mean time to first token','seconds')):
     values=[p[field]['mean'] if field=='ttft_seconds' else p[field] for c,p in rows]
     ax.bar(x,values,color=colors,width=.6)
     ax.set_xticks(x,[f'C{c}' for c,p in rows]);ax.set_title(title);ax.set_ylabel(unit)
     ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
     for i,value in enumerate(values):ax.annotate(f'{value:,.1f}',(i,value),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
     ax.set_ylim(0,max(values)*1.22)
    axes[0].axhline(7000,color='#bc3b36',linestyle='--',linewidth=1,label='7,000 tok/s target')
    axes[0].set_ylim(0,max(7700,axes[0].get_ylim()[1]));axes[0].legend(frameon=False,fontsize=9)
    fig.suptitle('Qwen3.5-35B-A3B-FP8 · one H200 · 32K input / 16K output',fontsize=12)
    fig.supxlabel('One observation per point · distinct synthetic prompts · capacity-specific profiles · serial quality gate fails',fontsize=8)
    fig.savefig(out/'qwen35-h200-batch-scaling.svg')
    fig.savefig(previews/'batch-scaling.png',dpi=160)

    plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(10,8),layout='constrained')
    diagnostics=[]
    for c,color in ((16,'#607d8b'),(32,'#1976d2'),(64,'#dd7b28')):
     paths=list((root/f'c{c}').rglob('*requests.jsonl'))
     if not paths:raise RuntimeError('complete client records required for each point')
     rows=[json.loads(line) for line in paths[0].read_text().splitlines()]
     if len(rows)!=c or any(r['status']!='completed' for r in rows):raise RuntimeError('complete point required')
     events=sorted((event['seconds'],event['delta_tokens']) for row in rows for event in row['token_events'])
     t,deltas=np.asarray(events).T;tokens=np.cumsum(deltas)
     if tokens[-1]!=16000*c:raise RuntimeError('stream event totals must equal reported outputs')
     elapsed=json.loads((root/f'c{c}/summary.json').read_text())['client_report']['servers'][0]['points'][0]['summary']['elapsed_seconds']
     axes[0].step(np.r_[0,t,elapsed],np.r_[0,tokens,tokens[-1]]/1000,where='post',color=color,label=f'C{c}')
     ended=np.sort([row['ended_seconds'] for row in rows])
     axes[1].step(np.r_[0,ended],np.r_[c,c-np.arange(1,c+1)],where='post',color=color,label=f'C{c}')
     # Exactly counted received tokens in nonoverlapping complete 10-second bins.
     edges=np.arange(0,elapsed,10);counts,_=np.histogram(t,bins=edges,weights=deltas);rates=counts/10
     axes[2].stairs(rates,edges,color=color,label=f'C{c}')
     i=int(np.argmax(rates));peak=float(rates[i])
     diagnostics.append(dict(concurrency=c,client_interval_seconds=10,peak_output_tokens_per_second=peak,
      peak_interval=[float(edges[i]),float(edges[i+1])],observed_client_output_tokens=int(tokens[-1]),
      metric='client received tokens in complete nonoverlapping 10-second bins',full_stress_target_reached=False,
      model_throughput_qualified=False))
    axes[0].set_ylabel('Received output (K tokens)')
    axes[1].set_ylabel('Unfinished client requests')
    axes[2].set_ylabel('Received output tok/s\n(complete 10 s bins)');axes[2].set_xlabel('Seconds from client replay start')
    for ax in axes:
     ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.legend(frameon=False)
    fig.suptitle('Native H200 cohort drain · 32K input / 16K output')
    fig.supxlabel('Client stream diagnostics; primary throughput uses full elapsed. One observation per profile; serial quality gate fails.',fontsize=8)
    fig.savefig(out/'qwen35-h200-batch-drain.svg')
    fig.savefig(previews/'batch-drain.png',dpi=160)
    (root/'client-interval-rates.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))
    for name in ('qwen35-h200-batch-scaling.svg','qwen35-h200-batch-drain.svg'):
        path=out/name
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')

    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('docs/research/data/qwen35-native-h200/scaling'))
    parser.add_argument('--out',type=Path,default=Path('docs/research'))
    parser.add_argument('--previews',type=Path,default=Path('build/qwen35-h200-akbar-scaling'))
    args=parser.parse_args();render(args.data,args.out,args.previews)
