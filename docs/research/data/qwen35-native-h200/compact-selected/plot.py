"""Compare complete client throughput and time decomposition from retained reports."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
r=Path(__file__).resolve().parent
paths=[r.parent.parent/'qwen35-native-l40s/completed-c8-mtp-lookup-client/report.json',
       r.parent/'tensor-h200-mtp-lookup-c8/report.json',
       r/'tensor-h200-compact-mtp-lookup-c8/report.json']
rows=[json.loads(p.read_text())['servers'][0]['points'][0]['summary'] for p in paths]
labels=['L40S','H200 initial','H200 compact']
rates=[s['output_tokens_per_second'] for s in rows]
first=[s['ttft_seconds']['mean'] for s in rows]
rest=[s['elapsed_seconds']-s['ttft_seconds']['mean'] for s in rows]
x=np.arange(3)
fig,axes=plt.subplots(1,2,figsize=(10.5,4.3),layout='constrained')
bars=axes[0].bar(x,rates,color=['#6d90bd','#25a48f','#168473'])
axes[0].bar_label(bars,fmt='%.1f',padding=3)
axes[0].set_ylabel('Aggregate output tokens / second')
axes[0].set_title('Whole HTTP client replay')
axes[0].set_ylim(0,max(rates)*1.16)
axes[1].bar(x,first,label='Mean time to first token',color='#f0ae54')
axes[1].bar(x,rest,bottom=first,label='Elapsed less mean TTFT (includes drain)',color='#6d90bd')
for i,s in enumerate(rows): axes[1].text(i,s['elapsed_seconds']+2,f"{s['elapsed_seconds']:.1f}s",ha='center')
axes[1].set_ylabel('Seconds')
axes[1].set_ylim(0,max(s['elapsed_seconds'] for s in rows)*1.2)
axes[1].set_title('Client elapsed-time decomposition')
axes[1].legend(frameon=False,fontsize=8)
for a in axes:
 a.set_xticks(x,labels);a.spines[['top','right']].set_visible(False)
 a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
fig.suptitle('Qwen3.5-35B-A3B — C8, 32K in / 16K out, MTP + output lookup',fontsize=12)
fig.supxlabel('Official FP8 weights / FP8 KV; synthetic stress; one replay/profile; model numerical qualification fails',fontsize=9)
fig.savefig(r/'phase-comparison.svg')
