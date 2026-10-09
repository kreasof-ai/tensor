"""Plot retained, unqualified C8 rates using complete-client and decode windows."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
h200=json.loads((ROOT/'comparison.json').read_text())['profiles']
l40s=ROOT.parent/'qwen35-native-l40s'
paths=[l40s/'completed-c8-fp8kv/report.json', l40s/'completed-c8-mtp-lookup-client/report.json']
old=[json.loads(p.read_text())['servers'][0]['points'][0]['summary']['output_tokens_per_second'] for p in paths]
new=[p['summary']['output_tokens_per_second'] for p in h200]
x=np.arange(2)
fig,axs=plt.subplots(1,2,figsize=(10.8,4.6),layout='constrained')
for axis,title,rates,labels in [
    (axs[0],'Whole client replay, including prefill',[(old,'L40S'),(new,'H200')],['Pure AR','MTP + output lookup']),
    (axs[1],'H200 client decode window: eight active',[( [p['client_eight_active_window']['output_tokens_per_second'] for p in h200],'H200')],['Pure AR','MTP + output lookup'])]:
    if len(rates)==2:
        for offset,(values,label),color in zip((-.18,.18),rates,('#6d90bd','#25a48f')):
            bars=axis.bar(x+offset,values,.35,label=label,color=color)
            axis.bar_label(bars,fmt='%.0f',padding=3)
        axis.legend(frameon=False)
    else:
        bars=axis.bar(x,rates[0][0],.55,color='#25a48f')
        axis.bar_label(bars,fmt='%.0f',padding=3)
    axis.set_xticks(x,labels)
    axis.set_title(title,fontsize=11)
    axis.set_ylabel('Aggregate output tokens / second')
    axis.set_ylim(0,max(max(values) for values,_ in rates)*1.2)
    axis.grid(axis='y',alpha=.2)
    axis.set_axisbelow(True)
    axis.spines[['top','right']].set_visible(False)
fig.suptitle('Qwen3.5-35B-A3B official FP8 weights / FP8 KV — C8, 32K in / 16K out',fontsize=12)
fig.supxlabel('Experimental: numerical qualification fails; synthetic random-token prompts; one replay/profile',fontsize=9)
fig.savefig(ROOT/'c8-throughput.svg')
