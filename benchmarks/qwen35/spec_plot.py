"""Plot retained full-client throughput and the native decode/drain trace."""
import argparse
import json
from pathlib import Path
import numpy as np


def plot(root,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=Path(out)
    ar=json.loads((root/'completed-c8-fp8kv/report.json').read_text())
    client=json.loads((root/'completed-c8-mtp-lookup-client/summary.json').read_text())
    native=json.loads((root/'completed-c8-mtp-lookup-k7-native/report.json').read_text())
    rows=[json.loads(line) for line in (root/'completed-c8-mtp-lookup-k7-native/rounds.jsonl').read_text().splitlines()]
    def summaries(value):
        for server in value['servers']:
            for point in server['points']:yield point['summary']
    ar_rate=next(summaries(ar))['output_tokens_per_second']
    rates=[ar_rate,client['client']['output_tokens_per_second']]
    fig,(left,right)=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    colors=['#7b8794','#1565c0'];left.bar(['AR','MTP + output lookup'],rates,color=colors,width=.58)
    left.axhline(600,color='#b45309',linestyle='--',linewidth=1,label='600 tok/s speed milestone')
    for i,rate in enumerate(rates):left.text(i,rate+15,f'{rate:.1f}',ha='center',fontsize=11)
    left.set(ylim=(0,900),ylabel='Aggregate output tokens / second',title='Client: entire 32K / 16K replay')
    left.legend(frameon=False,fontsize=9)
    times=np.array([r['ended_seconds'] for r in rows]);counts=np.cumsum([r['output_tokens'] for r in rows])
    window=128;speed=(counts[window:]-counts[:-window])/(times[window:]-times[:-window])
    right.plot(times[window:],speed,color='#1565c0',linewidth=1.5)
    boundary=native['all_requests_active_decode_seconds']
    right.axvspan(boundary,times[-1],color='#e5e7eb',alpha=.7,label='Finishing remaining requests')
    right.set(xlabel='Seconds after prefix processing',ylabel='Aggregate output tokens / second',
              title='Native: rolling 128 verification rounds',ylim=(0,max(speed)*1.15))
    right.legend(frameon=False,fontsize=9)
    for axis in (left,right):
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.15);axis.set_axisbelow(True)
    fig.suptitle('Qwen3.5-35B-A3B · one L40S · C8 · official FP8 weights / FP8 KV',fontsize=12)
    fig.supxlabel('Experimental performance measurements; model numerical qualification failed. Synthetic forced-length workload.',fontsize=9)
    out.parent.mkdir(parents=True,exist_ok=True);fig.savefig(out);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();plot(a.data,a.out)
