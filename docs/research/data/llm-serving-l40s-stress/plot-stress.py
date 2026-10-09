"""Reproduce the bounded long-context figure from a retained combined report."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('report',type=Path)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
report=json.loads(args.report.read_text())
args.out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
metrics=[('output_tokens_per_second',None,'Output throughput (tokens/s)',1),('ttft_seconds','mean','Mean time to first token (s)',1),('tpot_seconds','mean','Mean time per output token (ms)',1000),('latency_seconds','mean','Mean request completion time (s)',1)]
labels={'vllm-fp8':'vLLM 0.17.1 (FP8)','sglang-fp8':'SGLang 0.5.9 (FP8)','llama-gguf-q8-fp8-source':'llama.cpp b11429 (Q8 from FP8; quality unqualified)'}
for ax,(metric,stat,ylabel,scale) in zip(axes.flat,metrics):
 for server in report['servers']:
  points=sorted((p for p in server['points'] if p['disposition']=='measured'),key=lambda p:p['concurrency'])
  if not points:continue
  config=server['configuration'];name=config['name']
  y=[p['summary'][metric][stat]*scale if stat else p['summary'][metric]*scale for p in points]
  ax.plot([p['concurrency'] for p in points],y,marker='o',linestyle='--' if config['engine']=='llama.cpp' else '-',label=labels.get(name,name))
 ax.set_xscale('log',base=2);ax.set_xticks([1,2,4],['1','2','4']);ax.set_xlabel('Client concurrency')
 ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.grid(alpha=.25)
fig.suptitle('Qwen3.5-35B-A3B on one NVIDIA L40S | 32,000 input / 16,000 output tokens\n4 requests per point; finite replay including fill/drain and lazy setup; one repetition',fontsize=12)
handles,labels_=axes.flat[0].get_legend_handles_labels()
fig.legend(handles,labels_,loc='outside lower center',frameon=False,fontsize=9)
for ext in ('png','svg'):fig.savefig(args.out/f'stress-overview.{ext}',dpi=180)
plt.close(fig)
