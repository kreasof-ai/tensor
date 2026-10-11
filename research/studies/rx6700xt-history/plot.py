"""Derive an auditable historical record and static plots from retained GPU evidence.

No GPU work or historical-source mutation. Each plotted median is checked against
its original samples when available; different operations/contracts stay separate.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

NATIVE='f872b591121761ac7b2af18283bd99bdc092a63a'
CUTOFF='d5ee5087f94fa12e3002146730f0354632749fb5'
ROOT=Path(__file__).resolve().parents[2]
HISTORICAL_SNAPSHOTS={
    'lfm2-2.6b-q4_0-initial-vulkan':(
        '5bd34b97e54c6c462e87b48cb2d01231370b6bd9','lfm2-2.6b-q4_0-matched-run')}
SHA_F16='4d364976c7ae1b85bd380f743155aa2d532f7a10291beaa6b27a7d6c9b10527f'
SHA_Q4='430fbec5b1b355e9bb12cd0638c9f2a8f21fedd6eafb4103e42c7e88887daa73'
SHA_BIG='a247afd6414918eac8e520a9e6137dc271235461ecbe1180462221d5b8d40b03'
BLUE='#2563a5'; TEAL='#13877d'; ORANGE='#bf661e'; GRAY='#727a85'; INK='#202d3b'


class Evidence:
    def __init__(self,directory):self.directory=directory;self.cache={};self.index={}
    def read(self,name,pointer=''):
        if name not in self.cache:
            revision,original_name=HISTORICAL_SNAPSHOTS.get(name,(CUTOFF,name))
            repository_path='docs/research/data/'+original_name+'.json'
            blob=subprocess.check_output(['git','show',revision+':'+repository_path],cwd=ROOT)
            path=self.directory/(name+'.json')
            if name in HISTORICAL_SNAPSHOTS:path.write_bytes(blob)
            raw=path.read_bytes()
            self.cache[name]=json.loads(raw)
            self.index[name+'.json']={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            # Git's text normalization can change CRLF to LF without changing evidence.
            assert json.loads(blob)==self.cache[name],f'{name}: differs from pinned archive'
            commit,date=subprocess.check_output(['git','log','-1','--format=%H%n%cI',
                revision,'--',repository_path],cwd=ROOT,text=True).strip().splitlines()
            self.index[name+'.json'].update({'git_blob_sha256':hashlib.sha256(blob).hexdigest(),
                'snapshot_commit':commit,'snapshot_committed_at':date})
            if name in HISTORICAL_SNAPSHOTS:
                self.index[name+'.json']['recovered_from_git']=revision+':'+repository_path
        value=self.cache[name]
        for part in pointer.strip('/').split('/') if pointer else []:
            part=part.replace('~1','/').replace('~0','~')
            value=value[int(part)] if isinstance(value,list) else value[part]
        return value


def stats(samples,center):
    if not samples:return {'median':center,'min':None,'max':None,'samples':None}
    assert all(math.isfinite(v) and v>0 for v in samples)
    assert math.isclose(statistics.median(samples),center,rel_tol=1e-9,abs_tol=1e-9)
    return {'median':center,'min':min(samples),'max':max(samples),'samples':samples}


def inference(evidence,code,label,name,pointer,runner,native,sha,chunks,contract='F16 operands / F32 totals'):
    report=evidence.read(name,pointer)
    assert report['status']=='passed' and report['model_sha256']==sha
    protocol=report['protocol']
    assert protocol['context']==512 and len(report['validation'])==19
    commit=report.get('native_commit',report.get('llama_cpp',{}).get('commit'))
    assert commit==NATIVE
    quality=[row['numpy'] if runner=='tensor' and 'numpy' in row else row[runner]
             for row in report['validation']]
    assert all(math.isfinite(row['relative_rms']) and math.isfinite(row['cosine'])
               and row['relative_rms']<.01 and row['cosine']>.9999 and row['argmax'][0]==row['argmax'][1]
               for row in quality)
    result={'id':code,'label':label,'source':name+'.json','report_pointer':pointer,
            'runner':runner,'native_runner':native,'model_sha256':sha,'chunks':chunks,
            'arithmetic':contract,'protocol':protocol,'measurement_revision':report.get('source_commit'),
            'archive_base_revision':report.get('repository_head',evidence.read(name).get('repository_head')),
            'snapshot_commit':evidence.index[name+'.json']['snapshot_commit'],
            'accuracy':{'fixtures':19,'max_relative_rms':max(row['relative_rms'] for row in quality),
                        'min_cosine':min(row['cosine'] for row in quality)},'measurements':[]}
    for i,row in enumerate(report['benchmarks']):
        observations=row['samples'];sample_count=len(observations[runner])
        assert sample_count==protocol['repeats'] and row['decode_tokens']==64
        entry={'prompt_tokens':row['prompt_tokens'],'decode_tokens':row['decode_tokens'],
               'pointer':pointer+f'/benchmarks/{i}','samples_per_runner':sample_count}
        for phase in ('prefill','decode'):
            count=row['prompt_tokens'] if phase=='prefill' else row['decode_tokens']
            for who,key in ((runner,'tensor'),(native,'native')):
                seconds=[v[phase+'_seconds'] for v in observations[who]]
                center=count/row[who][phase+'_tokens_per_second']
                entry[key+'_'+phase]={'tokens_per_second':count/center,
                    'latency_seconds':stats(seconds,center),
                    'min_tokens_per_second':count/max(seconds),'max_tokens_per_second':count/min(seconds)}
        result['measurements'].append(entry)
    assert {r['prompt_tokens'] for r in result['measurements']}=={32,128,384}
    return result


def one(stage,prompt=128):return next(r for r in stage['measurements'] if r['prompt_tokens']==prompt)


def histories(e):
    f16=[];q4=[]
    pairs=[('H0','First Vulkan','vulkan'),('H1','Register tiles','optimized'),
           ('H2','Microtiles + subgroups','compiler'),('H3','Native encoding','native')]
    for code,label,suffix in pairs:
        for kind,sha,destination in (('f16',SHA_F16,f16),('q4_0',SHA_Q4,q4)):
            name=(f'lfm2-230m-{kind}-{suffix}' if suffix in ('vulkan','optimized') else
                  f'lfm2-230m-{suffix}-{kind}-after')
            destination.append(inference(e,code,label,name,'','tensor','llama_cpp',sha,[32]))
    for kind,sha,destination in (('f16',SHA_F16,f16),('230m',SHA_Q4,q4)):
        destination.append(inference(e,'H4','Q4 dots / residual fusion','lfm2-webgpu-decode-push',
                                     f'/matched/{kind}/after','tensor','llama_cpp',sha,[32]))
    for code,label,name,pointer,runner,native,chunks in [
        ('H5','Prefill search','lfm2-prefill-search-throughput','/full_model','tensor','llama_cpp',[32]),
        ('H6','Decode search','lfm2-230m-decode-search','/full_model','tensor_searched','llama_cpp',[32]),
        ('H7','Ordered readback','lfm2-230m-runtime-search','/full_model','tensor_runtime','llama_cpp',[32]),
        ('H8','Adaptive prefill','lfm2-prefill-chase','/adaptive_comparison','adaptive','llama128',[32,128])]:
        f16.append(inference(e,code,label,name,pointer,runner,native,SHA_F16,chunks))
    big=[]
    for code,label,name,pointer,runner,native,chunks,contract in [
        ('B0','First matched run','lfm2-2.6b-q4_0-initial-vulkan','','tensor','llama_cpp',[32],'F16 operands / F32 totals'),
        ('B1','RMS / staging corrections','lfm2-2.6b-q4_0-matched-run','','tensor','llama_cpp',[32],'F16 operands / F32 totals'),
        ('B2','Packed dots + residual','lfm2-webgpu-decode-push','/matched/2.6b/after','tensor','llama_cpp',[32],'F16 operands / F32 totals'),
        ('B3','Readback + larger chunks','lfm2-2.6b-q4_0-revisit','/comparison','adaptive','llama128',[32,128],'F16 operands / F32 totals'),
        ('B4','Packed schedule search','lfm2-2.6b-q4_0-parity','/comparison','adaptive','llama128',[32,128],'F16 operands / F32 totals'),
        ('B5','Two-component Q8 pilot','lfm2-2.6b-prefill-1k','/pilot','q16','llama128',[32,128],'Two Q8 activation components; F32 weight scales/totals'),
        ('B6','Short F16 chains','lfm2-2.6b-prefill-1k','/intermediate_comparisons/mixed_before_liveness','mixed','llama128',[32,128],'Mixed short F16 chains + two-component Q8'),
        ('B7','Suffix + query liveness','lfm2-2.6b-prefill-1k','/intermediate_comparisons/query_tail_before_layout','mixed','llama128',[32,128],'Mixed short F16 chains + two-component Q8'),
        ('B8','Padded shared tiles','lfm2-2.6b-prefill-1k','/comparison','mixed','llama128',[32,128],'Mixed short F16 chains + two-component Q8')]:
        big.append(inference(e,code,label,name,pointer,runner,native,SHA_BIG,chunks,contract))
    return {'230m_f16':f16,'230m_q4_0':q4,'2.6b_qad_q4_0':big}


def kernels(e):
    scaling=[]
    for label,name in [('First scaling run','webgpu-rx6700xt-scaling'),
        ('2x2 microtiles','webgpu-rx6700xt-scaling-optimization-after'),
        ('Whole-K accumulators','webgpu-gemm-accumulation-scaling-default')]:
        report=e.read(name);assert report['status']=='passed'
        i,row=next((i,r) for i,r in enumerate(report['cases']) if r['name']=='gemm-4096-4096-4096')
        assert row['dtype']=='float16' and row['status']=='passed'
        timing=row['serialized']
        scaling.append({'label':label,'source':name+'.json','pointer':f'/cases/{i}',
            'timestamp':report['timestamp'],'input_sha256':row['inputs_sha256'],
            'output_sha256':row['output_sha256'],'wgsl_sha256':row['wgsl_sha256'],
            'milliseconds':stats([v/1000 for v in timing['samples_us']],timing['median_us']/1000)})
    assert all(r['input_sha256']==scaling[0]['input_sha256'] and r['output_sha256']==scaling[0]['output_sha256'] for r in scaling)
    name='webgpu-gemm-accumulation-default-comparison';report=e.read(name)
    i,row=next((i,r) for i,r in enumerate(report['cases']) if r['name']=='float32-4096-4096-4096-gemm')
    assert row['same_output'] and row['precision_matched']
    timings=row['timings']['preallocated'];fp32=[]
    for label,key in [('Shared fragment','before_ms'),('Whole-K private','after_ms')]:
        fp32.append({'label':label,'source':name+'.json','pointer':f'/cases/{i}/timings/preallocated/{key}',
                     'milliseconds':stats([],timings[key])})
    name='webgpu-outer-product-comparison';report=e.read(name)
    i,row=next((i,r) for i,r in enumerate(report['cases']) if r['name']=='float32-4096-4096-4096-gemm')
    for label,key,mode in [('Fresh generic','tensor_generic','preallocated'),
                          ('Outer product','tensor_outer','prepared_completed'),('CLBlast OpenCL','clblast','preallocated')]:
        assert row[key]['correctness']['passed']
        timing=row[key][mode]
        fp32.append({'label':label,'source':name+'.json','pointer':f'/cases/{i}/{key}/{mode}',
                     'milliseconds':stats(timing['samples_ms'],timing['median_ms'])})
    name='lfm2-2.6b-prefill-1k';pointer='/searches/half-layout-replay/report/groups/0'
    group=e.read(name,pointer);assert group['kind']=='ffn' and group['rows']==128 and group['k']==2048 and group['o']==10752
    half=[]
    for label,record in [('Previous short-half tile',next(r for r in group['records'] if r['label']=='configured-short')),
                         ('Padded + row-first',group['best'])]:
        assert record['status']=='passed'
        half.append({'label':label,'source':name+'.json','group_pointer':pointer,
            'artifact':record['artifact'],'parameters':record['parameters'],
            'milliseconds':stats([v*1000 for v in record['replay_samples_seconds']],record['replay_median_seconds']*1000)})
    return {'fp16_linear_allocating':scaling,'fp32_gemm_preallocated':fp32,'packed_ffn_layout_replay':half}


def style(ax):
    ax.set_axisbelow(True);ax.grid(axis='y',color='#e7eaf0',linewidth=.8)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#b8c0c9');ax.tick_params(colors=INK)


def save(fig,out,name):
    fig.savefig(out/(name+'.png'),dpi=180,facecolor='white')
    fig.savefig(out/(name+'.svg'),metadata={'Date':None},facecolor='white');plt.close(fig)


def bars(ax,rows,positions,colors):
    for row,x,color in zip(rows,positions,colors):
        t=row['milliseconds'];v=t['median'];ax.bar(x,v,color=color,width=.66,zorder=3)
        if t['min'] is not None:ax.errorbar(x,v,yerr=[[v-t['min']],[t['max']-v]],color=INK,linewidth=.8,capsize=3,zorder=4)
        ax.text(x,(t['max'] if t['max'] is not None else v)+v*.07,f'{v:.2f}',ha='center',va='bottom',fontsize=10,color=INK)
    ax.set_ylim(bottom=0);style(ax)


def plot_kernels(data,out):
    fig,axes=plt.subplots(1,3,figsize=(15,5.7),gridspec_kw={'width_ratios':[1.0,1.65,1.0]})
    rows=data['kernels']['fp16_linear_allocating']
    bars(axes[0],rows,range(3),[GRAY,BLUE,TEAL]);axes[0].set_xticks(range(3),['First\nscaling','2x2\nmicrotiles','Whole K\nprivate'])
    axes[0].set_title('A  |  FP16 linear 4096³',loc='left',fontweight='bold');axes[0].set_ylabel('Completed allocating call (ms)')
    axes[0].set_ylim(0,280)
    rows=data['kernels']['fp32_gemm_preallocated'];x=[0,1,3,4,5]
    bars(axes[1],rows,x,[GRAY,BLUE,GRAY,TEAL,'#9aa1aa']);axes[1].set_xticks(x,['Shared\nfragment','Whole K\nprivate','Fresh\ngeneric','Outer\nproduct','CLBlast\nOpenCL'])
    axes[1].set_title('B  |  FP32 pure GEMM 4096³',loc='left',fontweight='bold');axes[1].set_ylabel('Completed preallocated call (ms)')
    axes[1].axvline(2,color='#c6ccd4',linestyle=':',linewidth=1)
    axes[1].text(.5,240,'Oct 2: matched 1.75×',ha='center',fontsize=9,color=INK)
    axes[1].text(4,240,'Oct 3: matched 3.37×',ha='center',fontsize=9,color=INK);axes[1].set_ylim(0,270)
    rows=data['kernels']['packed_ffn_layout_replay'];bars(axes[2],rows,[0,1],[ORANGE,TEAL])
    axes[2].set_xticks([0,1],['Previous\nshort F16','Padded +\nrow-first'])
    axes[2].set_title('C  |  Packed 2.6B FFN',loc='left',fontweight='bold');axes[2].set_ylabel('GPU ms per paired gate/up/SwiGLU')
    axes[2].set_ylim(0,2.6);axes[2].text(.5,2.32,'12.4% less GPU time',ha='center',fontsize=10,color=INK)
    fig.suptitle('RX 6700 XT: compiler kernel improvements',x=.055,ha='left',fontsize=17,fontweight='bold',color=INK)
    fig.text(.055,.89,'A: same allocating workload across milestones.  B: independent matched pairs.  C: identical short-half arithmetic.',fontsize=10,color=GRAY)
    fig.text(.055,.025,'Whiskers show archived sample min/max, where available. Panels have different workloads and timing boundaries.\nB: outer product uses a prepared plan; fresh generic prepared repeat was 113.05 ms. CLBlast is an OpenCL reference. C: 128 rows, K=2048, N=10752; all 30 matrix pairs streamed.',fontsize=8.5,color=GRAY)
    fig.subplots_adjust(left=.065,right=.985,bottom=.21,top=.78,wspace=.4)
    save(fig,out,'rx6700xt-vulkan-history-kernels')


def curve(ax,stages,phase,key,color,label,prompt=128,native=False,native_style='--'):
    x=[int(s['id'][1:]) for s in stages];values=[];low=[];high=[]
    for stage in stages:
        t=one(stage,prompt)[key+'_'+phase];v=t['tokens_per_second'];values.append(v)
        low.append(v-t['min_tokens_per_second']);high.append(t['max_tokens_per_second']-v)
    ax.errorbar(x,values,yerr=[low,high],color=color,linestyle=native_style if native else '-',
                marker='o',markersize=4,linewidth=1.7,capsize=2,label=label,zorder=4)
    return values


def plot_230m(data,out):
    fig,axes=plt.subplots(2,2,figsize=(14.5,8.7))
    for col,(key,title) in enumerate([('230m_f16','Native F16 checkpoint'),('230m_q4_0','Q4_0 checkpoint')]):
        stages=data['inference'][key]
        for row,phase in enumerate(('prefill','decode')):
            ax=axes[row,col];curve(ax,stages,phase,'tensor',BLUE,'Tensor Vulkan')
            # Native chunk configuration changes at H8: do not join across it.
            curve(ax,[s for s in stages if s['id']!='H8'],phase,'native',GRAY,'llama.cpp Vulkan',native=True)
            if key=='230m_f16':
                s=stages[-1];t=one(s)['native_'+phase];v=t['tokens_per_second']
                ax.errorbar([8],[v],yerr=[[v-t['min_tokens_per_second']],
                    [t['max_tokens_per_second']-v]],color=GRAY,marker='D',
                    linestyle='none',markersize=5,capsize=2,zorder=5)
                ax.axvspan(7.5,8.45,color=TEAL,alpha=.07)
                end=one(s)['tensor_'+phase]['tokens_per_second']
                ax.annotate(f'{end:,.0f}',(8,end),xytext=(-12,10),textcoords='offset points',ha='right',color=BLUE,fontweight='bold')
            else:
                ax.text(6.55,.5,'Later searches\nmeasured F16 only',transform=ax.get_xaxis_transform(),ha='center',va='center',fontsize=10,color=GRAY)
            style(ax);ax.set_xlim(-.3,8.5);ax.set_ylim(bottom=0)
            ax.set_ylabel(phase.capitalize()+' tokens/s')
            ax.set_xticks(range(9),[f'H{i}' for i in range(9)]);ax.set_xlabel('Milestone')
            if row==0:ax.set_title(title,loc='left',fontsize=12,fontweight='bold')
    axes[0,0].legend(frameon=False,loc='upper left')
    fig.suptitle('LFM2.5-230M: from the first Vulkan forward to searched inference',x=.065,ha='left',fontsize=16,fontweight='bold',color=INK)
    fig.text(.065,.91,'128-token prompt / prefix; 64 forced decode tokens; completed host F32 logits. Each encoding has its own immutable checkpoint.',fontsize=10,color=GRAY)
    fig.text(.065,.045,'H0 First Vulkan   ·   H1 Register tiles   ·   H2 Microtiles/subgroups   ·   H3 Native encoding   ·   H4 Q4 dots/residual\nH5 Prefill search   ·   H6 Decode search   ·   H7 Ordered readback   ·   H8 Adaptive 32/128-row prefill\nBars span sample min/max, not confidence intervals. H0–H4: five samples; H5–H8: seven. Native uses chunk 32 before H8; the separate diamond uses chunk 128.',fontsize=9,color=GRAY)
    fig.subplots_adjust(left=.07,right=.975,bottom=.22,top=.84,hspace=.45,wspace=.24)
    save(fig,out,'rx6700xt-vulkan-history-230m')


def plot_big(data,out):
    stages=data['inference']['2.6b_qad_q4_0'];fig,axes=plt.subplots(1,2,figsize=(14.5,6.5))
    for ax,phase in zip(axes,('prefill','decode')):
        ax.axvspan(4.5,8.45,color=ORANGE,alpha=.075)
        for prompt,color in ((128,BLUE),(384,TEAL)):
            curve(ax,stages,phase,'tensor',color,f'Tensor, {prompt} tokens',prompt)
            # Match native chunk 32 for B0–B2, chunk 128 from B3 onward.
            native_style='--' if prompt==128 else ':'
            curve(ax,stages[:3],phase,'native',GRAY,'llama.cpp, 128' if prompt==128 else 'llama.cpp, 384',prompt,True,native_style)
            curve(ax,stages[3:],phase,'native',GRAY,'_nolegend_',prompt,True,native_style)
        style(ax);ax.set_xlim(-.3,8.5);ax.set_ylim(bottom=0)
        ax.set_xticks(range(9),[f'B{i}' for i in range(9)]);ax.set_xlabel('Milestone')
        ax.set_title(phase.capitalize(),loc='left',fontsize=13,fontweight='bold');ax.set_ylabel('Tokens/s')
        if phase=='prefill':
            ax.axhline(1000,color=ORANGE,linestyle=':',linewidth=1)
            ax.text(.05,1015,'1K target',color=ORANGE,fontsize=9)
            ax.annotate('1,058 / 1,047',(8,1058),xytext=(5.8,1430),arrowprops={'arrowstyle':'-','color':BLUE},fontsize=11,color=BLUE,fontweight='bold')
        else:ax.set_ylim(0,195);ax.text(4.8,182,'Decode remains ~165',fontsize=10,color=BLUE,ha='center')
    fig.legend(handles=[Line2D([],[],color=BLUE,marker='o',label='Tensor: 128 tokens'),
        Line2D([],[],color=TEAL,marker='o',label='Tensor: 384 tokens'),
        Line2D([],[],color=GRAY,linestyle='--',marker='o',label='llama.cpp: 128 tokens'),
        Line2D([],[],color=GRAY,linestyle=':',marker='o',label='llama.cpp: 384 tokens')],
        frameon=False,fontsize=9,loc='upper left',bbox_to_anchor=(.06,.825),ncol=4)
    fig.suptitle('LFM2.5-2.6B QAD Q4_0: crossing 1K prefill on Vulkan',x=.065,ha='left',fontsize=17,fontweight='bold',color=INK)
    fig.text(.065,.87,'Same GGUF and context 512. Shaded region introduces approximate prefill math; independent model gates remain unchanged.',fontsize=10,color=GRAY)
    fig.text(.065,.05,'B0 First matched run   ·   B1 RMS/staging corrections   ·   B2 Packed dots/residual   ·   B3 Readback + larger chunks   ·   B4 Packed search\nB5 Two-component Q8 pilot (3 samples)   ·   B6 Short F16 chains   ·   B7 Suffix/query liveness   ·   B8 Padded shared tiles\nB0–B2: five samples; B3/B4/B6–B8: seven. Whiskers span sample min/max. Native chunks change from 32 to 128 at B3; no line joins that boundary.',fontsize=9,color=GRAY)
    fig.subplots_adjust(left=.065,right=.975,bottom=.26,top=.73,wspace=.24)
    save(fig,out,'rx6700xt-vulkan-history-2.6b')


def table(stages):
    lines=['| Stage | Change | Prefill 128 | Prefill 384 | Decode at 128 | Native prefill / decode at 128 | Samples |',
           '|---|---|---:|---:|---:|---:|---:|']
    for s in stages:
        a,b=one(s),one(s,384)
        link=f"[{s['id']}](data/{s['source']})"
        lines.append(f"| {link} | {s['label']} | {a['tensor_prefill']['tokens_per_second']:,.1f} | {b['tensor_prefill']['tokens_per_second']:,.1f} | {a['tensor_decode']['tokens_per_second']:,.1f} | {a['native_prefill']['tokens_per_second']:,.1f} / {a['native_decode']['tokens_per_second']:,.1f} | {a['samples_per_runner']} |")
    return '\n'.join(lines)


def archive_timeline(data):
    lines=['| Archive time (Asia/Jakarta) | Retained snapshot | Evidence recorded |',
           '|---|---|---|']
    milestones=[
        ('webgpu-rx6700xt-vulkan','[First hardware acceptance](webgpu-rx6700xt.md) and initial generic scaling'),
        ('lfm2-230m-f16-vulkan','[230M initial implementation and optimizations](lfm2-230m-vulkan.md): H0–H3, F16 and Q4'),
        ('webgpu-rx6700xt-scaling-optimization-after','[Generic microtile scaling repeat](latency-scaling.md#rx-6700-xt-repeat-after-compiler-optimization)'),
        ('webgpu-gemm-accumulation-scaling-default','[Whole-loop accumulator lowering](webgpu-gemm-accumulation.md)'),
        ('lfm2-2.6b-q4_0-initial-vulkan','[First 2.6B snapshot](data/lfm2-2.6b-q4_0-initial-vulkan.json), recovered verbatim from Git: B0'),
        ('lfm2-2.6b-q4_0-matched-run','[Corrected 2.6B matched run](lfm2-2.6b-q4_0-matched-run.md): B1'),
        ('lfm2-webgpu-decode-push','[Packed decode/residual improvements](lfm2-webgpu-decode-push.md): H4 and B2'),
        ('lfm2-230m-runtime-search','[Producer search and runtime transfer](lfm2-230m-runtime-search.md): H5–H7'),
        ('webgpu-outer-product-comparison','[Outer-product GEMM](webgpu-outer-product-gemm.md) and adaptive 230M prefill: H8'),
        ('lfm2-2.6b-prefill-1k','[2.6B revisit through 1K prefill](lfm2-2.6b-prefill-1k.md): B3–B8')]
    for name,description in milestones:
        source=data['sources'][name+'.json']
        time=datetime.fromisoformat(source['snapshot_committed_at']).astimezone(timezone(timedelta(hours=7)))
        lines.append(f"| {time:%Y-%m-%d %H:%M} | `{source['snapshot_commit'][:7]}` | {description} |")
    return '\n'.join(lines)


def document(data,path):
    h=data['inference'];first,last=one(h['230m_f16'][0]),one(h['230m_f16'][-1]);b0,b7=one(h['2.6b_qad_q4_0'][0]),one(h['2.6b_qad_q4_0'][-1])
    text=f'''# RX 6700 XT Vulkan: a history of Tensor kernel and inference improvements

[Research index](README.md) · [Derived measurements and source hashes](data/rx6700xt-vulkan-history.json) ·
[Milestone CSV](data/rx6700xt-vulkan-history.csv) · [Plot/extraction script](../../scripts/plots/plot_rx6700xt_vulkan_history.py)

This follows the recorded campaign from the first physical-GPU validation on
October 1, 2026 through the 2.6B 1K-prefill result measured on October 3 and
committed as `d5ee508`. It brings the kernel, runtime and model experiments into
one history. The measurements are retained observations; this document does
not rerun historical revisions on a new driver.

At a 128-token prompt, native-F16 230M prefill moves from **{first['tensor_prefill']['tokens_per_second']:,.1f} to {last['tensor_prefill']['tokens_per_second']:,.1f} tokens/s**;
decode moves from **{first['tensor_decode']['tokens_per_second']:.1f} to {last['tensor_decode']['tokens_per_second']:.1f}**.
The first archived 2.6B QAD Q4_0 matched run records **{b0['tensor_prefill']['tokens_per_second']:.1f} prefill / {b0['tensor_decode']['tokens_per_second']:.1f} decode**;
the final run records **{b7['tensor_prefill']['tokens_per_second']:,.1f} / {b7['tensor_decode']['tokens_per_second']:.1f}**.
These endpoint ratios describe the campaign, including runtime, chunking and
later arithmetic changes. Controlled gains come from the paired experiments
linked below.

## The archive timeline

{archive_timeline(data)}

These are the commits retaining the exact evidence snapshots used here, with
commit times converted to Asia/Jakarta. Several experiments landed together;
the dates are archive times, not reconstructed measurement times. The initial
2.6B file was added in `5bd34b9`; its corrected snapshot used for B1 landed in
`9c710e1` after the implementation correction in `002c2c7`. Both original and
corrected measurements are retained as separate stages, B0 and B1.

## The machine and the first attempt

All plotted Tensor GPU measurements use the physical **RX 6700 XT 12 GB**,
Ryzen 5 5600, Windows 11 26200, AMD Vulkan driver **26.6.2**
(Windows driver `32.0.21043.19003`), Python 3.12.13, NumPy 2.5.3 and
wgpu-py 0.29.0 / wgpu-native 27.0.2.0. The first hardware validation requested
`shader-f16`; later profiles also request subgroups. Generic schedules stay
within 32 KiB of workgroup storage. The 2.6B output matrix requires a 256 MiB
buffer-binding opt-in.

The [first hardware run](webgpu-rx6700xt.md) validates 33 inference/composition
checks on Vulkan. D3D12 completes six FP32 affine checks, then rejects FP16
because `shader-f16` is absent. Its partial result is not a Vulkan performance
point. The subsequent [two-host transfer](webgpu-rx6700xt-transfer.md) validates
compiler-free execution and source/artifact hashes across Linux and Windows.

The [first 230M forward](lfm2-230m-vulkan.md) works and passes independent
NumPy gates, but decode is about 5.3× slower than native for F16 and 9.1× for
Q4_0. Q4 saves weight memory without yet improving decoding. Packed field
extraction, tiled accumulation, attention masking and submission overhead
become the concrete optimization targets.

## Kernel evolution: retain outputs, then improve ownership and reuse

![Kernel improvements with separate timing contracts](data/rx6700xt-vulkan-history-kernels.png)

[Vector figure](data/rx6700xt-vulkan-history-kernels.svg).
The panels deliberately retain separate operations and timing boundaries.

**Generic FP16 linear, 4096³.** The first allocating-call scaling sweep records
**206.811 ms**. Default 2×2 private microtiles later record **108.726 ms**;
whole-K private accumulators record **67.811 ms**. Shapes, seeded inputs and
output hashes agree across these three archived points. The fresh causal
controls are **204.620 → 108.726 ms (1.88×)** for
[microtiles and runtime](latency-scaling.md#rx-6700-xt-repeat-after-compiler-optimization)
and **108.728 → 67.811 ms (1.60×)** for
[whole-loop accumulation](webgpu-gemm-accumulation.md). The initial-to-final
3.05× ratio is a historical comparison, not one interleaved experiment.

**FP32 pure GEMM, 4096³.** A preallocated matched whole-K ablation is
**197.451 → 113.056 ms (1.75×)**, with bitwise-identical outputs. A separate
[outer-product comparison](webgpu-outer-product-gemm.md) is
**112.956 → 33.515 ms (3.37×)**. The latter uses a prepared plan; its generic
prepared repeat is **113.051 ms**, isolating the schedule gain from binding
overhead. Stock **CLBlast 1.6.3 OpenCL** completes the same FP32 operation in
**16.373 ms**. Its remaining 2.05× lead supplies evidence of further compute
headroom. These FP32 bars are not appended to the FP16 linear curve.

**Packed 2.6B FFN.** The final same-arithmetic layout replay reduces GPU time
from **1.951 to 1.710 ms per paired gate/up/SwiGLU**: **12.4% less latency**.
It streams all 30 gate/up matrix pairs at rows 128, K=2048, N=10752. One packed
word of shared padding and row-first workgroup placement are selected; most
of the observed gain is padding. The schedule uses 6,240 bytes of shared
storage. This is a controlled kernel improvement after the short-F16 arithmetic
had already been introduced.

The useful compiler sequence is materialized per-tile fragments → register
microtiles → private lifetime across K → explicit outer-product ownership,
layout and unroll → packed F16/integer schedules with independent validation.
Whole-K lifetime does not alter every attention loop: attention consumes and
rescales intermediate results. Small-K regressions also prevent blanket use of
the private-accumulator transformation.

The broader [scaling repeat](latency-scaling.md#rx-6700-xt-repeat-after-compiler-optimization)
also records limits of the gain: noncausal/causal attention at S8192 improves
from 267.721/134.475 to 163.900/84.631 ms under microtiles, while the byte-identical
64M-element pointwise shader stays at 28.429 ms in both fresh controls.
The 512³ allocating linear case regresses from 1.333 to 1.832 ms in that repeat.
Whole-K lowering subsequently leaves streaming attention essentially unchanged,
because its intermediate rescaling prevents the transformation. The large-GEMM
plot therefore illustrates one workload family rather than a universal gain.

## 230M: make the model small enough to iterate quickly

![230M F16 and Q4 histories](data/rx6700xt-vulkan-history-230m.png)

[Vector figure](data/rx6700xt-vulkan-history-230m.svg).
Each column follows its own checkpoint; rates are completed tokens/s.

### Native F16 checkpoint

{table(h['230m_f16'])}

H1 puts projection accumulators in registers, reuses packed fields, masks
inactive attention positions and selects fusion by encoding. H2 adds generic
microtiles/tree reductions, subgroup decode, typed packed loads and prepared
binding reuse. H3 moves dispatch encoding into an optional C helper and adds
vector prefill dots.

H4 is principally a Q4 optimization. Its F16 decode before/after is
268.5 → 268.8 tokens/s, so the historical point is a repeat, not evidence of
a material F16 gain. H5 transfers searched 32-row FFN schedules into full
inference. H6 separately searches GEMV and attention and improves decode.
H7 changes ordered readback/submission with unchanged shaders. H8 adapts
prefill chunks to 32/128 rows and transfers shape-specific outer products.

The [tinygrad comparison](lfm2-tinygrad-comparison.md) motivates a producer-side
search layer over legal initial tile programs, checked by an independent NumPy
oracle. The [30-minute Tensor search](lfm2-tensor-search-comparison.md) considers
2,368 configurations and times 1,769 valid candidates; the projection winners
then become the H5 full-model schedules. The separate OpenCL tinygrad adapter
also improves prefill, but its decoding remains near 40 tokens/s. Its initially
FP32-expanded F16 storage was corrected before the accepted full-model comparison.
Faster isolated projections therefore do not directly imply faster decoding
or whole-model prefill; kernel transfer and runtime measurements are necessary.

The H7→H8 decode dip is preserved. The H8 same-session chunk-32 versus adaptive
comparison is 437.6 versus 437.4 tokens/s; it does not establish a regression
from the prefill schedules. Histories retain separate-session variation.

### Q4_0 checkpoint

{table(h['230m_q4_0'])}

Q4 gains a real decode advantage once its packed fields/scales are reused and
gate/up projections are fused. The later prefill/decode searches above were
measured for native F16. The plot stops the Q4 trace at H4 instead of filling
later stages with assumed gains.

Both use context 512, identical forced token IDs within comparisons, F16 K/V
caches and 64 completed decode calls after each prefix. H0–H4 have five timed
samples after one warmup; H5–H8 have seven after three. Tensor prefill chunks
are 32 through H7 and adaptive 32/128 at H8. Native batch/ubatch also changes
from 32 to 128 at H8; the plot separates that native reference point.

## 2.6B: expose the bandwidth gain, then attack prefill compute

![2.6B packed-model throughput history](data/rx6700xt-vulkan-history-2.6b.png)

[Vector figure](data/rx6700xt-vulkan-history-2.6b.svg).
The shaded region changes prefill arithmetic. The 1K line is a measured target,
not an extrapolation from GEMM throughput.

{table(h['2.6b_qad_q4_0'])}

B0 comes from the initial JSON committed in `5bd34b9`, recovered byte-for-byte
into [a separate snapshot](data/lfm2-2.6b-q4_0-initial-vulkan.json). The ordinary
matched-run JSON was later replaced by the corrected result, now B1. The first
completed-call medians are **195.4 prefill / 120.8 decode tokens/s** at 128.
B1 corrects RMS reduction width and staging-loop divisions/output tiles,
reaching **232.5 / 130.6**. Its instrumented decode profile separately records
**7.836 → 6.520 ms**; full-forward rates are taken directly from their samples,
not reconstructed from those GPU timestamps.

B2 uses packed floating dots and FFN-down residual fusion. B3 isolates a
readback improvement and reuses weights across larger chunks. B4 searches
Q4/Q6 projections and fused attention while retaining F32 decode arithmetic.
These bring decode near native, while prefill is still around 522 tokens/s.

B5 is an experimental three-sample pilot: two Q8 components approximate
F16-rounded activations with F32 scale/total accumulation. B6 adds short
even/odd F16 FMA chains for selected projections, contributing into F32 totals.
B7 preserves all final-attention K/V but executes only the required query and
convolution suffix and last FFN token; intermediate chunks skip unused logits.
B8 retains that arithmetic/liveness plan and adds the independently replayed
shared layout. Public chunk choices remain 1/32/128; the final suffix uses
eight internal rows. The two three-tap convolutions need a five-row receptive
field, and tests verify both final output and persistent histories.

All seven B8 prefill samples exceed 1K at 128 and 384 tokens. At prefix 128,
decode remains about 165 tokens/s against native 170; native prefill is about
1,941 against Tensor 1,058. Native remains faster. The final timestamp trace
puts linear plus FFN work at roughly 94% of instrumented GPU time: the next
prefill bottleneck is still projection compute.

## What the controlled experiments actually establish

| Experiment | Fresh control → selected result | Attribution |
|---|---|---|
| Initial 230M F16 optimization | Prefill 212.6 → 1,569.7; decode 89.7 → 141.3 tok/s | Combined kernel/plan changes, separate frozen before/after runs |
| 230M prefill search | 1,773.8 → 2,612.2 tok/s | Same 32-row chunks; decode shaders unchanged |
| 230M decode search | 271.2 → 396.3 tok/s | Separate GEMV/attention search; prefill nearly unchanged |
| 230M readback | 396.7 → 441.9 tok/s | Same shaders, bitwise-equal 19-fixture logits |
| 230M adaptive prefill | 5,314 → 5,788 tok/s at fixed rows 128 | Schedule gain 8.9%; most total gain comes from larger chunks |
| 2.6B readback | 138.7 → 145.7 tok/s | Same-shader runtime ablation; bitwise-equal logits |
| 2.6B packed prefill search | 345.2 → 522.1 tok/s | Same adaptive chunk sizes |
| 2.6B final mixed prefill | 525.6 → 1,058.4 tok/s | Mixed arithmetic, reduced suffix work and layout, measured together |

The history does not multiply every advertised stage gain. Fresh controls can
differ from the previous report's endpoint, and stages can combine changes.
Kernel GPU timestamps, allocating calls, completed preallocated calls,
full-logit forward rates and greedy generation remain separate measurements.

## Precision, rejected routes and the evidence boundary

Each plotted inference point passes **19 independent NumPy fixtures**, with
finite logits, relative RMS below 1%, cosine above 0.9999 and matching argmax.
Each checkpoint SHA is checked during extraction. Native-F16 and ordinary
230M Q4_0 are separate files; 2.6B uses the distinct **QAD Q4_0** file throughout.
Native b11310 is pinned to `{NATIVE}`.

Before B5, prefill uses nearest-even F16 operands and F32 totals; decode uses
F32 arithmetic. B5–B8 are approximate prefill contracts. B8's maximum relative
RMS is 0.71023%; native ordinary Vulkan arithmetic is about 10.88% against
these fixtures, although both match every argmax. Passing the same model gate
does not make their arithmetic equivalent. Reset and bounded host/GPU greedy
checks are recorded in the original reports.

Rejected routes are part of the engineering history: F16 paired decode fusion
was initially slower than separate projections; a 230M prefill RMS reorder
failed the unchanged model gate; small generic GEMMs regressed under forced
whole-K accumulation; wider decode discovery and fused attention did not beat
the retained runtime profile; blanket prefill unrolling regressed throughput.
Later expanded caches, wider integer panels and direct F16 shared staging did
not justify transfer into the final packed runtime. The parallel RMS accepted
in the later 2.6B experiment does not retroactively validate the rejected 230M
case. A six-runner memory-pressure run that slowed native to roughly 13 tokens/s
is excluded from the accepted milestones, as its source report documents.

This is the first **retained** evidence, not a claim to enumerate unrecorded
prototype attempts. Stage order follows experiment dependencies, not uniform
elapsed time. Archived base Git SHAs sometimes predate dirty measured code;
source snapshots, implementation fingerprints and WGSL identify those kernels.
The derived JSON records each retained snapshot commit and its canonical Git
blob digest alongside the working-file digest; Git CRLF normalization can make
those byte digests differ without changing the JSON evidence.
The final implementation lands in `d5ee508`; the generic whole-K and outer-product
experiments precede it. Driver/compiler upgrades and longer contexts need new
measurements. No GPU performance ceiling or general model-quality claim follows
from this bounded record.

## Rebuild the historical document and figures

Run from the repository root with Git history through `d5ee508` available;
this reads evidence and restores the initial 2.6B snapshot without running a GPU:

```powershell
uv run --no-project --python 3.12 --with matplotlib==3.11.2 --with numpy==2.5.3 python scripts/plots/plot_rx6700xt_vulkan_history.py
```

The generator validates recorded medians against sample latencies, model/native
hashes, 19-fixture gates and matching generic scaling input/output hashes.
It writes this report, three PNG/SVG figures, a compact JSON extraction and CSV.
Each point retains its original source file and JSON pointer; source SHA256
digests make historical edits detectable. PNGs are for reading, SVGs for export.
Sample whiskers are min/max ranges, not confidence intervals. No later point is
backfilled for an unmeasured encoding and no result from a rejected variant is
substituted for an accepted milestone.
'''
    path.write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('docs/research/data'))
    parser.add_argument('--document',type=Path,default=Path('docs/research/rx6700xt-vulkan-history.md'))
    args=parser.parse_args();e=Evidence(args.data)
    first_hardware=e.read('webgpu-rx6700xt-vulkan')
    assert first_hardware['status']=='passed' and len(first_hardware['cases'])==33
    data={'schema':'tensor.rx6700xt-vulkan-history.v1','evidence_cutoff_commit':CUTOFF,
          'compiled_at':'2026-10-04','measured_campaign':'2026-10-01 through 2026-10-03',
          'inference':histories(e),'kernels':kernels(e),'sources':e.index,
          'policy':'Historical observations, not a cumulative controlled ablation. GPU kernels and completed forwards are separate; arithmetic/chunk/sample changes are marked.'}
    args.data.mkdir(parents=True,exist_ok=True)
    (args.data/'rx6700xt-vulkan-history.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    with (args.data/'rx6700xt-vulkan-history.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['workload','stage','change','prompt_tokens','tensor_prefill_tps','tensor_decode_tps','native_prefill_tps','native_decode_tps','samples','chunks','arithmetic','source','snapshot_commit','pointer'])
        for kind,stages in data['inference'].items():
            for stage in stages:
                for row in stage['measurements']:
                    writer.writerow([kind,stage['id'],stage['label'],row['prompt_tokens'],
                        *[row[key]['tokens_per_second'] for key in ('tensor_prefill','tensor_decode','native_prefill','native_decode')],
                        row['samples_per_runner'],'/'.join(map(str,stage['chunks'])),stage['arithmetic'],stage['source'],stage['snapshot_commit'],row['pointer']])
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none',
                         'svg.hashsalt':'rx6700xt-vulkan-history','axes.labelcolor':INK,'text.color':INK})
    plot_kernels(data,args.data);plot_230m(data,args.data);plot_big(data,args.data)
    document(data,args.document)
    print(f"Verified {len(e.index)} evidence files and {sum(len(v) for v in data['inference'].values())} inference milestones; wrote report, JSON/CSV and three PNG/SVG figures.")


if __name__=='__main__':main()
