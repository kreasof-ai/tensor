"""Single-sequence LFM2 inference with packed weights and persistent device state."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import ctypes as ct
import hashlib
import json
import numpy as np
from tensor.runtime.abi import BoundCall
from tensor.providers.cuda_graph import CudaGraph
from .config import Config
from .gguf import GGUF, TYPES
from .kernels import identity
from .tokenizer import Tokenizer
from .provenance import implementation_hashes


DECODE_GEMV_COMMON=dict(lanes=32,micro_rows=1,dot_width=4,unroll=8,
                        accumulators=4,k_layout='striped',shared_input=False)
WEBGPU_PROFILES=('portable','subgroup','searched','decode_searched','decode_fused',
                 'prefill_unrolled','prefill_outer','prefill_chunked','quant_searched','prefill_q16','prefill_mixed')
SUBGROUP_PROFILES=WEBGPU_PROFILES[1:]
QUANT_DECODE={
    (2048,128000,14):dict(gemv_q6_dot=True,gemv_threads=256,gemv_unroll=4,gemv_chains=1),
    (2048,6144,2):dict(gemv_unroll=8,gemv_chains=8),
    (10752,2048,2):dict(gemv_lanes=16,gemv_threads=64,gemv_dot=True,gemv_unroll=4),
    (2048,2048,2):dict(gemv_unroll=8,gemv_chains=8),
    (2048,512,2):dict(gemv_lanes=32,gemv_threads=64,gemv_dot=True,gemv_unroll=4),
}
QUANT_ATTENTION=dict(attention_schedule='partitioned_values',channels=64,
                     value_parts=8,fused_scores=True)
Q16_SMALL=dict(tile_m=16,tile_n=32,micro_m=2,micro_n=2,threads=128,
               owner_axis='column',group_order='row')
Q16_MEDIUM=dict(tile_m=32,tile_n=64,micro_m=2,micro_n=4,threads=256,
                owner_axis='column',group_order='row')
Q16_PREFILL={
    ('ffn',128,2048,10752,2):Q16_SMALL,
    ('linear',128,2048,6144,2):Q16_MEDIUM,
    ('linear',128,10752,2048,2):dict(tile_m=64,tile_n=128,micro_m=4,micro_n=8,
        threads=256,owner_axis='column',group_order='row'),
    ('linear',128,2048,2048,2):Q16_MEDIUM,
    ('linear',128,2048,512,2):Q16_SMALL,
}
HALF_PREFILL={
    ('ffn',128,2048,10752,2):dict(tile_m=64,tile_n=64,tile_k=16,micro_m=4,micro_n=4,
        threads=256,lhs_layout='km',unroll=8,explicit_unroll=True,
        dot_width=2,packed_pairs=True,half_accum=True,lhs_pad=1,rhs_pad=1,group_order='row'),
    ('linear',128,2048,6144,2):dict(tile_m=64,tile_n=128,tile_k=16,micro_m=4,micro_n=8,
        threads=256,lhs_layout='km',unroll=8,explicit_unroll=True,
        dot_width=2,packed_pairs=True,half_accum=True),
    ('ffn',8,2048,10752,2):dict(tile_m=8,tile_n=64,tile_k=16,micro_m=1,micro_n=4,
        threads=128,lhs_layout='km',unroll=8,explicit_unroll=True,
        dot_width=2,packed_pairs=True,half_accum=True),
    **{('linear',8,k,o,2):dict(tile_m=8,tile_n=32,tile_k=16,micro_m=1,micro_n=2,
        threads=128,lhs_layout='km',unroll=8,explicit_unroll=True,dot_width=2,packed_pairs=True)
        for k,o in ((2048,6144),(2048,2048),(10752,2048))},
}


def prefill_tail_rows(cfg,gguf,profile):
    """Bounded suffix liveness: two width-three convs need five input rows.

    Crop queries after the final attention has stored every K/V. Eight rows
    cover its pointwise FFN and both conv histories; their discarded early
    outputs may differ, but the final output and persistent histories agree.
    """
    if profile!='prefill_mixed' or (cfg.width,cfg.ff)!=(2048,10752) or tuple(cfg.layers[-3:])!=('attention','conv','conv'):
        return 0
    for i in range(len(cfg.layers)-3,len(cfg.layers)):
        names=['ffn_gate','ffn_up','ffn_down']
        if cfg.layers[i]=='conv':names += ['shortconv.in_proj','shortconv.out_proj']
        else:names += ['attn_q','attn_output']
        if any(gguf.tensors[f'blk.{i}.{name}.weight'].type!=2 for name in names):return 0
    return 8
# Experimental schedule selection populated from independent F16 replay.
PREFILL_OUTER_MK32=dict(tile_m=32,tile_n=32,tile_k=32,micro_m=2,micro_n=2,
    threads=256,lhs_layout='mk',lhs_pad=0,rhs_pad=0,owner_axis='column',
    unroll=8,explicit_unroll=True,fma=False)
PREFILL_OUTER_KM16={**PREFILL_OUTER_MK32,'tile_n':64,'tile_k':16,
    'lhs_layout':'km','fma':True}
PREFILL_OUTER={
    ('linear',32,1024,3072):dict(explicit_unroll=True,unroll=True),
    **{('linear',r,1024,o):dict(schedule='outer',outer=PREFILL_OUTER_MK32)
       for r,o in ((32,1024),(32,512),(64,3072),(64,1024),(64,512),(128,512))},
    ('linear',128,1024,3072):dict(schedule='outer',outer={**PREFILL_OUTER_KM16,
        'micro_m':4,'micro_n':4,'threads':128}),
    ('linear',128,1024,1024):dict(schedule='outer',outer={**PREFILL_OUTER_KM16,
        'micro_n':4}),
}
QUANT_KM16={**PREFILL_OUTER_MK32,'tile_k':16,'lhs_layout':'km','fma':True}
QUANT_LARGE={**QUANT_KM16,'tile_m':64,'tile_n':64,'micro_m':4,'micro_n':4}
QUANT_PREFILL={
    ('ffn',32,2048,10752,2):dict(schedule='outer',outer={**QUANT_KM16,
        'tile_n':64,'micro_m':4,'micro_n':4,'threads':128}),
    ('linear',32,10752,2048,2):dict(schedule='outer',outer=QUANT_KM16),
    **{('linear',r,2048,o,2):dict(schedule='outer',outer=PREFILL_OUTER_MK32)
       for r,o in ((32,6144),(32,2048),(32,512),(128,512))},
    ('ffn',128,2048,10752,2):dict(schedule='outer',outer=QUANT_LARGE),
    ('linear',128,2048,6144,2):dict(schedule='outer',outer={**QUANT_LARGE,
        'tile_n':128,'micro_n':8}),
    **{('linear',128,k,2048,2):dict(schedule='outer',outer=QUANT_LARGE)
       for k in (2048,10752)},
}


def valid_rows(provider,rows,profile):
    if type(rows) is not tuple or not rows or any(type(r) is not int for r in rows):return False
    if provider=='cuda':return rows==(1,128)
    if provider!='webgpu':return False
    return rows==(1,32) or (profile in ('prefill_unrolled','prefill_outer','prefill_chunked','quant_searched','prefill_q16','prefill_mixed')
        and rows[0]==1 and len(rows)>1 and rows[1:]==tuple(sorted(set(rows[1:])))
        and all(r in (32,64,128) for r in rows[1:]))
# Independently replayed winners on full 230M F16 weight traffic.
# Keys are (input depth, output columns). Other models retain their schedules.
DECODE_GEMV={
    (1024,2560):dict(family='paired',threads=256),
    (1024,65536):dict(family='separate',threads=256),
    (2560,1024):dict(family='separate',threads=64,micro_rows=2,unroll=4),
    (1024,3072):dict(family='separate',threads=256),
    (1024,1024):dict(family='separate',threads=64),
    (1024,512):dict(family='separate',threads=64),
}


def paired_ffn(rows,gate,up,profile=None):
    # Quantized decode benefits from shared unpack/input work; the measured
    # F16 decode pair loses occupancy. Prefill benefits from register reuse.
    searched=(profile in ('decode_searched','decode_fused','prefill_unrolled','prefill_outer','prefill_chunked','quant_searched','prefill_q16','prefill_mixed') and rows==1 and gate.type==1 and
              DECODE_GEMV.get(tuple(reversed(gate.shape)),{}).get('family')=='paired')
    return gate.type==up.type and (rows>1 or gate.type not in (0,1) or searched)


# A full-width output tile doubles the work per workgroup and halves the launch
# count. It only pays while enough column tiles remain to fill the device:
# measured on the 2.6B FFN shape (o=10752), while the narrower ffn_down
# (o=2048) regresses by 12%. The guard is therefore on column count, not depth.
WIDE_TILE_COLUMNS=5120
WIDE_TILE=(32,32,64)

def projection_tile(rows,columns):
    return {'tile':WIDE_TILE} if rows>1 and columns>=WIDE_TILE_COLUMNS else {}


def webgpu_parameters(kind,p,profile):
    """Apply opt-in schedules measured on RX 6700 XT."""
    if profile=='prefill_mixed':
        selected=HALF_PREFILL.get((kind,p.get('r'),p.get('k'),p.get('o'),p.get('type')))
        if selected is not None:return dict(p,sg=True,schedule='outer',outer=dict(selected))
        return webgpu_parameters(kind,p,'prefill_q16')
    if profile=='prefill_q16':
        result=webgpu_parameters(kind,p,'quant_searched')
        selected=Q16_PREFILL.get((kind,p.get('r'),p.get('k'),p.get('o'),p.get('type')))
        if selected is not None:return dict(p,q16=True,integer=dict(selected))
        if kind=='rms' and p.get('r',1)>1:result['parallel_rows']=True
        return result
    if profile=='quant_searched':
        result=webgpu_parameters(kind,p,'prefill_chunked')
        if p.get('type') in (2,14) and kind in ('linear','linear_add','ffn'):
            selected=(QUANT_DECODE.get((p['k'],p['o'],p['type'])) if p.get('r')==1 and kind!='ffn'
                      else QUANT_PREFILL.get((kind,p.get('r'),p['k'],p['o'],p['type'])))
            if selected is not None:
                result.update(selected)
                if 'outer' in selected:result['outer']=dict(selected['outer'])
        if kind=='attention' and p.get('r')==1 and (p['h'],p['kh'],p['d'])==(32,8,64):
            result.update(QUANT_ATTENTION)
        return result
    if profile in ('prefill_unrolled','prefill_outer','prefill_chunked'):
        rows=p.get('r',1)
        # Carry the current F16 FFN schedules over to larger prefill chunks.
        seed={**p,'r':32} if rows>1 and kind in ('linear','ffn') else p
        result=webgpu_parameters(kind,seed,'decode_searched')
        if 'r' in p:result['r']=rows
        if rows>1 and kind in ('linear','ffn') and p.get('type')==1:
            selected=PREFILL_OUTER.get((kind,rows,p['k'],p['o'])) if profile=='prefill_outer' else None
            if selected is not None:
                result.update(selected)
                if 'outer' in selected:result['outer']=dict(selected['outer'])
            elif profile=='prefill_unrolled':
                result['explicit_unroll']=True
                result['unroll']=16 if result.get('schedule')=='partitioned' else True
        return result
    if profile=='decode_fused':
        result=webgpu_parameters(kind,p,'decode_searched')
        if kind=='attention' and p.get('r')==1 and (p['h'],p['kh'],p['d'])==(16,8,64):
            result.update(fused_scores=True,channels=32,value_parts=16)
        return result
    p=dict(p)
    if profile in ('subgroup','searched','decode_searched') and kind in ('linear','linear_add','ffn','attention','attention_scores','rms','add_rms'):
        p['sg']=True
    if profile in ('searched','decode_searched') and kind in ('linear','ffn') and p.get('r')==32 and p.get('type')==1:
        shape=(p['k'],p['o'])
        if shape in ((1024,2560),(2560,1024)):
            gate=shape==(1024,2560)
            p.update(schedule='partitioned',tile=(8 if kind=='ffn' else 16,8,64) if gate else (4,8,64),
                     threads=128 if gate else 64,partitions=16,unroll=4,dot_width=4,
                     owner_axis='row',k_layout='blocked' if gate else 'striped')
    if profile=='decode_searched' and p.get('r')==1:
        if kind in ('linear','linear_add','ffn') and p.get('type')==1:
            config=DECODE_GEMV.get((p['k'],p['o']))
            if config is not None:
                p.update(DECODE_GEMV_COMMON);p.update({k:v for k,v in config.items() if k!='family'})
                p['decode_schedule']='streamed'
        if kind=='attention' and (p['h'],p['kh'],p['d'])==(16,8,64):
            p.update(attention_schedule='partitioned_values',channels=64,value_parts=16)
    return p


def requirements(gguf,capacity,rows=(1,128),*,provider='cuda',webgpu_profile='portable'):
    cfg=Config.from_gguf(gguf);c=cfg.width
    values={}
    if webgpu_profile not in WEBGPU_PROFILES:raise ValueError('unsupported WebGPU kernel profile')
    def add(kind,**p):
        if provider=='webgpu':p=webgpu_parameters(kind,p,webgpu_profile)
        values[identity(kind,p)]=(kind,p)
    for r in rows:
        for info in gguf.tensors.values():
            if len(info.shape)==2 and not info.name.endswith('conv.weight'):
                o,k=info.shape
                if info.name=='token_embd.weight':
                    add('embedding',r=r,c=c,v=cfg.vocab,type=info.type)
                    add('linear',r=1,k=k,o=o,type=info.type)
                elif info.name=='output.weight':add('linear',r=1,k=k,o=o,type=info.type)
                else:
                    add('linear',r=r,k=k,o=o,type=info.type,**projection_tile(r,o))
                    if provider=='webgpu' and webgpu_profile in ('prefill_q16','prefill_mixed') and r>1 and any(key[1]==r and key[2]==k for key in Q16_PREFILL):
                        add('quantize_q16',r=r,k=k,sg=True)
                    if provider=='webgpu' and r==1 and info.name.endswith('ffn_down.weight'):
                        add('linear_add',r=r,k=k,o=o,type=info.type)
        add('rms',r=r,c=c,eps=cfg.epsilon)
        if provider=='webgpu':
            if r==1:add('add_rms',r=r,c=c,eps=cfg.epsilon)
            for i in range(len(cfg.layers)):
                gate,up=(gguf.tensors[f'blk.{i}.ffn_{name}.weight'] for name in ('gate','up'))
                if paired_ffn(r,gate,up,webgpu_profile):add('ffn',r=r,k=c,o=cfg.ff,type=gate.type,**projection_tile(r,cfg.ff))
        add('add',r=r,c=c);add('swiglu',r=r,c=cfg.ff);add('conv',r=r,c=c)
        for kind in (('qnorm','kvnorm') if provider=='webgpu' else ('qkv',)):
            add(kind,r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=capacity,eps=cfg.epsilon,theta=cfg.theta)
        if provider=='webgpu' and webgpu_profile in SUBGROUP_PROFILES and r==1:
            add('attention_scores',r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=capacity)
        add('attention',r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=capacity)
        add('last',r=r,c=c);add('advance',r=r)
    add('rms',r=1,c=c,eps=cfg.epsilon)
    tail=prefill_tail_rows(cfg,gguf,webgpu_profile) if provider=='webgpu' and 128 in rows else 0
    if tail:
        add('prefill_tail',r=128,c=c,t=tail)
        for k,o in ((c,3*c),(c,c),(cfg.ff,c)):
            add('linear',r=tail,k=k,o=o,type=2,**projection_tile(tail,o))
        add('ffn',r=tail,k=c,o=cfg.ff,type=2,**projection_tile(tail,cfg.ff))
        add('rms',r=tail,c=c,eps=cfg.epsilon);add('add',r=tail,c=c)
        add('conv',r=tail,c=c);add('last',r=tail,c=c)
        add('attention',r=tail,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=capacity)
        add('qnorm',r=tail,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=capacity,eps=cfg.epsilon,theta=cfg.theta)
    if provider=='webgpu':add('argmax',n=cfg.vocab)
    return values


class LFM2:
    """Execute an installed GGUF against a matching AOT Tensor kernel bundle.

One sequence; chunked prompt prefill and single-token decode. Weights retain
GGML encodings. Convolution history is FP32 and attention caches are FP16.
"""
    def __init__(self,model,bundle,device,*,context=8448,graphs=True):
        self.gguf=GGUF(model);self.config=Config.from_gguf(self.gguf)
        self.tokenizer=Tokenizer(self.gguf.metadata);self.device=device;self.directory=Path(bundle)
        self.manifest=json.loads((self.directory/'inference.json').read_text())
        self.provider=device.info['provider']
        if self.provider not in ('cuda','webgpu'):raise ValueError('LFM2 requires CUDA or WebGPU')
        if self.manifest.get('provider','cuda')!=self.provider:raise ValueError('LFM2 bundle requires a different provider')
        if self.manifest.get('schema')!='tensor.lfm2-inference.v1':raise ValueError('unsupported LFM2 bundle schema')
        if self.manifest['config']!=json.loads(json.dumps(asdict(self.config))):raise ValueError('bundle belongs to a different LFM2 architecture')
        if self.manifest.get('implementation') != implementation_hashes(self.provider):
            raise ValueError('LFM2 bundle implementation mismatch; rebuild with the installed producer')
        self.capacity=self.manifest['capacity'];self.rows=tuple(self.manifest['rows'])
        profile=self.manifest.get('webgpu_profile','portable')
        if not valid_rows(self.provider,self.rows,profile) or self.capacity % 64:
            raise ValueError('unsupported LFM2 prefill/capacity profile')
        if not 1<=context<=min(self.capacity-max(self.rows),self.config.max_context):raise ValueError('context exceeds the compiled capacity')
        self.context=context;self.graphs_enabled=graphs;self.position=0;self.closed=False
        self.webgpu_profile=self.manifest.get('webgpu_profile','portable')
        # The final FFN feeds only logits; all persistent layer state is already
        # written by its mixer. Restrict this measured opt-in shape to one row.
        self.prefill_last=(self.provider=='webgpu' and self.webgpu_profile=='prefill_mixed'
            and (self.config.width,self.config.ff)==(2048,10752)
            and all(self.gguf.tensors[f'blk.{len(self.config.layers)-1}.ffn_{name}.weight'].type==2
                    for name in ('gate','up','down')))
        self.prefill_tail=prefill_tail_rows(self.config,self.gguf,self.webgpu_profile) if self.provider=='webgpu' else 0
        expected=requirements(self.gguf,self.capacity,self.rows,provider=self.provider,webgpu_profile=self.webgpu_profile)
        if set(expected)!=set(self.manifest['kernels']):raise ValueError('incomplete/incompatible LFM2 kernel coverage')
        for record in self.manifest['kernels'].values():
            path=(self.directory/record['artifact']).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError('LFM2 artifact path escapes its bundle')
            with path.open('rb') as image:
                if hashlib.file_digest(image,'sha256').hexdigest()!=record['sha256']:
                    raise ValueError('LFM2 kernel checksum mismatch')
        self.kernels={};self.graphs={};self.plans={};self.buffers=[];self.prepared={};self.prepared_states={};self.greedy={}
        def upload(host):
            b=device.from_numpy(np.ascontiguousarray(host));self.buffers.append(b);return b
        self.weights={}
        for name,info in self.gguf.tensors.items():
            raw=self.gguf.packed(name)
            if info.type in (0,1):raw=raw.view(np.float32 if info.type==0 else np.float16)
            elif self.provider=='webgpu':raw=raw.view(np.uint32)
            self.weights[name]=upload(raw)
        self.control=upload(np.array([0,1],np.int32));self.logits=upload(np.zeros(self.config.vocab,np.float32))
        self.states={};self.caches={}
        for i,kind in enumerate(self.config.layers):
            if kind=='conv':self.states[i]=upload(np.zeros(2*self.config.width,np.float32))
            else:self.caches[i]=(upload(np.zeros(self.capacity*self.config.kv_heads*self.config.head_dim,np.float16)),
                                 upload(np.zeros(self.capacity*self.config.kv_heads*self.config.head_dim,np.float16)))
        self.workspaces={}
        for r in self.rows:
            c,f,kh,d=self.config.width,self.config.ff,self.config.kv_heads,self.config.head_dim
            shapes={'tokens':(r,np.int32),'hidden':(r*c,np.float32),'normal':(r*c,np.float32),
                    'mixed':(r*c,np.float32),'project':(r*3*c,np.float32),'convout':(r*c,np.float32),
                    'q':(r*c,np.float32),'k':(r*kh*d,np.float32),'v':(r*kh*d,np.float32),
                    'qo':(r*c,np.float32),'attn':(r*c,np.float32),'gate':(r*f,np.float32),
                    'up':(r*f,np.float32),'activated':(r*f,np.float32),'last':(c,np.float32),'final':(c,np.float32)}
            self.workspaces[r]={name:upload(np.zeros(count,dtype)) for name,(count,dtype) in shapes.items()}
            if self.provider=='webgpu':self.workspaces[r]['hidden2']=upload(np.zeros(r*c,np.float32))
            if self.prefill_last and r==128:
                for name,count in (('last_normal',c),('last_activated',f),('last_output',c)):
                    self.workspaces[r][name]=upload(np.zeros(count,np.float32))
            if self.prefill_tail and r==128:
                t=self.prefill_tail
                tail_shapes={'hidden':t*c,'hidden2':t*c,'normal':t*c,'mixed':t*c,
                    'project':t*3*c,'convout':t*c,'activated':t*f,'last':c,'final':c,'q':t*c,'qo':t*c,'attn':t*c,
                    'last_normal':c,'last_activated':f,'last_output':c}
                self.workspaces[r]['tail']={name:upload(np.zeros(count,np.float32)) for name,count in tail_shapes.items()}
                self.workspaces[r]['tail_control']=upload(np.zeros(2,np.int32))
            if self.provider=='webgpu' and self.webgpu_profile in SUBGROUP_PROFILES and r==1:
                self.workspaces[r]['scores']=upload(np.zeros(self.config.heads*self.capacity,np.float32))
            if self.provider=='webgpu' and self.webgpu_profile in ('prefill_q16','prefill_mixed') and r>1:
                self.workspaces[r]['q16']={k:(upload(np.zeros(r*k//2,np.uint32)),
                    upload(np.zeros(r*k//16,np.float32)),upload(np.zeros(r*k//16,np.int32)))
                    for k in sorted({key[2] for key in Q16_PREFILL if key[1]==r})}
        for key,record in self.manifest['kernels'].items():
            path=self.directory/record['artifact']
            self.kernels[key]=device.load(path)
        for r in self.rows:self.plans[r]=self._plan(r)
        if self.provider=='webgpu':
            for r,plan in self.plans.items():self.prepared[r]=device.prepare_plan(plan)
            if self.prefill_tail:
                kinds={id(self.kernels[key]):record['kind'] for key,record in self.manifest['kernels'].items()}
                for r,plan in self.plans.items():
                    if r==1:continue
                    end=max(i for i,(kernel,_) in enumerate(plan) if kinds[id(kernel)]=='conv')
                    self.prepared_states[r]=device.prepare_plan((*plan[:end+1],plan[-1]))
            sample=self.kernels[identity('argmax',dict(n=self.config.vocab))]
            values,symbols,launch=sample._bind((self.logits,self.workspaces[1]['tokens'],self.control),{},include_outputs=True)
            call=BoundCall(device,sample.manifest,values,symbols,launch,validated=True)
            for r,plan in self.plans.items():self.greedy[r]=device.prepare_plan((*plan,(sample,call)))
            self.graphs_enabled=False
        elif graphs:
            for r,plan in self.plans.items():
                self.graphs[r]=CudaGraph(device,lambda plan=plan:self._submit(plan),resources=(*self.buffers,*self.kernels.values()))
        self.allocated_bytes=sum(b.nbytes for b in self.buffers)

    def _plan(self,r):
        cfg=self.config;c=cfg.width;ws=self.workspaces[r];plan=[];final_hidden=None
        scheduled_rows=r;control=self.control
        last_only=self.prefill_last and r==128
        hidden=ws['hidden'];other=ws.get('hidden2',hidden)
        def add(kind,p,*args):
            if self.provider=='webgpu':p=webgpu_parameters(kind,p,self.webgpu_profile)
            executable=self.kernels[identity(kind,p)]
            values,symbols,launch=executable._bind(args,{},include_outputs=True)
            plan.append((executable,BoundCall(self.device,executable.manifest,values,symbols,launch,validated=True)))
        def inputs(kind,p,x):
            selected=webgpu_parameters(kind,p,self.webgpu_profile) if self.provider=='webgpu' else p
            if not selected.get('q16'):return (x,)
            buffers=ws['q16'][p['k']]
            add('quantize_q16',dict(r=p['r'],k=p['k'],sg=True),x,*buffers)
            return buffers
        def linear(name,x,out,rows=None):
            rows=r if rows is None else rows
            info=self.gguf.tensors[name];o,k=info.shape
            p=dict(r=rows,k=k,o=o,type=info.type,**projection_tile(rows,o))
            add('linear',p,*inputs('linear',p,x),self.weights[name],out)
        def rms(name,x,out,rows=None):
            rows=r if rows is None else rows
            add('rms',dict(r=rows,c=c,eps=cfg.epsilon),x,self.weights[name],out)
        add('embedding',dict(r=r,c=c,v=cfg.vocab,type=self.gguf.tensors['token_embd.weight'].type),ws['tokens'],self.weights['token_embd.weight'],hidden)
        for i,kind in enumerate(cfg.layers):
            prefix=f'blk.{i}.'
            rms(prefix+'attn_norm.weight',hidden,ws['normal'])
            if kind=='conv':
                linear(prefix+'shortconv.in_proj.weight',ws['normal'],ws['project'])
                add('conv',dict(r=r,c=c),ws['project'],self.weights[prefix+'shortconv.conv.weight'],self.states[i],ws['convout'],control)
                linear(prefix+'shortconv.out_proj.weight',ws['convout'],ws['mixed'])
            else:
                tail_attention=self.prefill_tail and scheduled_rows==128 and i==len(cfg.layers)-3
                for name in (('k','v') if tail_attention else ('q','k','v')):
                    linear(prefix+'attn_'+name+'.weight',ws['normal'],ws[name])
                kc,vc=self.caches[i]
                p=dict(r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=self.capacity,eps=cfg.epsilon,theta=cfg.theta)
                if self.provider=='webgpu':
                    if tail_attention:
                        add('kvnorm',p,ws['k'],ws['v'],self.weights[prefix+'attn_k_norm.weight'],kc,vc,control)
                        add('prefill_tail',dict(r=r,c=c,t=self.prefill_tail),ws['normal'],ws['tail']['normal'],control,ws['tail_control'])
                        add('prefill_tail',dict(r=r,c=c,t=self.prefill_tail),hidden,ws['tail']['hidden'],control,ws['tail_control'])
                        control=ws['tail_control'];ws=ws['tail'];r=self.prefill_tail
                        hidden,other=ws['hidden'],ws['hidden2']
                        linear(prefix+'attn_q.weight',ws['normal'],ws['q'])
                        add('qnorm',dict(p,r=r),ws['q'],self.weights[prefix+'attn_q_norm.weight'],ws['qo'],control)
                    else:
                        add('qnorm',p,ws['q'],self.weights[prefix+'attn_q_norm.weight'],ws['qo'],control)
                        add('kvnorm',p,ws['k'],ws['v'],self.weights[prefix+'attn_k_norm.weight'],kc,vc,control)
                else:
                    add('qkv',p,ws['q'],ws['k'],ws['v'],self.weights[prefix+'attn_q_norm.weight'],self.weights[prefix+'attn_k_norm.weight'],ws['qo'],kc,vc,self.control)
                attention_input=ws['qo']
                fused=(self.provider=='webgpu' and r==1 and
                    ((self.webgpu_profile=='decode_fused' and (cfg.heads,cfg.kv_heads,cfg.head_dim)==(16,8,64)) or
                     (self.webgpu_profile in ('quant_searched','prefill_q16','prefill_mixed') and (cfg.heads,cfg.kv_heads,cfg.head_dim)==(32,8,64) and QUANT_ATTENTION.get('fused_scores'))))
                if self.provider=='webgpu' and self.webgpu_profile in SUBGROUP_PROFILES and r==1 and not fused:
                    add('attention_scores',dict(r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=self.capacity),ws['qo'],kc,ws['scores'],self.control)
                    attention_input=ws['scores']
                arguments=(attention_input,vc,ws['attn'],control) if attention_input is not ws['qo'] else (attention_input,kc,vc,ws['attn'],control)
                add('attention',dict(r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=self.capacity),*arguments)
                linear(prefix+'attn_output.weight',ws['attn'],ws['mixed'])
            if self.provider=='webgpu' and r==1:
                add('add_rms',dict(r=r,c=c,eps=cfg.epsilon),hidden,ws['mixed'],other,self.weights[prefix+'ffn_norm.weight'],ws['normal'])
                hidden,other=other,hidden
            else:
                add('add',dict(r=r,c=c),hidden,ws['mixed'],other)
                hidden,other=other,hidden
                if last_only and i==len(cfg.layers)-1:
                    add('last',dict(r=r,c=c),hidden,ws['last'],control)
                    rms(prefix+'ffn_norm.weight',ws['last'],ws['last_normal'],rows=1)
                    gate,up=(self.gguf.tensors[prefix+'ffn_'+name+'.weight'] for name in ('gate','up'))
                    add('ffn',dict(r=1,k=c,o=cfg.ff,type=gate.type),ws['last_normal'],
                        self.weights[gate.name],self.weights[up.name],ws['last_activated'])
                    info=self.gguf.tensors[prefix+'ffn_down.weight'];o,k=info.shape
                    add('linear_add',dict(r=1,k=k,o=o,type=info.type),ws['last_activated'],
                        self.weights[info.name],ws['last'],ws['last_output'])
                    final_hidden=ws['last_output']
                    break
                rms(prefix+'ffn_norm.weight',hidden,ws['normal'])
            gate,up=(self.gguf.tensors[prefix+'ffn_'+name+'.weight'] for name in ('gate','up'))
            if self.provider=='webgpu' and paired_ffn(r,gate,up,self.webgpu_profile):
                p=dict(r=r,k=c,o=cfg.ff,type=gate.type,**projection_tile(r,cfg.ff))
                add('ffn',p,*inputs('ffn',p,ws['normal']),self.weights[gate.name],self.weights[up.name],ws['activated'])
            else:
                linear(prefix+'ffn_gate.weight',ws['normal'],ws['gate']);linear(prefix+'ffn_up.weight',ws['normal'],ws['up'])
                add('swiglu',dict(r=r,c=cfg.ff),ws['gate'],ws['up'],ws['activated'])
            if self.provider=='webgpu' and r==1:
                info=self.gguf.tensors[prefix+'ffn_down.weight'];o,k=info.shape
                add('linear_add',dict(r=r,k=k,o=o,type=info.type),ws['activated'],self.weights[info.name],hidden,other)
            else:
                linear(prefix+'ffn_down.weight',ws['activated'],ws['mixed'])
                add('add',dict(r=r,c=c),hidden,ws['mixed'],other)
            hidden,other=other,hidden
        if final_hidden is None:
            if self.provider=='webgpu' and r==1:final_hidden=hidden
            else:
                add('last',dict(r=r,c=c),hidden,ws['last'],control)
                final_hidden=ws['last']
        rms('token_embd_norm.weight',final_hidden,ws['final'],rows=1)
        linear('output.weight' if 'output.weight' in self.weights else 'token_embd.weight',ws['final'],self.logits,rows=1)
        add('advance',dict(r=scheduled_rows),self.control)
        return plan

    def _submit(self,plan):
        for executable,call in plan:self.device._launch(executable,call)

    def _write(self,buffer,host):
        host=np.ascontiguousarray(host,dtype=buffer.dtype)
        if host.nbytes!=buffer.nbytes:raise ValueError('host write shape mismatch')
        self.device._check();buffer._check()
        if self.provider=='webgpu':self.device.write(buffer,host.reshape(buffer.shape))
        else:self.device.driver.call('cuMemcpyHtoD_v2',buffer.pointer,host.ctypes.data,host.nbytes)

    def reset(self):
        if self.closed:raise RuntimeError('LFM2 is closed')
        if self.provider!='webgpu':self.device.synchronize()
        self.position=0
        self._write(self.control,np.array([0,1],np.int32))
        for state in self.states.values():self._write(state,np.zeros(state.shape,np.float32))

    def forward(self,tokens,*,read=True,_greedy=False):
        if self.closed:raise RuntimeError('LFM2 is closed')
        tokens=np.asarray(tokens)
        if tokens.ndim!=1 or not tokens.size or tokens.dtype.kind not in 'iu':raise ValueError('requires nonempty integer token IDs')
        if np.any(tokens<0) or np.any(tokens>=self.config.vocab):raise ValueError('token ID outside vocabulary')
        if self.position+len(tokens)>self.context:raise ValueError('context capacity exceeded')
        result=None
        start=0
        while start<len(tokens):
            remaining=len(tokens)-start
            prefill=[r for r in self.rows if r>1]
            available=[r for r in prefill if r<=remaining]
            chunk=max(available) if available else min(prefill)
            values=tokens[start:start+chunk];r=1 if len(values)==1 else chunk
            host=np.zeros(r,np.int32);host[:len(values)]=values
            if self.provider!='webgpu':self.device.synchronize()
            self._write(self.workspaces[r]['tokens'],host)
            self._write(self.control,np.array([self.position,len(values)],np.int32))
            if self.provider=='webgpu':
                last=start+len(values)==len(tokens)
                output=self.workspaces[1]['tokens'] if _greedy else self.logits
                executable=(self.prepared_states[r] if not last and r in self.prepared_states
                            else (self.greedy if _greedy and last else self.prepared)[r])
                result=executable.launch(readback=output if read and last else None)
            elif self.graphs_enabled:self.graphs[r].launch()
            else:self._submit(self.plans[r])
            self.position+=len(values)
            start+=len(values)
        return result if self.provider=='webgpu' else self.logits.to_numpy() if read else None

    def generate(self,prompt,*,max_tokens=128,chat=True,gpu_greedy=True):
        if type(max_tokens)!=int or max_tokens<1:raise ValueError('max_tokens must be positive')
        tokens=self.tokenizer.chat(prompt) if chat else self.tokenizer.encode(prompt)
        if len(tokens)+max_tokens>self.context:raise ValueError('generation exceeds context capacity')
        if self.provider=='webgpu' and gpu_greedy:
            self.reset();next_token=self.forward(tokens,_greedy=True);generated=[]
            for step in range(max_tokens):
                token=int(next_token[0]);generated.append(token)
                if token==self.tokenizer.eos:break
                next_token=self.greedy[1].launch(readback=self.workspaces[1]['tokens'] if step+1<max_tokens else None);self.position+=1
            return {'prompt_tokens':tokens,'generated_tokens':generated,'text':self.tokenizer.decode(generated)}
        self.reset();logits=self.forward(tokens);generated=[]
        for _ in range(max_tokens):
            token=int(np.argmax(logits));generated.append(token)
            if token==self.tokenizer.eos:break
            logits=self.forward([token])
        return {'prompt_tokens':tokens,'generated_tokens':generated,'text':self.tokenizer.decode(generated)}

    def close(self):
        if self.closed:return
        self.device.synchronize()
        for graph in self.graphs.values():graph.close()
        for plan in self.prepared.values():plan.close()
        for plan in self.prepared_states.values():plan.close()
        for plan in self.greedy.values():plan.close()
        for kernel in self.kernels.values():kernel.release()
        for buffer in self.buffers:buffer.release()
        self.closed=True

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
