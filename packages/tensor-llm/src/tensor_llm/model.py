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
                 'prefill_unrolled','prefill_outer','prefill_chunked')
SUBGROUP_PROFILES=WEBGPU_PROFILES[1:]
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


def valid_rows(provider,rows,profile):
    if type(rows) is not tuple or not rows or any(type(r) is not int for r in rows):return False
    if provider=='cuda':return rows==(1,128)
    if provider!='webgpu':return False
    return rows==(1,32) or (profile in ('prefill_unrolled','prefill_outer','prefill_chunked')
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
    searched=(profile in ('decode_searched','decode_fused','prefill_unrolled','prefill_outer','prefill_chunked') and rows==1 and gate.type==1 and
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
        expected=requirements(self.gguf,self.capacity,self.rows,provider=self.provider,webgpu_profile=self.webgpu_profile)
        if set(expected)!=set(self.manifest['kernels']):raise ValueError('incomplete/incompatible LFM2 kernel coverage')
        for record in self.manifest['kernels'].values():
            path=(self.directory/record['artifact']).resolve()
            if not path.is_relative_to(self.directory.resolve()):
                raise ValueError('LFM2 artifact path escapes its bundle')
            with path.open('rb') as image:
                if hashlib.file_digest(image,'sha256').hexdigest()!=record['sha256']:
                    raise ValueError('LFM2 kernel checksum mismatch')
        self.kernels={};self.graphs={};self.plans={};self.buffers=[];self.prepared={};self.greedy={}
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
            if self.provider=='webgpu' and self.webgpu_profile in SUBGROUP_PROFILES and r==1:
                self.workspaces[r]['scores']=upload(np.zeros(self.config.heads*self.capacity,np.float32))
        for key,record in self.manifest['kernels'].items():
            path=self.directory/record['artifact']
            self.kernels[key]=device.load(path)
        for r in self.rows:self.plans[r]=self._plan(r)
        if self.provider=='webgpu':
            for r,plan in self.plans.items():self.prepared[r]=device.prepare_plan(plan)
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
        cfg=self.config;c=cfg.width;ws=self.workspaces[r];plan=[]
        hidden=ws['hidden'];other=ws.get('hidden2',hidden)
        def add(kind,p,*args):
            if self.provider=='webgpu':p=webgpu_parameters(kind,p,self.webgpu_profile)
            executable=self.kernels[identity(kind,p)]
            values,symbols,launch=executable._bind(args,{},include_outputs=True)
            plan.append((executable,BoundCall(self.device,executable.manifest,values,symbols,launch,validated=True)))
        def linear(name,x,out,rows=r):
            info=self.gguf.tensors[name];o,k=info.shape
            add('linear',dict(r=rows,k=k,o=o,type=info.type,**projection_tile(rows,o)),x,self.weights[name],out)
        def rms(name,x,out,rows=r):add('rms',dict(r=rows,c=c,eps=cfg.epsilon),x,self.weights[name],out)
        add('embedding',dict(r=r,c=c,v=cfg.vocab,type=self.gguf.tensors['token_embd.weight'].type),ws['tokens'],self.weights['token_embd.weight'],hidden)
        for i,kind in enumerate(cfg.layers):
            prefix=f'blk.{i}.'
            rms(prefix+'attn_norm.weight',hidden,ws['normal'])
            if kind=='conv':
                linear(prefix+'shortconv.in_proj.weight',ws['normal'],ws['project'])
                add('conv',dict(r=r,c=c),ws['project'],self.weights[prefix+'shortconv.conv.weight'],self.states[i],ws['convout'],self.control)
                linear(prefix+'shortconv.out_proj.weight',ws['convout'],ws['mixed'])
            else:
                for name in ('q','k','v'):linear(prefix+'attn_'+name+'.weight',ws['normal'],ws[name])
                kc,vc=self.caches[i]
                p=dict(r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=self.capacity,eps=cfg.epsilon,theta=cfg.theta)
                if self.provider=='webgpu':
                    add('qnorm',p,ws['q'],self.weights[prefix+'attn_q_norm.weight'],ws['qo'],self.control)
                    add('kvnorm',p,ws['k'],ws['v'],self.weights[prefix+'attn_k_norm.weight'],kc,vc,self.control)
                else:
                    add('qkv',p,ws['q'],ws['k'],ws['v'],self.weights[prefix+'attn_q_norm.weight'],self.weights[prefix+'attn_k_norm.weight'],ws['qo'],kc,vc,self.control)
                attention_input=ws['qo']
                fused=(self.provider=='webgpu' and self.webgpu_profile=='decode_fused' and r==1 and (cfg.heads,cfg.kv_heads,cfg.head_dim)==(16,8,64))
                if self.provider=='webgpu' and self.webgpu_profile in SUBGROUP_PROFILES and r==1 and not fused:
                    add('attention_scores',dict(r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=self.capacity),ws['qo'],kc,ws['scores'],self.control)
                    attention_input=ws['scores']
                arguments=(attention_input,vc,ws['attn'],self.control) if attention_input is not ws['qo'] else (attention_input,kc,vc,ws['attn'],self.control)
                add('attention',dict(r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=self.capacity),*arguments)
                linear(prefix+'attn_output.weight',ws['attn'],ws['mixed'])
            if self.provider=='webgpu' and r==1:
                add('add_rms',dict(r=r,c=c,eps=cfg.epsilon),hidden,ws['mixed'],other,self.weights[prefix+'ffn_norm.weight'],ws['normal'])
                hidden,other=other,hidden
            else:
                add('add',dict(r=r,c=c),hidden,ws['mixed'],other)
                hidden,other=other,hidden
                rms(prefix+'ffn_norm.weight',hidden,ws['normal'])
            gate,up=(self.gguf.tensors[prefix+'ffn_'+name+'.weight'] for name in ('gate','up'))
            if self.provider=='webgpu' and paired_ffn(r,gate,up,self.webgpu_profile):
                add('ffn',dict(r=r,k=c,o=cfg.ff,type=gate.type,**projection_tile(r,cfg.ff)),ws['normal'],self.weights[gate.name],self.weights[up.name],ws['activated'])
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
        if self.provider=='webgpu' and r==1:
            final_hidden=hidden
        else:
            add('last',dict(r=r,c=c),hidden,ws['last'],self.control)
            final_hidden=ws['last']
        rms('token_embd_norm.weight',final_hidden,ws['final'],rows=1)
        linear('output.weight' if 'output.weight' in self.weights else 'token_embd.weight',ws['final'],self.logits,rows=1)
        add('advance',dict(r=r),self.control)
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
                result=(self.greedy if _greedy and last else self.prepared)[r].launch(readback=output if read and last else None)
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
        for plan in self.greedy.values():plan.close()
        for kernel in self.kernels.values():kernel.release()
        for buffer in self.buffers:buffer.release()
        self.closed=True

    def __enter__(self):return self
    def __exit__(self,*exc):self.close()
