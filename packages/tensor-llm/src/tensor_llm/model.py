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


def paired_ffn(rows,gate,up):
    # Quantized decode benefits from shared unpack/input work; the measured
    # F16 decode pair loses occupancy. Prefill benefits from register reuse.
    return gate.type==up.type and (rows>1 or gate.type not in (0,1))


# A full-width output tile doubles the work per workgroup and halves the launch
# count. It only pays while enough column tiles remain to fill the device:
# measured on the 2.6B FFN shape (o=10752), while the narrower ffn_down
# (o=2048) regresses by 12%. The guard is therefore on column count, not depth.
WIDE_TILE_COLUMNS=5120
WIDE_TILE=(32,32,64)

def projection_tile(rows,columns):
    return {'tile':WIDE_TILE} if rows>1 and columns>=WIDE_TILE_COLUMNS else {}


def requirements(gguf,capacity,rows=(1,128),*,provider='cuda',webgpu_profile='portable'):
    cfg=Config.from_gguf(gguf);c=cfg.width
    values={}
    if webgpu_profile not in ('portable','subgroup'):raise ValueError('unsupported WebGPU kernel profile')
    def add(kind,**p):
        if provider=='webgpu' and webgpu_profile=='subgroup' and kind in ('linear','ffn','attention','attention_scores','rms','add_rms'):
            p['sg']=True
        values[identity(kind,p)]=(kind,p)
    for r in rows:
        for info in gguf.tensors.values():
            if len(info.shape)==2 and not info.name.endswith('conv.weight'):
                o,k=info.shape
                if info.name=='token_embd.weight':
                    add('embedding',r=r,c=c,v=cfg.vocab,type=info.type)
                    add('linear',r=1,k=k,o=o,type=info.type)
                elif info.name=='output.weight':add('linear',r=1,k=k,o=o,type=info.type)
                else:add('linear',r=r,k=k,o=o,type=info.type,**projection_tile(r,o))
        add('rms',r=r,c=c,eps=cfg.epsilon)
        if provider=='webgpu':
            if r==1:add('add_rms',r=r,c=c,eps=cfg.epsilon)
            for i in range(len(cfg.layers)):
                gate,up=(gguf.tensors[f'blk.{i}.ffn_{name}.weight'] for name in ('gate','up'))
                if paired_ffn(r,gate,up):add('ffn',r=r,k=c,o=cfg.ff,type=gate.type,**projection_tile(r,cfg.ff))
        add('add',r=r,c=c);add('swiglu',r=r,c=cfg.ff);add('conv',r=r,c=c)
        for kind in (('qnorm','kvnorm') if provider=='webgpu' else ('qkv',)):
            add(kind,r=r,h=cfg.heads,kh=cfg.kv_heads,d=cfg.head_dim,cap=capacity,eps=cfg.epsilon,theta=cfg.theta)
        if provider=='webgpu' and webgpu_profile=='subgroup' and r==1:
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
        if self.rows != ((1,32) if self.provider=='webgpu' else (1,128)) or self.capacity % 64:
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
            if self.provider=='webgpu' and self.webgpu_profile=='subgroup' and r==1:
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
            if self.provider=='webgpu' and self.webgpu_profile=='subgroup' and kind in ('linear','ffn','attention','attention_scores','rms','add_rms'):
                p={**p,'sg':True}
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
                if self.provider=='webgpu' and self.webgpu_profile=='subgroup' and r==1:
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
            if self.provider=='webgpu' and paired_ffn(r,gate,up):
                add('ffn',dict(r=r,k=c,o=cfg.ff,type=gate.type,**projection_tile(r,cfg.ff)),ws['normal'],self.weights[gate.name],self.weights[up.name],ws['activated'])
            else:
                linear(prefix+'ffn_gate.weight',ws['normal'],ws['gate']);linear(prefix+'ffn_up.weight',ws['normal'],ws['up'])
                add('swiglu',dict(r=r,c=cfg.ff),ws['gate'],ws['up'],ws['activated'])
            linear(prefix+'ffn_down.weight',ws['activated'],ws['mixed'])
            add('add',dict(r=r,c=c),hidden,ws['mixed'],other)
            hidden,other=other,hidden
        add('last',dict(r=r,c=c),hidden,ws['last'],self.control)
        rms('token_embd_norm.weight',ws['last'],ws['final'],rows=1)
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
        chunk=max(self.rows)
        for start in range(0,len(tokens),chunk):
            values=tokens[start:start+chunk];r=1 if len(values)==1 else chunk
            host=np.zeros(r,np.int32);host[:len(values)]=values
            if self.provider!='webgpu':self.device.synchronize()
            self._write(self.workspaces[r]['tokens'],host)
            self._write(self.control,np.array([self.position,len(values)],np.int32))
            if self.provider=='webgpu':
                (self.greedy if _greedy and start+len(values)==len(tokens) else self.prepared)[r].launch()
            elif self.graphs_enabled:self.graphs[r].launch()
            else:self._submit(self.plans[r])
            self.position+=len(values)
        return self.logits.to_numpy() if read else None

    def generate(self,prompt,*,max_tokens=128,chat=True,gpu_greedy=True):
        if type(max_tokens)!=int or max_tokens<1:raise ValueError('max_tokens must be positive')
        tokens=self.tokenizer.chat(prompt) if chat else self.tokenizer.encode(prompt)
        if len(tokens)+max_tokens>self.context:raise ValueError('generation exceeds context capacity')
        if self.provider=='webgpu' and gpu_greedy:
            self.reset();self.forward(tokens,read=False,_greedy=True);generated=[]
            for _ in range(max_tokens):
                token=int(self.workspaces[1]['tokens'].to_numpy()[0]);generated.append(token)
                if token==self.tokenizer.eos:break
                self.greedy[1].launch();self.position+=1
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
