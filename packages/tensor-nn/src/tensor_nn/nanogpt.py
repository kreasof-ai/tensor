"""A bounded GPT training plan with explicit user-reachable manual backward.

Consumers import Tensor and NumPy only. Buffers are flattened; the immutable
configuration records their logical shapes. All training arithmetic is executed
by packaged Tensor kernels, including gradients, clipping and AdamW.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import ctypes as ct
import hashlib
import json
import math
import time
import numpy as np
from tensor.runtime.manual import ManualFunction
from tensor.runtime.abi import BoundCall
from .kernels import identity
from .provenance import implementation_hashes


@dataclass(frozen=True)
class GPTConfig:
    layers: int = 12
    heads: int = 12
    width: int = 768
    vocab: int = 50304
    batch: int = 2
    sequence: int = 512
    loss_scale: float = 128.0
    learning_rate: float = 0.0006
    clip: float = 1.0

    def __post_init__(self):
        for name in ('layers','heads','width','vocab','batch','sequence'):
            if type(getattr(self,name)) is not int or getattr(self,name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        if self.width % self.heads or self.width // self.heads % 8:
            raise ValueError('head dimensions must be a multiple of eight')
        if self.sequence % 16 or self.width % 16 or self.vocab % 16:
            raise ValueError('training dimensions must be multiples of sixteen')
        for name in ('loss_scale','learning_rate','clip'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be finite and positive')

    @property
    def rows(self): return self.batch*self.sequence

    @classmethod
    def diagnostic(cls): return cls(layers=1,heads=2,width=32,vocab=256,batch=1,sequence=16)


def parameter_shapes(config):
    c=config.width
    result={'token':(config.vocab,c),'position':(config.sequence,c)}
    for layer in range(config.layers):
        result.update({f'{layer}.ln1':(c,),f'{layer}.qkv':(3*c,c),f'{layer}.proj':(c,c),
                       f'{layer}.ln2':(c,),f'{layer}.fc':(4*c,c),f'{layer}.fcproj':(c,4*c)})
    result['ln_f']=(c,)
    return result


def initial_weights(config,seed=20260930):
    rng=np.random.default_rng(seed)
    result={}
    for name,shape in parameter_shapes(config).items():
        values=np.ones(shape,np.float32) if len(shape)==1 else rng.standard_normal(shape).astype(np.float32)*np.float32(0.02)
        if name.endswith(('.proj','.fcproj')): values/=np.float32(math.sqrt(2*config.layers))
        result[name]=values
    return result


def requirements(config):
    r,c=config.rows,config.width
    values={}
    def add(kind,**p): values[identity(kind,p)]=(kind,p)
    for i,o in [(c,3*c),(c,c),(c,4*c),(4*c,c),(c,config.vocab)]:
        kind='gemm_gelu' if (i,o)==(c,4*c) else 'gemm_residual' if (i,o) in ((c,c),(4*c,c)) else 'gemm'
        add(kind,batch=1,m=r,k=i,cols=o,tb=True)
        add('gemm',batch=1,m=r,k=o,cols=i)
        add('gemm',batch=1,m=o,k=r,cols=i,ta=True)
        add('cast',n=i*o)
    bh=config.batch*config.heads;s=config.sequence;d=c//config.heads
    add('gemm',batch=bh,m=s,k=d,cols=s,tb=True)
    add('gemm',batch=bh,m=s,k=s,cols=d)
    add('gemm',batch=bh,m=s,k=s,cols=d,ta=True)
    for kind in ('pack_qkv','unpack_qkv','merge_heads','split_heads'):
        add(kind,b=config.batch,s=s,c=c,h=config.heads)
    for kind in ('softmax','softmax_backward'): add(kind,s=s,bh=bh,d=d)
    for kind in ('embedding','embedding_backward'): add(kind,b=config.batch,s=s,c=c,v=config.vocab)
    for kind in ('norm','norm_backward','column_sum'): add(kind,r=r,c=c)
    add('add',n=r*c)
    for kind in ('gelu','gelu_backward'): add(kind,n=r*4*c)
    for kind in ('ce_parts','ce_loss','ce_backward'): add(kind,r=r,v=config.vocab)
    add('zero',n=s*c)
    shapes=parameter_shapes(config)
    total=sum(math.ceil(math.prod(shape)/1024) for shape in shapes.values())
    for name,shape in shapes.items():
        n=math.prod(shape)
        add('sumsq',n=n,total=total,scale=config.loss_scale)
        add('adamw',n=n,scale=config.loss_scale,decay=0.1 if len(shape)>1 else 0.0)
    add('sum_parts',n=total)
    add('clip',n=math.ceil(total/1024),limit=config.clip)
    return values


class KernelLibrary:
    """Verify a precompiled training bundle; cache validated fixed-buffer calls."""
    def __init__(self,directory,device):
        self.directory=Path(directory); self.device=device
        self.manifest=json.loads((self.directory/'training.json').read_text())
        if self.manifest['schema']!='tensor.manual-nanogpt.v2':
            raise ValueError('unsupported training bundle schema; rebuild for the reorganized Tensor/tensor-nn wheels')
        if device.info['provider']!='cuda':
            raise ValueError('manual nanoGPT currently requires a CUDA training bundle')
        self.config=GPTConfig(**self.manifest['config'])
        if implementation_hashes()!=self.manifest.get('implementation_sha256'):
            raise ValueError('install the matching Tensor and tensor-nn wheels for this training bundle')
        if hashlib.sha256((self.directory/'training.tpack').read_bytes()).hexdigest()!=self.manifest['module_sha256']:
            raise ValueError('training module checksum mismatch')
        self.loaded={};self.calls={}
        for record in self.manifest['kernels'].values():
            path=self.directory/record['artifact']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError('training artifact checksum mismatch')
        expected=requirements(self.config)
        if set(expected)!=set(self.manifest['kernels']):
            raise ValueError('incomplete training kernel coverage')
        self.launch_count=0

    def run(self,kind,p,*args):
        key=identity(kind,p)
        if key not in self.loaded:
            self.loaded[key]=self.device.load(self.directory/self.manifest['kernels'][key]['artifact'])
        executable=self.loaded[key]
        call_key=(key,tuple(id(v) if hasattr(v,'pointer') else v for v in args))
        if call_key not in self.calls:
            values,symbols,launch=executable._bind(args,{},include_outputs=True)
            self.calls[call_key]=BoundCall(self.device,executable.manifest,values,symbols,launch,validated=True)
        call=self.calls[call_key]
        executable._check()
        for buffer in call.storage:
            if hasattr(buffer,'_released') and (buffer._released or buffer._generation!=executable._generation):
                raise RuntimeError('training plan references a released buffer')
        self.device._launch(executable,call)
        self.launch_count+=1


@dataclass
class Parameter:
    weight: object
    half: object
    grad: object
    moment: object
    variance: object
    shape: tuple


class NanoGPT:
    """Static nanoGPT model, reusable storage and manually composed backward."""
    def __init__(self,directory,device,*,seed=20260930):
        started=time.perf_counter()
        previous_buffers=set(device._buffers)
        self.library=KernelLibrary(directory,device);self.device=device;self.config=self.library.config
        self.parameters={};self.operations=[];self.steps=0
        self.initial=initial_weights(self.config,seed)
        for name,array in self.initial.items():
            flat=array.reshape(-1)
            self.parameters[name]=Parameter(device.from_numpy(flat),device.from_numpy(flat.astype(np.float16)),
                device.empty(flat.shape),device.zeros(flat.shape),device.zeros(flat.shape),array.shape)
        cfg=self.config;r,c=cfg.rows,cfg.width
        self.blocks=[]
        for i in range(cfg.layers):
            self.blocks.append({'ln1':self.norm(f'{i}.ln1'),'qkv':self.linear(f'{i}.qkv'),
                'attn':self.attention(),'proj':self.linear(f'{i}.proj',residual=True),'ln2':self.norm(f'{i}.ln2'),
                'fc':self.linear(f'{i}.fc',activation=True),'fcproj':self.linear(f'{i}.fcproj',residual=True),
                'grad1':self.empty(r*c),'grad2':self.empty(r*c)})
        self.final_norm=self.norm('ln_f');self.head=self.linear('token')
        self.embedded=self.empty(r*c)
        self.tokens=device.empty((r,), 'int32');self.targets=device.empty((r,),'int32')
        self.loss=device.empty((r,));self.lse=device.empty((r,));self.loss_parts=device.empty((r*math.ceil(cfg.vocab/1024)*2,))
        self.logit_grad=self.empty(r*cfg.vocab);self.loss_grad=device.full((r,),cfg.loss_scale/r)
        total=sum(math.ceil(p.weight.shape[0]/1024) for p in self.parameters.values())
        self.norm_parts=device.empty((total,));self.norm_sums=device.empty((math.ceil(total/1024),));self.clip=device.empty((2,))
        self.function=ManualFunction(self._forward,self._backward,name='nanogpt')
        self.context=None
        self.allocated_bytes=sum(buffer.nbytes for buffer in device._buffers if buffer not in previous_buffers)
        self.prepare_seconds=time.perf_counter()-started

    def empty(self,n,dtype='float16'): return self.device.empty((n,),dtype)
    def run(self,kind,p,*args): self.library.run(kind,p,*args)
    def parameter(self,name): return self.parameters[name]
    def operation(self,forward,backward,name):
        op=ManualFunction(forward,backward,name=name);self.operations.append(op);return op

    def linear(self,name,*,activation=False,residual=False):
        param=self.parameter(name);o,i=param.shape;r=self.config.rows
        y=self.empty(r*o);dx=self.empty(r*i);dw=self.empty(o*i)
        pre=self.empty(r*o) if activation else None
        dpre=self.empty(r*o) if activation else None
        def forward(ctx,x,weight,*skip):
            ctx.save_for_backward(x,weight,param.half)
            kind='gemm_gelu' if activation else 'gemm_residual' if residual else 'gemm'
            extra=(pre,) if activation else skip if residual else ()
            self.run(kind,dict(batch=1,m=r,k=i,cols=o,tb=True),x,param.half,*extra,y)
            return y
        def backward(ctx,dy):
            x,_,half=ctx.saved_tensors
            original_gradient=dy
            if activation:
                self.run('gelu_backward',dict(n=r*o),pre,dy,dpre)
                dy=dpre
            self.run('gemm',dict(batch=1,m=r,k=o,cols=i),dy,half,dx)
            self.run('gemm',dict(batch=1,m=o,k=r,cols=i,ta=True),dy,x,dw)
            self.run('cast',dict(n=o*i),dw,param.grad)
            return (dx,param.grad,original_gradient) if residual else (dx,param.grad)
        return self.operation(forward,backward,name)

    def norm(self,name):
        r,c=self.config.rows,self.config.width;param=self.parameter(name)
        y=self.empty(r*c);normal=self.empty(r*c,'float32');inverse=self.empty(r,'float32')
        dx=self.empty(r*c);parts=self.empty(r*c,'float32')
        def forward(ctx,x,weight):
            self.run('norm',dict(r=r,c=c),x,weight,y,normal,inverse)
            ctx.save_for_backward(weight,normal,inverse)
            return y
        def backward(ctx,dy):
            weight,normal,inverse=ctx.saved_tensors
            self.run('norm_backward',dict(r=r,c=c),dy,weight,normal,inverse,dx,parts)
            self.run('column_sum',dict(r=r,c=c),parts,param.grad)
            return dx,param.grad
        return self.operation(forward,backward,name)

    def gelu(self,n):
        y=self.empty(n);dx=self.empty(n)
        def forward(ctx,x):
            ctx.save_for_backward(x);self.run('gelu',dict(n=n),x,y);return y
        def backward(ctx,dy):
            self.run('gelu_backward',dict(n=n),ctx.saved_tensors[0],dy,dx);return dx
        return self.operation(forward,backward,'gelu')

    def attention(self):
        cfg=self.config; b,s,c,h=cfg.batch,cfg.sequence,cfg.width,cfg.heads;d=c//h;n=b*s*c;bh=b*h
        pack=dict(b=b,s=s,c=c,h=h);soft=dict(s=s,bh=bh,d=d)
        q,k,v,y,dq,dk,dv,dyh=[self.empty(n) for _ in range(8)]
        scores,prob,dp,ds=[self.empty(bh*s*s) for _ in range(4)]
        saved=self.empty(bh*s*s,'float32');merged=self.empty(n);dx=self.empty(3*n)
        def forward(ctx,x):
            self.run('pack_qkv',pack,x,q,k,v)
            self.run('gemm',dict(batch=bh,m=s,k=d,cols=s,tb=True),q,k,scores)
            self.run('softmax',soft,scores,prob,saved)
            self.run('gemm',dict(batch=bh,m=s,k=s,cols=d),prob,v,y)
            self.run('merge_heads',pack,y,merged)
            ctx.save_for_backward(q,k,v,prob,saved)
            return merged
        def backward(ctx,dy):
            q,k,v,prob,saved=ctx.saved_tensors
            self.run('split_heads',pack,dy,dyh)
            self.run('gemm',dict(batch=bh,m=s,k=d,cols=s,tb=True),dyh,v,dp)
            self.run('gemm',dict(batch=bh,m=s,k=s,cols=d,ta=True),prob,dyh,dv)
            self.run('softmax_backward',soft,dp,saved,ds)
            self.run('gemm',dict(batch=bh,m=s,k=s,cols=d),ds,k,dq)
            self.run('gemm',dict(batch=bh,m=s,k=s,cols=d,ta=True),ds,q,dk)
            self.run('unpack_qkv',pack,dq,dk,dv,dx)
            return dx
        return self.operation(forward,backward,'causal_attention')

    def _forward(self,ctx,tokens):
        cfg=self.config;r,c=cfg.rows,cfg.width
        self.run('embedding',dict(b=cfg.batch,s=cfg.sequence,c=c,v=cfg.vocab),tokens,
                 self.parameter('token').half,self.parameter('position').half,self.embedded)
        x=self.embedded;tape=[]
        def call(op,*args):
            y,context=op.forward(*args);tape.append((op,context));return y
        for i,block in enumerate(self.blocks):
            ln1=call(block['ln1'],x,self.parameter(f'{i}.ln1').weight)
            qkv=call(block['qkv'],ln1,self.parameter(f'{i}.qkv').weight)
            attended=call(block['attn'],qkv)
            residual=call(block['proj'],attended,self.parameter(f'{i}.proj').weight,x)
            ln2=call(block['ln2'],residual,self.parameter(f'{i}.ln2').weight)
            activated=call(block['fc'],ln2,self.parameter(f'{i}.fc').weight)
            x=call(block['fcproj'],activated,self.parameter(f'{i}.fcproj').weight,residual)
        normalized=call(self.final_norm,x,self.parameter('ln_f').weight)
        logits=call(self.head,normalized,self.parameter('token').weight)
        ctx.save_for_backward(tokens);ctx.metadata['tape']=tape
        return logits

    def _backward(self,ctx,gradient):
        cfg=self.config;tape=list(ctx.metadata['tape'])
        def reverse(dy):
            op,context=tape.pop();return op.backward(context,dy)[0]
        dy=reverse(gradient);dy=reverse(dy)
        for block in reversed(self.blocks):
            branch=reverse(dy) # fcproj
            branch=reverse(branch) # fused fc + gelu
            branch=reverse(branch) # ln2
            self.run('add',dict(n=cfg.rows*cfg.width),dy,branch,block['grad1'])
            dy1=block['grad1']
            branch=reverse(dy1) # attention projection
            branch=reverse(branch) # attention
            branch=reverse(branch) # qkv projection
            branch=reverse(branch) # ln1
            self.run('add',dict(n=cfg.rows*cfg.width),dy1,branch,block['grad2'])
            dy=block['grad2']
        assert not tape
        self.run('zero',dict(n=cfg.sequence*cfg.width),self.parameter('position').grad)
        self.run('embedding_backward',dict(b=cfg.batch,s=cfg.sequence,c=cfg.width,v=cfg.vocab),
            ctx.saved_tensors[0],dy,self.parameter('token').grad,self.parameter('position').grad)
        return None

    def forward_backward(self):
        cfg=self.config
        logits,self.context=self.function.forward(self.tokens)
        params=dict(r=cfg.rows,v=cfg.vocab)
        self.run('ce_parts',params,logits,self.loss_parts)
        self.run('ce_loss',params,logits,self.targets,self.loss_parts,self.lse,self.loss)
        self.run('ce_backward',params,logits,self.targets,self.lse,self.loss_grad,self.logit_grad)
        self.function.backward(self.context,self.logit_grad)
        self.context=None

    def update(self):
        cfg=self.config;offset=0
        for parameter in self.parameters.values():
            n=parameter.weight.shape[0]
            self.run('sumsq',dict(n=n,total=self.norm_parts.shape[0],scale=cfg.loss_scale),parameter.grad,self.norm_parts,offset)
            offset+=math.ceil(n/1024)
        self.run('sum_parts',dict(n=self.norm_parts.shape[0]),self.norm_parts,self.norm_sums)
        self.run('clip',dict(n=self.norm_sums.shape[0],limit=cfg.clip),self.norm_sums,self.clip)
        # Fail before mutating optimizer state or advancing its step counter.
        self.read_norm()
        self.steps+=1
        for param in self.parameters.values():
            self.run('adamw',dict(n=param.weight.shape[0],scale=cfg.loss_scale,decay=0.1 if len(param.shape)>1 else 0.0),
                param.weight,param.half,param.grad,param.moment,param.variance,self.clip,
                cfg.learning_rate,1-0.9**self.steps,1-0.95**self.steps)

    def step(self):
        self.forward_backward();self.update()

    def upload(self,buffer,value):
        value=np.ascontiguousarray(value,dtype=buffer.dtype).reshape(buffer.shape)
        buffer._check();self.device.synchronize()
        self.device.driver.call('cuMemcpyHtoD_v2',buffer.pointer,ct.c_void_p(value.ctypes.data),value.nbytes)
        self.device.driver.call('cuStreamSynchronize',None)

    def set_batch(self,tokens,targets):
        cfg=self.config
        for data in (tokens,targets):
            if np.asarray(data).shape!=(cfg.batch,cfg.sequence) or not np.issubdtype(np.asarray(data).dtype,np.integer):
                raise ValueError('token batches must match the training configuration')
            if np.any(np.asarray(data)<0) or np.any(np.asarray(data)>=cfg.vocab):
                raise ValueError('token ids must be within the vocabulary')
        self.upload(self.tokens,tokens);self.upload(self.targets,targets)

    def reset(self):
        for operation in [self.function,*self.operations]:
            if operation._pending is not None: operation._pending.discard()
        for name,param in self.parameters.items():
            initial=self.initial[name].reshape(-1)
            self.upload(param.weight,initial);self.upload(param.half,initial)
            self.upload(param.moment,np.zeros_like(initial));self.upload(param.variance,np.zeros_like(initial))
        self.context=None;self.steps=0

    def read_loss(self):
        losses=self.loss.to_numpy()
        if not np.isfinite(losses).all(): raise FloatingPointError('non-finite training loss')
        return float(np.mean(losses,dtype=np.float64))

    def read_norm(self):
        value=self.clip.to_numpy()
        if not np.isfinite(value).all(): raise FloatingPointError('non-finite gradients; optimizer update was skipped')
        return float(value[1])
