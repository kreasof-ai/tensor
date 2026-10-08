"""Small absorbed LLT and naive-loop fixtures with an identical Torch reference.

FP32 master/residual state, BF16 GEMMs/attention, exact loop checkpoints, and
optional decoupled RoPE. PyTorch handles layout and tied-gradient accumulation.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


@dataclass
class Config:
    width:int=64
    heads:int=2
    rank:int=32
    layers:int=2
    loops:int=2
    vocab:int=256
    capacity:int=128
    rotary:int=0
    architecture:str='llt'
    checkpoint:bool=False


class Block(nn.Module):
    def __init__(self,c):
        super().__init__();w,h,d,r=c.width,c.heads,c.width//c.heads,c.rank
        self.n1=nn.Parameter(torch.ones(w));self.n2=nn.Parameter(torch.ones(w))
        self.q=nn.Parameter(torch.randn(h,d,w)*0.02)
        self.k=nn.Parameter(torch.randn(h,d,r if c.architecture=='llt' else w)*0.02)
        self.v=nn.Parameter(torch.randn(h,d,r if c.architecture=='llt' else w)*0.02)
        self.o=nn.Parameter(torch.randn(w,h*d)*0.02)
        self.up=nn.Parameter(torch.randn(4*w,w)*0.02);self.down=nn.Parameter(torch.randn(w,4*w)*0.02)
        if c.rotary:
            self.qpos=nn.Parameter(torch.randn(h*c.rotary,w)*0.02)
    def folds(self,model):
        c=model.config
        q=torch.cat([model.mm(self.k[h],self.q[h],ta=True) for h in range(c.heads)],dim=0)
        o=torch.cat([model.mm(self.o[:,h*(c.width//c.heads):(h+1)*(c.width//c.heads)].contiguous(),self.v[h]) for h in range(c.heads)],dim=1)
        return q,o


class Model(nn.Module):
    def __init__(self,c,ops=None):
        super().__init__();self.config,self.ops=c,ops
        self.tokens=nn.Parameter(torch.randn(c.vocab,c.width)*0.02)
        self.positions=nn.Parameter(torch.randn(c.capacity,c.width)*0.02)
        self.final=nn.Parameter(torch.ones(c.width));self.lm=nn.Parameter(torch.randn(c.vocab,c.width)*0.02)
        self.blocks=nn.ModuleList([Block(c) for _ in range(c.layers)])
        if c.architecture=='llt':
            self.base_norm=nn.Parameter(torch.ones(c.width));self.latent=nn.Parameter(torch.randn(c.rank,c.width)*0.02)
        if c.rotary: self.kpos=nn.Parameter(torch.randn((1 if c.architecture=='llt' else c.heads)*c.rotary,c.width)*0.02)

    def mm(self,x,y,ta=False,tb=False):
        if self.ops: return self.ops.matmul(x,y,transpose_a=ta,transpose_b=tb)
        return (x.T if ta else x)@(y.T if tb else y)
    def linear(self,x,w):
        return self.ops.linear(x,w) if self.ops else torch.nn.functional.linear(x,w)
    def norm(self,x,w):
        return self.ops.rms_norm(x,w) if self.ops else x*torch.rsqrt(x.square().mean(-1,keepdim=True)+1e-6)*w
    def add(self,x,y): return self.ops.add(x,y) if self.ops else x+y
    def embed(self,index,w): return self.ops.embedding(index,w) if self.ops else torch.nn.functional.embedding(index,w)
    def gelu(self,x): return self.ops.gelu(x) if self.ops else torch.nn.functional.gelu(x)
    def rotary(self,x,offset=0):
        if self.ops:return self.ops.rotary(x,offset=offset)
        half=x.shape[-1]//2
        theta=(torch.arange(x.shape[2],device=x.device)+offset)[:,None]*10000.**(-torch.arange(half,device=x.device)/half)
        cs=torch.cat((theta.cos(),theta.cos()),-1);ss=torch.cat((theta.sin(),theta.sin()),-1)
        return (x.float()*cs+torch.cat((-x[...,half:],x[...,:half]),-1).float()*ss).to(x.dtype)
    def attention(self,q,k,v,offset=0):
        scale=(self.config.width//self.config.heads)**-0.5
        if self.ops:return self.ops.attention(q,k,v,causal=True,query_offset=offset,scale=scale)
        mask=torch.arange(k.shape[2],device=q.device)[None,:]<=torch.arange(q.shape[2],device=q.device)[:,None]+offset
        return torch.nn.functional.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=scale,enable_gqa=k.shape[1]!=q.shape[1])

    def forward(self,index,target=None):
        c=self.config;b,s=index.shape
        x=self.add(self.embed(index,self.tokens),self.embed(torch.arange(s,device=index.device),self.positions)[None,:,:].expand(b,-1,-1).contiguous())
        latent=None;keypos=None
        if c.architecture=='llt':
            latent=self.linear(self.norm(x,self.base_norm),self.latent).reshape(b,s,1,c.rank).transpose(1,2).contiguous()
            folds=[block.folds(self) for block in self.blocks]
            if c.rotary:keypos=self.rotary(self.linear(x,self.kpos).reshape(b,s,1,c.rotary).transpose(1,2).contiguous())
        def loop(x,latent,keypos,*flat_folds):
            for l,block in enumerate(self.blocks):
                z=self.norm(x,block.n1)
                if c.architecture=='llt':
                    q=self.linear(z,flat_folds[2*l]).reshape(b,s,c.heads,c.rank).transpose(1,2).contiguous();v=latent;k=latent
                else:
                    d=c.width//c.heads
                    q=self.linear(z,block.q.reshape(c.width,c.width)).reshape(b,s,c.heads,d).transpose(1,2).contiguous()
                    k=self.linear(z,block.k.reshape(c.width,c.width)).reshape(b,s,c.heads,d).transpose(1,2).contiguous()
                    v=self.linear(z,block.v.reshape(c.width,c.width)).reshape(b,s,c.heads,d).transpose(1,2).contiguous()
                if c.rotary:
                    qp=self.rotary(self.linear(z,block.qpos).reshape(b,s,c.heads,c.rotary).transpose(1,2).contiguous())
                    kp=keypos if c.architecture=='llt' else self.rotary(self.linear(z,self.kpos).reshape(b,s,c.heads,c.rotary).transpose(1,2).contiguous())
                    q=torch.cat((q,qp),-1);k=torch.cat((k,kp),-1)
                a=self.attention(q,k,v).transpose(1,2).contiguous().reshape(b,s,-1)
                x=self.add(x,self.linear(a,flat_folds[2*l+1] if c.architecture=='llt' else block.o))
                x=self.add(x,self.linear(self.gelu(self.linear(self.norm(x,block.n2),block.up)),block.down))
            return x
        flat=tuple(t for pair in folds for t in pair) if c.architecture=='llt' else ()
        for _ in range(c.loops):
            x=checkpoint(loop,x,latent,keypos,*flat,use_reentrant=False) if c.checkpoint and torch.is_grad_enabled() else loop(x,latent,keypos,*flat)
        logits=self.linear(self.norm(x,self.final),self.lm)
        if target is None:return logits
        return self.ops.cross_entropy(logits.reshape(-1,c.vocab),target.reshape(-1)) if self.ops else torch.nn.functional.cross_entropy(logits.float().reshape(-1,c.vocab),target.reshape(-1))
