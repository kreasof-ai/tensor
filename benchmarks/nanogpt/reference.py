"""Independent PyTorch reference for Tensor's pinned nanoGPT training profile.

Architecture follows karpathy/nanoGPT/model.py: pre-LN causal GPT, exact GELU,
tied embedding/head weights, no biases and zero dropout. The explicit attention
path and precision policy make each manual derivative independently comparable.
"""

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from tensor_nn.nanogpt import initial_weights


class Reference(nn.Module):
    def __init__(self,config,*,compiled=False,attention='explicit',fused_optimizer=False):
        super().__init__();self.config=config;self.attention=attention
        weights=initial_weights(config)
        self.names=tuple(weights)
        self.weights=nn.ParameterDict(OrderedDict((name.replace('.','_'),nn.Parameter(torch.from_numpy(value.copy()).cuda()))
                                    for name,value in weights.items()))
        parameters=self.mapping()
        groups=[{'params':[p for p in parameters.values() if p.ndim>=2],'weight_decay':0.1},
                {'params':[p for p in parameters.values() if p.ndim<2],'weight_decay':0.0}]
        self.optimizer=torch.optim.AdamW(groups,lr=config.learning_rate,betas=(0.9,0.95),eps=1e-8,foreach=False if not fused_optimizer else None,fused=fused_optimizer)
        self.compute=torch.compile(self.forward,fullgraph=True,dynamic=False) if compiled else self.forward
        self.initial={name:p.detach().clone() for name,p in parameters.items()}
        self.loss=None

    def mapping(self):return OrderedDict((name,self.w(name)) for name in self.names)
    def w(self,name):return self.weights[name.replace('.','_')]
    def linear(self,x,name):return F.linear(x,self.w(name).half())
    def norm(self,x,name):return F.layer_norm(x.float(),(self.config.width,),self.w(name),None,1e-5).half()

    def forward(self,tokens,targets):
        cfg=self.config;b,s=tokens.shape;c=cfg.width;d=c//cfg.heads
        x=F.embedding(tokens,self.w('token')).half()+F.embedding(torch.arange(s,device=tokens.device),self.w('position')).half()
        mask=torch.arange(s,device=x.device)[None,:]>torch.arange(s,device=x.device)[:,None]
        for layer in range(cfg.layers):
            y=self.norm(x,f'{layer}.ln1')
            q,k,v=self.linear(y,f'{layer}.qkv').reshape(b,s,3,cfg.heads,d).permute(2,0,3,1,4).unbind(0)
            if self.attention=='sdpa':
                y=F.scaled_dot_product_attention(q,k,v,is_causal=True)
            else:
                scores=(q@k.transpose(-2,-1))*d**-0.5
                probability=torch.softmax(scores.float().masked_fill(mask,float('-inf')),dim=-1).half()
                y=probability@v
            y=y.transpose(1,2).contiguous().reshape(b,s,c)
            x=x+self.linear(y,f'{layer}.proj')
            y=self.norm(x,f'{layer}.ln2')
            y=F.gelu(self.linear(y,f'{layer}.fc').float(),approximate='none').half()
            x=x+self.linear(y,f'{layer}.fcproj')
        logits=self.linear(self.norm(x,'ln_f'),'token')
        return F.cross_entropy(logits.float().reshape(-1,cfg.vocab),targets.reshape(-1))

    def forward_backward(self,tokens,targets):
        self.optimizer.zero_grad(set_to_none=True)
        self.loss=self.compute(tokens,targets)
        (self.loss*self.config.loss_scale).backward()
        return self.loss

    def update(self):
        for parameter in self.weights.values():parameter.grad.div_(self.config.loss_scale)
        norm=torch.nn.utils.clip_grad_norm_(list(self.weights.values()),self.config.clip,error_if_nonfinite=True,foreach=False)
        self.optimizer.step()
        return norm

    def reset(self):
        with torch.no_grad():
            for name,parameter in self.mapping().items():parameter.copy_(self.initial[name])
        self.optimizer.state.clear();self.optimizer.zero_grad(set_to_none=True)
