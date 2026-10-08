"""Pinned upstream nanoGPT and an explicit Tensor numerical adapter.

Torch retains module/parameter identity, views, concatenation, checkpointing and
shared-gradient accumulation. Dropout must be zero. Tensor computes projections,
embedding, LayerNorm, GELU, residuals, attention and cross entropy.
"""
import importlib.util
import inspect
import textwrap
from pathlib import Path
from types import MethodType
import torch

UPSTREAM_SHA='3adf61e154c3fe3fca428ad6bc3818b27a3b8291'
UPSTREAM_URL='https://github.com/karpathy/nanoGPT'
p=Path(__file__).parent/'third_party/nanogpt/model.py'
spec=importlib.util.spec_from_file_location('nanogpt_upstream',p)
upstream=importlib.util.module_from_spec(spec);spec.loader.exec_module(upstream)


def adapt(model,ops):
 if model.config.dropout!=0:raise ValueError('Tensor nanoGPT adapter requires dropout=0')
 def linear(self,x):return ops.linear(x,self.weight,self.bias)
 def embedding(self,x):return ops.embedding(x,self.weight)
 def ln(self,x):return ops.layer_norm(x,self.weight,self.bias)
 def gelu(self,x):return ops.gelu(x)
 def attention(self,x):
  b,s,w=x.shape
  q,k,v=self.c_attn(x).split(self.n_embd,dim=2)
  q,k,v=[z.reshape(b,s,self.n_head,w//self.n_head).transpose(1,2).contiguous() for z in (q,k,v)]
  y=ops.attention(q,k,v,causal=True).transpose(1,2).contiguous().reshape(b,s,w)
  return self.c_proj(y)
 def block(self,x):
  x=ops.add(x,self.attn(self.ln_1(x)))
  return ops.add(x,self.mlp(self.ln_2(x)))
 def forward(self,idx,targets=None):
  b,s=idx.shape
  if s>self.config.block_size:raise ValueError('context exceeds block size')
  pos=torch.arange(s,device=idx.device)
  tok=self.transformer.wte(idx);pos=self.transformer.wpe(pos)[None].expand(b,-1,-1).contiguous()
  x=ops.add(tok,pos)
  for block in self.transformer.h:x=block(x)
  x=self.transformer.ln_f(x)
  logits=self.lm_head(x if targets is not None else x[:,-1:,:])
  loss=None if targets is None else ops.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),ignore_index=-1)
  return logits,loss
 for m in model.modules():
  fn=None
  if isinstance(m,torch.nn.Linear):fn=linear
  elif isinstance(m,torch.nn.Embedding):fn=embedding
  elif isinstance(m,upstream.LayerNorm):fn=ln
  elif isinstance(m,torch.nn.GELU):fn=gelu
  elif isinstance(m,upstream.CausalSelfAttention):fn=attention
  elif isinstance(m,upstream.Block):fn=block
  if fn:m.forward=MethodType(fn,m)
 model.forward=MethodType(forward,model)
 model.tensor_ops=ops
 assert model.lm_head.weight is model.transformer.wte.weight
 return model


def make(ops=None,**config):
 model=upstream.GPT(upstream.GPTConfig(**config)).cuda()
 return adapt(model,ops) if ops else model


def graph_compatible(model):
 """Use a basic last-position slice; the upstream CPU-list index cannot capture.

 The vendored upstream source stays unchanged. This single equivalent slice
 replacement retains its complete forward body and all standard Torch modules.
 """
 source=textwrap.dedent(inspect.getsource(upstream.GPT.forward))
 original='x[:, [-1], :]'
 assert source.count(original)==1
 namespace=dict(vars(upstream))
 exec(source.replace(original,'x[:, -1:, :]'),namespace)
 model.forward=MethodType(namespace['forward'],model)
 return model
