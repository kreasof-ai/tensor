"""Small causal LLT with differentiable projection folding and full residuals.

Absolute learned positions deliberately keep position handling identical between
baselines. `layerwise` is an MLA-style control whose per-layer latents are computed
from initial embeddings. It is not a faithful YOCO or DeepSeek reproduction.
"""
from dataclasses import dataclass, asdict
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class Config:
    kind: str = 'llt'
    width: int = 128
    heads: int = 4
    layers: int = 2
    loops: int = 4
    rank: int = 32
    vocab: int = 65
    max_seq: int = 8192
    untied: bool = False
    gelu: str = 'tanh'


def norm(x):
    z = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
    return (z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.q = nn.Linear(c.width, c.width, bias=False)
        self.o = nn.Linear(c.width, c.width, bias=False)
        self.w1 = nn.Linear(c.width, c.width * 4, bias=False)
        self.w2 = nn.Linear(c.width * 4, c.width, bias=False)
        if c.kind == 'naive':
            self.k = nn.Linear(c.width, c.width, bias=False)
            self.v = nn.Linear(c.width, c.width, bias=False)
        else:
            self.uk = nn.Parameter(torch.empty(c.heads, c.width // c.heads, c.rank))
            self.uv = nn.Parameter(torch.empty(c.heads, c.width // c.heads, c.rank))
            nn.init.normal_(self.uk, std=c.rank ** -.5)
            nn.init.normal_(self.uv, std=c.rank ** -.5)

    def folds(self, c):
        fq = (self.uk.transpose(1, 2) @ self.q.weight.reshape(c.heads, c.width // c.heads, c.width)).flatten(0, 1)
        fo = torch.einsum('mhd,hdr->mhr', self.o.weight.reshape(c.width, c.heads, -1), self.uv).flatten(1)
        return fq, fo


class Transformer(nn.Module):
    def __init__(self, c):
        super().__init__()
        assert c.width % c.heads == 0 and c.kind in ('naive', 'llt', 'layerwise')
        self.c = c
        self.token = nn.Embedding(c.vocab, c.width)
        self.position = nn.Embedding(c.max_seq, c.width)
        count = c.layers * (c.loops if c.untied else 1)
        self.blocks = nn.ModuleList([Block(c) for _ in range(count)])
        if c.kind != 'naive':
            self.down = nn.ModuleList([nn.Linear(c.width, c.rank, bias=False)
                                       for _ in range(count if c.kind == 'layerwise' else 1)])
        self.output = nn.Linear(c.width, c.vocab, bias=False)
        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.normal_(mod.weight, std=.02)
            if isinstance(mod, nn.Embedding):
                nn.init.normal_(mod.weight, std=.02)
        # Use the same stabilization rule for all recurrence depths.
        for b in self.blocks:
            b.o.weight.data.div_(math.sqrt(2 * c.layers * c.loops))
            b.w2.weight.data.div_(math.sqrt(2 * c.layers * c.loops))

    def initial(self, tokens, offset=0):
        pos = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        return self.token(tokens) + self.position(pos)

    def forward(self, tokens, policy='none', attention=None, return_cache=False, unfolded=False):
        c = self.c
        x = self.initial(tokens)
        h, d = c.heads, c.width // c.heads
        if c.kind != 'naive':
            latents = [down(norm(x)).unsqueeze(1).contiguous() for down in self.down]
            folds = [b.folds(c) for b in self.blocks] if not unfolded else None
        caches = []
        def attend(q, k, v):
            if attention:
                return attention(q, k, v, True, d ** -.5)
            return F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=d ** -.5)

        def step(state, loop):
            for layer in range(c.layers):
                idx = loop * c.layers + layer if c.untied else layer
                b = self.blocks[idx]
                z = norm(state)
                if c.kind == 'naive':
                    q, k, v = [linear(z).reshape(*z.shape[:2], h, d).transpose(1, 2) for linear in (b.q, b.k, b.v)]
                    if return_cache:
                        caches.append((k.contiguous(), v.contiguous()))
                    a = attend(q, k, v).transpose(1, 2).reshape(*z.shape[:2], c.width)
                    delta = b.o(a)
                else:
                    latent = latents[idx if c.kind == 'layerwise' else 0]
                    if unfolded:
                        q = b.q(z).reshape(*z.shape[:2], h, d).transpose(1, 2)
                        k = torch.einsum('bsr,hdr->bhsd', latent[:, 0], b.uk)
                        v = torch.einsum('bsr,hdr->bhsd', latent[:, 0], b.uv)
                        delta = b.o(attend(q, k, v).transpose(1, 2).reshape(*z.shape[:2], c.width))
                    else:
                        fq, fo = folds[idx]
                        q = F.linear(z, fq).reshape(*z.shape[:2], h, c.rank).transpose(1, 2)
                        kv = latent.expand(-1, h, -1, -1)
                        a = attend(q, kv, kv).transpose(1, 2).reshape(*z.shape[:2], h * c.rank)
                        delta = F.linear(a, fo)
                state = state + delta
                state = state + b.w2(F.gelu(b.w1(norm(state)), approximate=c.gelu))
            return state

        for loop in range(c.loops):
            if policy == 'loop':
                x = checkpoint(lambda state, loop=loop: step(state, loop), x, use_reentrant=False)
            else:
                x = step(x, loop)
        logits = self.output(norm(x))
        if return_cache:
            assert policy == 'none'
            return logits, caches if c.kind == 'naive' else latents
        return logits

    def prepare_decode(self, caches, offset, attention=None, cache_capacity=False):
        """Freeze folded inference weights outside token latency measurement.

        Returns a fixed one-token decode against a read-only historical prefix.
        One preallocated current-token slot is overwritten on each invocation;
        historical cache copies are outside token latency measurement.
        """
        c, h, d = self.c, self.c.heads, self.c.width // self.c.heads
        folds = [b.folds(c) for b in self.blocks] if c.kind != 'naive' else None
        def extend(old):
            return torch.cat((old,torch.zeros_like(old[:,:,:1,:])),dim=2)
        if cache_capacity:
            stores=caches
        else:
            stores=[tuple(extend(t) for t in pair) for pair in caches] if c.kind=='naive' else [extend(t) for t in caches]
        def decode(tokens):
            x = self.initial(tokens, offset)
            if c.kind != 'naive':
                current = [down(norm(x)).unsqueeze(1) for down in self.down]
                for old,new in zip(stores,current): old[:,:,offset:offset+1,:].copy_(new)
            for loop in range(c.loops):
                for layer in range(c.layers):
                    idx = loop * c.layers + layer if c.untied else layer
                    b, z = self.blocks[idx], norm(x)
                    if c.kind == 'naive':
                        q, k, v = [linear(z).reshape(tokens.shape[0], 1, h, d).transpose(1, 2).contiguous() for linear in (b.q, b.k, b.v)]
                        oldk, oldv = stores[loop * c.layers + layer]
                        oldk[:,:,offset:offset+1,:].copy_(k);oldv[:,:,offset:offset+1,:].copy_(v)
                        k,v=oldk,oldv
                    else:
                        fq, fo = folds[idx]
                        q = F.linear(z, fq).reshape(tokens.shape[0], 1, h, c.rank).transpose(1, 2).contiguous()
                        k = v = stores[idx if c.kind == 'layerwise' else 0]
                    if attention:
                        a = attention(q, k, v, False, d ** -.5)
                    else:
                        kk, vv = k.expand(-1, h, -1, -1), v.expand(-1, h, -1, -1)
                        a = F.scaled_dot_product_attention(q, kk, vv, scale=d ** -.5)
                    a = a.transpose(1, 2).reshape(tokens.shape[0], 1, -1)
                    x = x + (b.o(a) if c.kind == 'naive' else F.linear(a, fo))
                    x = x + b.w2(F.gelu(b.w1(norm(x)), approximate=c.gelu))
            return self.output(norm(x))
        return decode
