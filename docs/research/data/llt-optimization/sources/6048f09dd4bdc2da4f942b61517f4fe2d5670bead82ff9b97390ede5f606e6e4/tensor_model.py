"""Explicit Tensor backend for the existing LLT geometry, with batched folds.

FP32 master weights/residuals; BF16 projections and attention; unweighted RMSNorm
(epsilon 1e-5); exact GELU. Layout, autograd/checkpoint scheduling and tied-gradient
accumulation remain PyTorch control work. No implicit numerical fallback.
"""
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from tensor_torch.llt import KVCache

from model import Transformer, norm


@dataclass
class TorchCache:
    keys: torch.Tensor
    values: torch.Tensor
    length: int = 0

    @property
    def nbytes(self):
        return self.keys.numel() * self.keys.element_size() + (
            0 if self.keys is self.values else self.values.numel() * self.values.element_size()
        )

    def append(self, k, v=None):
        v = k if v is None else v
        end = self.length + k.shape[2]
        if end > self.keys.shape[2]:
            raise ValueError('cache capacity exceeded')
        self.keys[:, :, self.length:end].copy_(k)
        if self.keys is not self.values:
            self.values[:, :, self.length:end].copy_(v)
        self.length = end


class BackendTransformer(Transformer):
    def __init__(self, c, ops=None):
        if c.gelu != 'none':
            raise ValueError('matched Tensor profile requires exact GELU')
        super().__init__(c)
        self.ops = ops
        self.register_buffer('norm_weight', torch.ones(c.width), persistent=False)

    def linear(self, x, weight):
        return self.ops.linear(x, weight) if self.ops else F.linear(x, weight)

    def normalize(self, x):
        return self.ops.rms_norm(x, self.norm_weight, eps=1e-5) if self.ops else norm(x)

    def add(self, x, y):
        return self.ops.add(x, y) if self.ops else x + y

    def activate(self, x):
        return self.ops.gelu(x) if self.ops else F.gelu(x, approximate='none')

    def initial(self, tokens, offset=0):
        pos = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        if self.ops:
            token = self.ops.embedding(tokens, self.token.weight)
            position = self.ops.embedding(pos, self.position.weight)
            position = position[None].expand(tokens.shape[0], -1, -1).contiguous()
            return self.add(token, position)
        return super().initial(tokens, offset)

    def folds(self, block):
        c, h, d = self.c, self.c.heads, self.c.width // self.c.heads
        q = block.q.weight.reshape(h, d, c.width)
        o = block.o.weight.reshape(c.width, h, d).permute(1, 0, 2).contiguous()
        if self.ops:
            fq = self.ops.bmm(block.uk, q, transpose_a=True)
            fo = self.ops.bmm(o, block.uv)
        else:
            fq = torch.bmm(block.uk.transpose(1, 2), q)
            fo = torch.bmm(o, block.uv)
        return fq.reshape(h * c.rank, c.width), fo.permute(1, 0, 2).reshape(c.width, h * c.rank)

    def attend(self, q, k, v, causal=True):
        scale = (self.c.width // self.c.heads) ** -0.5
        if self.ops:
            return self.ops.attention(q.contiguous(), k.contiguous(), v.contiguous(), causal=causal, scale=scale)
        # Expand views so the Flash control uses the same shared physical latent.
        return F.scaled_dot_product_attention(q, k.expand(-1, q.shape[1], -1, -1),
                                             v.expand(-1, q.shape[1], -1, -1),
                                             is_causal=causal, scale=scale)

    def heads(self, x, dim):
        return x.reshape(*x.shape[:2], self.c.heads, dim).transpose(1, 2).contiguous()

    def forward(self, tokens, policy='none', return_cache=False, return_hidden=False, last_logits=False):
        if policy not in ('none', 'loop') or (return_cache and policy != 'none'):
            raise ValueError('unsupported checkpoint/cache policy')
        c, d = self.c, self.c.width // self.c.heads
        x = self.initial(tokens)
        latents = [] if c.kind == 'naive' else [
            self.linear(self.normalize(x), down.weight).unsqueeze(1).contiguous()
            for down in self.down
        ]
        folds = [] if c.kind == 'naive' else [self.folds(block) for block in self.blocks]
        caches = []

        def step(state, loop, *shared):
            # Explicit checkpoint arguments preserve gradients through the
            # global latent and folded weights, shared by every recurrence.
            latent_args = shared[:len(latents)]
            fold_args = shared[len(latents):]
            for layer in range(c.layers):
                idx = loop * c.layers + layer if c.untied else layer
                block, z = self.blocks[idx], self.normalize(state)
                if c.kind == 'naive':
                    q, k, v = [self.heads(self.linear(z, linear.weight), d)
                               for linear in (block.q, block.k, block.v)]
                    if return_cache:
                        caches.append((k, v))
                    a = self.attend(q, k, v).transpose(1, 2).reshape(*z.shape[:2], c.width)
                    delta = self.linear(a, block.o.weight)
                else:
                    fq, fo = fold_args[2*idx:2*idx+2]
                    q = self.heads(self.linear(z, fq), c.rank)
                    kv = latent_args[idx if c.kind == 'layerwise' else 0]
                    a = self.attend(q, kv, kv).transpose(1, 2).reshape(*z.shape[:2], c.heads*c.rank)
                    delta = self.linear(a, fo)
                state = self.add(state, delta)
                delta = self.linear(self.activate(self.linear(self.normalize(state), block.w1.weight)), block.w2.weight)
                state = self.add(state, delta)
            return state

        shared = (*latents, *(weight for pair in folds for weight in pair))
        for loop in range(c.loops):
            if policy == 'loop':
                x = checkpoint(lambda state, *args, loop=loop: step(state, loop, *args),
                               x, *shared, use_reentrant=False)
            else:
                x = step(x, loop, *shared)
        hidden = self.normalize(x)
        result = hidden if return_hidden else self.linear(hidden[:, -1:, :] if last_logits else hidden, self.output.weight)
        return (result, caches if c.kind == 'naive' else latents) if return_cache else result

    def loss(self, tokens, targets, policy='none', chunk_size=0):
        if chunk_size and not self.ops:
            raise ValueError('streamed Tensor loss is a separate memory tradeoff')
        if chunk_size:
            hidden = self(tokens, policy, return_hidden=True).flatten(0, 1)
            return self.ops.linear_cross_entropy(hidden, self.output.weight, targets.flatten(), chunk_size=chunk_size)
        logits = self(tokens, policy).flatten(0, 1)
        return self.ops.cross_entropy(logits, targets.flatten()) if self.ops else F.cross_entropy(logits.float(), targets.flatten())

    @torch.no_grad()
    def prefill(self, tokens, capacity=None, last_logits=False):
        capacity = self.c.max_seq if capacity is None else capacity
        if not tokens.shape[1] <= capacity <= self.c.max_seq:
            raise ValueError('invalid capacity')
        logits, prefixes = self(tokens, return_cache=True, last_logits=last_logits)
        stores = []
        for prefix in prefixes:
            shared = self.c.kind != 'naive'
            k, v = (prefix, prefix) if shared else prefix
            if self.ops:
                cache = KVCache(self.ops, k.shape[0], k.shape[1], capacity, k.shape[-1],
                                dtype=k.dtype, shared=shared)
            else:
                keys = torch.empty((*k.shape[:2], capacity, k.shape[-1]), device=k.device, dtype=k.dtype)
                values = keys if shared else torch.empty_like(keys)
                cache = TorchCache(keys, values)
            cache.append(k, v)
            stores.append(cache)
        state = dict(caches=stores, length=tokens.shape[1],
                     folds=[] if self.c.kind == 'naive' else [self.folds(b) for b in self.blocks],
                     versions=tuple(p._version for p in self.parameters()))
        return logits, state

    @torch.no_grad()
    def decode_token(self, tokens, state):
        if tokens.shape[1] != 1 or state['versions'] != tuple(p._version for p in self.parameters()):
            raise ValueError('decode requires one token and unchanged inference weights')
        c, d = self.c, self.c.width // self.c.heads
        offset = state['length']
        if offset >= c.max_seq:
            raise ValueError('position capacity exceeded')
        x = self.initial(tokens, offset)
        if c.kind != 'naive':
            for cache, down in zip(state['caches'], self.down):
                cache.append(self.linear(self.normalize(x), down.weight).unsqueeze(1))
        for loop in range(c.loops):
            for layer in range(c.layers):
                idx = loop*c.layers+layer if c.untied else layer
                block, z = self.blocks[idx], self.normalize(x)
                cache = state['caches'][loop*c.layers+layer if c.kind == 'naive' else idx if c.kind == 'layerwise' else 0]
                if c.kind == 'naive':
                    q, k, v = [self.heads(self.linear(z, linear.weight), d) for linear in (block.q, block.k, block.v)]
                    cache.append(k, v)
                    output_weight = block.o.weight
                else:
                    fq, output_weight = state['folds'][idx]
                    q = self.heads(self.linear(z, fq), c.rank)
                if self.ops:
                    a = self.ops.decode(q, cache, scale=d**-0.5)
                else:
                    a = self.attend(q, cache.keys[:, :, :cache.length], cache.values[:, :, :cache.length], causal=False)
                a = a.transpose(1, 2).reshape(tokens.shape[0], 1, -1)
                x = self.add(x, self.linear(a, output_weight))
                x = self.add(x, self.linear(self.activate(self.linear(self.normalize(x), block.w1.weight)), block.w2.weight))
        state['length'] += 1
        return self.linear(self.normalize(x), self.output.weight)

    @staticmethod
    def rewind(state, length):
        """Reset only the logical tail; used to replay the same timed trajectory."""
        state['length'] = length
        for cache in state['caches']:
            cache.length = length
            if isinstance(cache, KVCache):
                cache.lengths.fill_(length)
                cache.overflow.zero_()
                cache.captured_mutation = False
