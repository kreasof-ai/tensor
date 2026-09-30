"""Independent eager reference for Tensor's explicit mixed precision contract.

GGML validates block decoding separately. This reference computes projections,
normalization, short convolution, rotary embeddings and cached attention with
Torch operators; it does not import Tensor's kernel templates or runtime.
"""
import numpy as np
import torch
from tensor_llm.gguf import GGUF
from tensor_llm.config import Config


class Reference:
    def __init__(self,path):
        self.gguf=GGUF(path);self.config=Config.from_gguf(self.gguf)
        torch.backends.cuda.matmul.allow_tf32=False
        self.weights={name:torch.from_numpy(np.array(self.gguf.array(name),copy=True)).cuda() for name in self.gguf.tensors}
        self.reset()

    def reset(self):
        cfg=self.config
        self.position=0;self.states={};self.caches={}
        for i,kind in enumerate(cfg.layers):
            if kind=='conv':self.states[i]=torch.zeros((2,cfg.width),device='cuda')
            else:self.caches[i]=(torch.empty((8576,cfg.kv_heads,cfg.head_dim),dtype=torch.float16,device='cuda'),
                                 torch.empty((8576,cfg.kv_heads,cfg.head_dim),dtype=torch.float16,device='cuda'))

    @torch.inference_mode()
    def forward(self,tokens):
        cfg=self.config;r=len(tokens);c,d=cfg.width,cfg.head_dim;start=self.position
        weights=self.weights
        def linear(name,x):
            w=weights[name]
            if r>1:return torch.mm(x.half(),w.half().T,out_dtype=torch.float32)
            return torch.mm(x,w.T)
        def norm(name,x):return x*torch.rsqrt((x*x).mean(-1,keepdim=True)+cfg.epsilon)*weights[name]
        def rotate(x):
            angle=torch.arange(start,start+r,device='cuda')[:,None]*torch.exp(torch.arange(d//2,device='cuda')*(-np.log(cfg.theta)*2/d))
            co,si=angle.cos()[:,None,:],angle.sin()[:,None,:]
            left,right=x.chunk(2,dim=-1)
            return torch.cat((left*co-right*si,left*si+right*co),dim=-1)
        hidden=weights['token_embd.weight'][torch.tensor(tokens,device='cuda')]
        for i,kind in enumerate(cfg.layers):
            p=f'blk.{i}.';x=norm(p+'attn_norm.weight',hidden)
            if kind=='conv':
                projected=linear(p+'shortconv.in_proj.weight',x).reshape(r,3,c)
                b,gate,v=projected.unbind(1);combined=torch.cat((self.states[i],b*v),dim=0)
                self.states[i]=combined[-2:].clone()
                filters=weights[p+'shortconv.conv.weight']
                conv=sum(combined[j:j+r]*filters[:,j] for j in range(3))*gate
                mix=linear(p+'shortconv.out_proj.weight',conv)
            else:
                q=linear(p+'attn_q.weight',x).reshape(r,cfg.heads,d)
                k=linear(p+'attn_k.weight',x).reshape(r,cfg.kv_heads,d)
                v=linear(p+'attn_v.weight',x).reshape(r,cfg.kv_heads,d)
                q=rotate(norm(p+'attn_q_norm.weight',q));k=rotate(norm(p+'attn_k_norm.weight',k))
                kc,vc=self.caches[i];kc[start:start+r]=k;vc[start:start+r]=v
                keys=kc[:start+r].repeat_interleave(cfg.heads//cfg.kv_heads,dim=1).permute(1,0,2)
                values=vc[:start+r].repeat_interleave(cfg.heads//cfg.kv_heads,dim=1).permute(1,0,2)
                queries=q.permute(1,0,2)
                if r>1:scores=torch.bmm(queries.half(),keys.transpose(1,2),out_dtype=torch.float32)*d**-0.5
                else:scores=torch.bmm(queries,keys.float().transpose(1,2))*d**-0.5
                mask=torch.arange(start+r,device='cuda')[None,:]>torch.arange(start,start+r,device='cuda')[:,None]
                scores.masked_fill_(mask[None],-torch.inf);prob=scores.softmax(-1)
                if r>1:attn=torch.bmm(prob.half(),values,out_dtype=torch.float32)
                else:attn=torch.bmm(prob,values.float())
                mix=linear(p+'attn_output.weight',attn.permute(1,0,2).reshape(r,c))
            hidden=hidden+mix
            x=norm(p+'ffn_norm.weight',hidden)
            gate=linear(p+'ffn_gate.weight',x);up=linear(p+'ffn_up.weight',x)
            hidden=hidden+linear(p+'ffn_down.weight',torch.nn.functional.silu(gate)*up)
        final=norm('token_embd_norm.weight',hidden[-1:])
        # The output projection is always a single-token FP32 accumulation.
        logits=final @ weights.get('output.weight',weights['token_embd.weight']).T
        self.position+=r
        return logits[0].cpu().numpy()
