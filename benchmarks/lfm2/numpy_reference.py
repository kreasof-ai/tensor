"""Independent CPU reference for the WebGPU projection/attention precision.

Prefill projections round operands to FP16 and accumulate in FP32. Decode
projections and attention use FP32 products, with FP16 persistent K/V caches.
No Tensor kernels or runtime are imported.
"""
import numpy as np
from tensor_llm.gguf import GGUF
from tensor_llm.config import Config


class _HalfWeights:
    """FP16-rounded 2-D operands, rounded on demand instead of all at once.

    Every weight is touched at most once per 32-token chunk, so a single-entry
    cache is enough. Materialising the whole map doubles resident FP32 weights,
    which does not fit host RAM for multi-billion-parameter checkpoints. The
    rounded values are identical to the eager dict comprehension.
    """
    def __init__(self,weights):
        self.weights=weights;self.name=None;self.value=None
    def __getitem__(self,name):
        if name!=self.name:
            self.value=self.weights[name].astype(np.float16).astype(np.float32);self.name=name
        return self.value


class Reference:
    def __init__(self,path,*,context=512):
        self.gguf=GGUF(path);self.config=Config.from_gguf(self.gguf);self.context=context
        self.weights={name:np.array(self.gguf.array(name),copy=True) for name in self.gguf.tensors}
        self.half_weights=_HalfWeights(self.weights)
        self.reset()

    def reset(self):
        c=self.config;self.position=0
        self.states={i:np.zeros((2,c.width),np.float32) for i,k in enumerate(c.layers) if k=='conv'}
        self.caches={i:(np.zeros((self.context,c.kv_heads,c.head_dim),np.float16),np.zeros((self.context,c.kv_heads,c.head_dim),np.float16)) for i,k in enumerate(c.layers) if k=='attention'}

    def forward(self,tokens):
        tokens=np.asarray(tokens,np.int32)
        for start in range(0,len(tokens),32):result=self._chunk(tokens[start:start+32])
        return result

    def _chunk(self,tokens):
        c=self.config;r=len(tokens);d=c.head_dim;start=self.position;w=self.weights
        if start+r>self.context:raise ValueError('context capacity exceeded')
        def project(name,x,single=False):
            half=r>1 and not single
            return (x.astype(np.float16).astype(np.float32) if half else x) @ (self.half_weights[name] if half else w[name]).T
        def norm(name,x):return x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+np.float32(c.epsilon))*w[name]
        def rotate(x):
            angles=np.arange(start,start+r,dtype=np.float32)[:,None]*np.exp(np.arange(d//2,dtype=np.float32)*np.float32(-np.log(c.theta)*2/d))
            co,si=np.cos(angles)[:,None,:],np.sin(angles)[:,None,:]
            left,right=np.split(x,2,axis=-1)
            return np.concatenate((left*co-right*si,left*si+right*co),axis=-1)
        hidden=w['token_embd.weight'][tokens]
        for i,kind in enumerate(c.layers):
            p=f'blk.{i}.';x=norm(p+'attn_norm.weight',hidden)
            if kind=='conv':
                b,gate,v=np.split(project(p+'shortconv.in_proj.weight',x),3,axis=-1)
                combined=np.concatenate((self.states[i],b*v),axis=0)
                self.states[i]=combined[-2:].copy()
                filters=w[p+'shortconv.conv.weight']
                conv=sum(combined[j:j+r]*filters[:,j] for j in range(3))*gate
                mix=project(p+'shortconv.out_proj.weight',conv)
            else:
                q=project(p+'attn_q.weight',x).reshape(r,c.heads,d)
                k=project(p+'attn_k.weight',x).reshape(r,c.kv_heads,d)
                v=project(p+'attn_v.weight',x).reshape(r,c.kv_heads,d)
                q=rotate(norm(p+'attn_q_norm.weight',q));k=rotate(norm(p+'attn_k_norm.weight',k))
                kc,vc=self.caches[i];kc[start:start+r]=k;vc[start:start+r]=v
                keys=np.repeat(kc[:start+r].astype(np.float32),c.heads//c.kv_heads,axis=1).transpose(1,0,2)
                vals=np.repeat(vc[:start+r].astype(np.float32),c.heads//c.kv_heads,axis=1).transpose(1,0,2)
                scores=(q.transpose(1,0,2) @ keys.transpose(0,2,1))*np.float32(d**-0.5)
                mask=np.arange(start+r)[None,:]>np.arange(start,start+r)[:,None]
                scores=np.where(mask[None],-np.inf,scores)
                prob=np.exp(scores-np.max(scores,axis=-1,keepdims=True));prob/=prob.sum(axis=-1,keepdims=True)
                attn=(prob @ vals).transpose(1,0,2).reshape(r,c.width)
                mix=project(p+'attn_output.weight',attn)
            hidden=hidden+mix;x=norm(p+'ffn_norm.weight',hidden)
            gate=project(p+'ffn_gate.weight',x);up=project(p+'ffn_up.weight',x)
            activated=gate/(1+np.exp(-gate))*up
            hidden=hidden+project(p+'ffn_down.weight',activated)
        final=norm('token_embd_norm.weight',hidden[-1:]);self.position+=r
        return project('output.weight' if 'output.weight' in w else 'token_embd.weight',final,single=True)[0]
