"""Experimental LFM2 adapter for tinygrad's generic compiler, not its AMD LLM kernels.

Matches the existing NumPy precision contract. Fixed 1/32-row JITs use symbolic
position and valid-row count; short convolution state is reset inside the graph.
"""
import numpy as np
from tinygrad import Tensor,TinyJit,UOp,Device,dtypes
from tinygrad.llm.gguf import ggml_data_to_tensor
from tensor_llm.common.gguf import GGUF
from tensor_llm.lfm2.config import Config


def rounded_half(value):
    """Explicit nearest-even half operands on the measured Dawn/Vulkan stack.

    Its native f16 conversion truncates; round in FP32 before an exact cast.
    The integer path covers normal halves and the scaled path subnormal halves.
    """
    if value.dtype==dtypes.half:return value.float()
    if not Device.DEFAULT.startswith('WEBGPU'):return value.half().float()
    value=value.float();bits=value.bitcast(dtypes.uint32);magnitude=value.abs()
    normal=((bits+4095+((bits>>13)&1))&0xffffe000).bitcast(dtypes.float)
    scaled=magnitude*16777216;lower=scaled.floor();fraction=scaled-lower
    increment=(fraction>.5)|((fraction==.5)&((lower.cast(dtypes.int32)&1)==1))
    small=(lower+increment.float())/16777216
    small=(small.bitcast(dtypes.uint32)|(bits&0x80000000)).bitcast(dtypes.float)
    result=(magnitude<2**-14).where(small,normal)
    infinity=(value<0).where(float('-inf'),float('inf'))
    return value.isnan().where(value,(magnitude>=65520).where(infinity,result))


class Reference:
    def __init__(self,path,*,context=512,weight_mode='packed',search_root=None):
        self.gguf=GGUF(path);self.config=Config.from_gguf(self.gguf);self.context=context
        self.device=Device.DEFAULT;self.capacity=(context+32+63)//64*64
        self.replay=None
        if search_root is not None:
            from benchmarks.lfm2.tinygrad_schedule_replay import ProjectionReplay
            self.replay=ProjectionReplay(search_root)
        self.weights={}
        for name,info in self.gguf.tensors.items():
            if info.type in (0,1):
                value=Tensor(np.array(self.gguf.array(name,dtype=np.float16 if info.type==1 else np.float32),copy=True),device=self.device).realize()
            elif weight_mode=='decoded':
                value=Tensor(np.array(self.gguf.array(name),copy=True),device=self.device).realize()
            else:
                raw=Tensor(np.array(self.gguf.packed(name),copy=True),device=self.device).realize()
                value=ggml_data_to_tensor(raw,int(np.prod(info.shape)),info.type).reshape(info.shape)
            self.weights[name]=value
        if any(self.weights[name].dtype!=dtypes.half for name,info in self.gguf.tensors.items() if info.type==1):
            raise ValueError('F16 GGUF tensors must retain native half storage')
        c=self.config
        self.states={i:Tensor.zeros(2,c.width,device=self.device).realize() for i,k in enumerate(c.layers) if k=='conv'}
        self.caches={i:tuple(Tensor.zeros(self.capacity,c.kv_heads,c.head_dim,dtype=dtypes.half,device=self.device).realize() for _ in range(2))
                     for i,k in enumerate(c.layers) if k=='attention'}
        angles=np.arange(self.capacity,dtype=np.float32)[:,None]*np.exp(np.arange(c.head_dim//2,dtype=np.float32)*np.float32(-np.log(c.theta)*2/c.head_dim))
        self.cos=Tensor(np.cos(angles),device=self.device).realize();self.sin=Tensor(np.sin(angles),device=self.device).realize()
        self.jits={r:TinyJit(self._chunk) for r in (1,32)}
        self.reset();Device[self.device].synchronize()

    def reset(self):self.position=0

    def _chunk(self,tokens,start,valid):
        c=self.config;r=tokens.shape[0];d=c.head_dim;w=self.weights
        def project(name,x,single=False):
            weight=w[name]
            replay=self.replay is not None and r==32 and '.ffn_' in name
            if replay:x=x.contiguous().realize()
            if r>1 and not single:x,weight=rounded_half(x),rounded_half(weight)
            else:x,weight=x.float(),weight.float()
            result=x@weight.T
            return result.realize() if replay else result
        def norm(name,x):return x*(x.square().mean(axis=-1,keepdim=True)+c.epsilon).rsqrt()*w[name].float()
        def rotate(x):
            left,right=x.chunk(2,dim=-1)
            co=self.cos[start:start+r].reshape(r,1,d//2);si=self.sin[start:start+r].reshape(r,1,d//2)
            return (left*co-right*si).cat(left*si+right*co,dim=-1)
        hidden=w['token_embd.weight'][tokens].float()
        for i,kind in enumerate(c.layers):
            p=f'blk.{i}.';x=norm(p+'attn_norm.weight',hidden)
            if kind=='conv':
                b,gate,v=project(p+'shortconv.in_proj.weight',x).chunk(3,dim=-1)
                state=Tensor(start).eq(0).where(0,self.states[i])
                combined=state.cat(b*v,dim=0).contiguous()
                state_store=self.states[i].uop.store(combined[valid:valid+2].uop)
                filters=w[p+'shortconv.conv.weight'].float()
                conv=sum(combined[j:j+r]*filters[:,j] for j in range(3))*gate
                mix=project(p+'shortconv.out_proj.weight',conv)
                mix=Tensor(mix.contiguous().uop.after(state_store))
            else:
                q=project(p+'attn_q.weight',x).reshape(r,c.heads,d)
                k=project(p+'attn_k.weight',x).reshape(r,c.kv_heads,d)
                v=project(p+'attn_v.weight',x).reshape(r,c.kv_heads,d)
                q=rotate(norm(p+'attn_q_norm.weight',q));k=rotate(norm(p+'attn_k_norm.weight',k))
                kc,vc=self.caches[i]
                ks=kc[start:start+r].uop.store(rounded_half(k).half().uop);vs=vc[start:start+r].uop.store(rounded_half(v).half().uop)
                keys=Tensor(kc.uop.after(ks)).float().unsqueeze(2).expand(self.capacity,c.kv_heads,c.heads//c.kv_heads,d).reshape(self.capacity,c.heads,d).permute(1,0,2)
                vals=Tensor(vc.uop.after(vs)).float().unsqueeze(2).expand(self.capacity,c.kv_heads,c.heads//c.kv_heads,d).reshape(self.capacity,c.heads,d).permute(1,0,2)
                scores=(q.permute(1,0,2)@keys.transpose(-1,-2))*d**-0.5
                mask=Tensor.arange(self.capacity).to(self.device).reshape(1,1,self.capacity)>(Tensor.arange(r).to(self.device)+start).reshape(1,r,1)
                prob=mask.where(float('-inf'),scores).softmax(-1)
                attn=(prob@vals).permute(1,0,2).reshape(r,c.width)
                mix=project(p+'attn_output.weight',attn)
            hidden=hidden+mix
            # FFN realization boundaries must also complete their residual.
            # Otherwise its lazy convolution/state write can execute again
            # when the residual is consumed after the materialized projection.
            if self.replay is not None and r==32:hidden=hidden.contiguous().realize()
            x=norm(p+'ffn_norm.weight',hidden)
            gate=project(p+'ffn_gate.weight',x);up=project(p+'ffn_up.weight',x)
            hidden=hidden+project(p+'ffn_down.weight',gate.silu()*up)
        final=norm('token_embd_norm.weight',hidden[valid-1:valid])
        return project('output.weight' if 'output.weight' in w else 'token_embd.weight',final,single=True)[0].realize()

    def forward(self,tokens):
        tokens=np.asarray(tokens)
        if tokens.ndim!=1 or not tokens.size or tokens.dtype.kind not in 'iu':raise ValueError('requires nonempty integer token IDs')
        if np.any(tokens<0) or np.any(tokens>=self.config.vocab):raise ValueError('token ID outside vocabulary')
        if self.position+len(tokens)>self.context:raise ValueError('context capacity exceeded')
        for start in range(0,len(tokens),32):
            values=tokens[start:start+32];r=1 if len(values)==1 else 32
            host=np.zeros(r,np.int32);host[:len(values)]=values
            pos=UOp.variable('position',0,self.context-1).bind(self.position)
            valid=UOp.variable('valid',2,32).bind(len(values)) if r==32 else 1
            if self.replay is None:result=self.jits[r](Tensor(host,device=self.device),pos,valid)
            else:
                with self.replay:
                    result=self.jits[r](Tensor(host,device=self.device),pos,valid)
            self.position+=len(values)
        return result.numpy()
