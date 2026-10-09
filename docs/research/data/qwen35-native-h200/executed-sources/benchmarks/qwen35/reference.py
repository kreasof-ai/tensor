"""Independent eager CPU oracle with streamed native checkpoint matrices.

Torch is confined to this producer-side reference. The consumer uses only
Tensor's CUDA driver runtime. Expert matrices are materialized only while
needed, keeping the reference below the host's 30 GiB memory limit.
"""
import numpy as np
import torch
import torch.nn.functional as F
from tensor_llm.qwen35.checkpoint import Qwen35Checkpoint


class Reference:
    def __init__(self,path,slots):
        torch.set_num_threads(2)
        self.checkpoint=Qwen35Checkpoint(path)
        self.config=self.checkpoint.config
        self.slots=slots
        self.position=np.zeros(slots,dtype='int32')
        self.states={};self.histories={};self.cache={}
        for layer,kind in enumerate(self.config.layers):
            if kind=='linear_attention':
                self.states[layer]=torch.zeros(slots,32,128,128)
                self.histories[layer]=torch.zeros(slots,8192,3)
            else:self.cache[layer]=[([],[]) for _ in range(slots)]

    def weight(self,name):
        info=self.checkpoint.tensors[name]
        array=self.checkpoint.read(name)
        raw=torch.from_numpy(array.copy())
        if info.dtype=='BF16':return raw.view(torch.bfloat16).float()
        if info.dtype=='F8_E4M3':return raw.view(torch.float8_e4m3fn).float()
        return raw

    @staticmethod
    def quantize(x):
        shape=x.shape
        blocks=x.float().reshape(-1,shape[-1]//128,128)
        scales=blocks.abs().amax(-1,keepdim=True).clamp_min(1e-12)/448
        return ((blocks/scales).to(torch.float8_e4m3fn).float()*scales).reshape(shape)

    def linear(self,name,x):
        w=self.weight(name+'.weight')
        if self.checkpoint.tensors[name+'.weight'].dtype=='F8_E4M3':
            scales=self.weight(name+'.weight_scale_inv')
            w*=scales.repeat_interleave(128,0).repeat_interleave(128,1)
            x=self.quantize(x)
        return x.float()@w.T

    def norm(self,name,x):
        values=x.float()
        return (values*torch.rsqrt(values.square().mean(-1,keepdim=True)+self.config.epsilon)
                *(1+self.weight(name+'.weight'))).bfloat16()

    def gdn(self,layer,x,active):
        root=f'model.language_model.layers.{layer}.linear_attn.'
        projected=self.linear(root+'in_proj_qkv',x).bfloat16().float()
        z=self.linear(root+'in_proj_z',x).bfloat16().float().reshape(self.slots,32,128)
        a=self.linear(root+'in_proj_a',x).bfloat16().float()
        b=self.linear(root+'in_proj_b',x).bfloat16().float()
        history=self.histories[layer]
        taps=self.weight(root+'conv1d.weight')[:,0,:]
        window=torch.cat((history,projected.unsqueeze(-1)),dim=-1)
        convolved=F.silu((window*taps).sum(-1)).bfloat16().float()
        history[active]=window[active,:,1:]
        q,k,v=convolved.split([2048,2048,4096],dim=-1)
        q=q.reshape(self.slots,16,128).repeat_interleave(2,1)
        k=k.reshape(self.slots,16,128).repeat_interleave(2,1)
        v=v.reshape(self.slots,32,128)
        q=q/torch.sqrt(q.square().sum(-1,keepdim=True)+1e-6)/np.sqrt(128)
        k=k/torch.sqrt(k.square().sum(-1,keepdim=True)+1e-6)
        decay=(-self.weight(root+'A_log').exp()*F.softplus(a+self.weight(root+'dt_bias'))).exp()
        beta=b.sigmoid()
        old=self.states[layer]
        decayed=old*decay[...,None,None]
        delta=v-(decayed@k.unsqueeze(-1)).squeeze(-1)
        new=decayed+torch.einsum('bhv,bhk->bhvk',delta*beta.unsqueeze(-1),k)
        old[active]=new[active]
        value=(old@q.unsqueeze(-1)).squeeze(-1).bfloat16().float()
        normal=value*torch.rsqrt(value.square().mean(-1,keepdim=True)+self.config.epsilon)
        normal=normal*self.weight(root+'norm.weight')*F.silu(z)
        return self.linear(root+'out_proj',normal.reshape(self.slots,4096).bfloat16())

    def rotary(self,x,position):
        frequencies=self.config.rope_theta**(-torch.arange(0,64,2,dtype=torch.float32)/64)
        angle=position*frequencies
        c,s=angle.cos(),angle.sin()
        result=x.float().clone()
        lo,hi=x[...,:32].float(),x[...,32:64].float()
        result[...,:32]=lo*c-hi*s
        result[...,32:64]=hi*c+lo*s
        return result.bfloat16()

    def attention(self,layer,x,active):
        root=f'model.language_model.layers.{layer}.self_attn.'
        qgate=self.linear(root+'q_proj',x).bfloat16().reshape(self.slots,16,512)
        q,gate=qgate.chunk(2,dim=-1)
        k=self.linear(root+'k_proj',x).bfloat16().reshape(self.slots,2,256)
        v=self.linear(root+'v_proj',x).bfloat16().reshape(self.slots,2,256)
        q=self.norm(root+'q_norm',q)
        k=self.norm(root+'k_norm',k)
        out=torch.zeros(self.slots,16,256)
        for slot in range(self.slots):
            if not active[slot]:continue
            qq=self.rotary(q[slot],self.position[slot])
            kk=self.rotary(k[slot],self.position[slot])
            keys,values=self.cache[layer][slot]
            keys.append(kk);values.append(v[slot].clone())
            keys_tensor=torch.stack(keys).repeat_interleave(8,1).transpose(0,1).float()
            values_tensor=torch.stack(values).repeat_interleave(8,1).transpose(0,1).float()
            scores=torch.einsum('hd,htd->ht',qq.float(),keys_tensor)/16
            probabilities=scores.softmax(-1).bfloat16().float()
            out[slot]=torch.einsum('ht,htd->hd',probabilities,values_tensor)
        out=(out.bfloat16().float()*gate.float().sigmoid()).bfloat16().reshape(self.slots,4096)
        return self.linear(root+'o_proj',out)

    def mlp(self,layer,x):
        root=f'model.language_model.layers.{layer}.mlp.'
        router=self.linear(root+'gate',x).bfloat16().float()
        # The consumer resolves exactly equal BF16 router logits by expert ID.
        # torch.topk does not define the order of tied elements.
        ids=router.argsort(dim=-1,descending=True,stable=True)[:,:8]
        values=router.gather(-1,ids)
        weights=values.softmax(-1)
        output=torch.zeros(self.slots,2048)
        for expert in torch.unique(ids).tolist():
            rows,ranks=torch.where(ids==expert)
            name=root+f'experts.{expert}.'
            gate=self.linear(name+'gate_proj',x[rows]).bfloat16().float()
            up=self.linear(name+'up_proj',x[rows]).bfloat16().float()
            inner=(F.silu(gate)*up).bfloat16()
            down=self.linear(name+'down_proj',inner).bfloat16().float()
            output.index_add_(0,rows,down*weights[rows,ranks,None])
        gate=self.linear(root+'shared_expert.gate_proj',x).bfloat16().float()
        up=self.linear(root+'shared_expert.up_proj',x).bfloat16().float()
        shared=self.linear(root+'shared_expert.down_proj',(F.silu(gate)*up).bfloat16()).bfloat16().float()
        shared_gate=self.linear(root+'shared_expert_gate',x).bfloat16().float().sigmoid()
        return output+shared*shared_gate

    def forward(self,tokens,debug=None,progress=None):
        tokens=np.asarray(tokens,dtype='int32')
        active=torch.from_numpy(tokens>=0)
        info=self.checkpoint.tensors['model.language_model.embed_tokens.weight']
        rows=[]
        with info.shard.open('rb') as stream:
            for token in tokens:
                stream.seek(info.offset+max(int(token),0)*2048*2)
                rows.append(np.frombuffer(stream.read(2048*2),dtype='uint16').copy())
        hidden=torch.from_numpy(np.stack(rows)).view(torch.bfloat16)
        hidden[~active]=0
        residual=hidden
        for layer,kind in enumerate(self.config.layers):
            root=f'model.language_model.layers.{layer}.'
            normal=self.norm(root+'input_layernorm',residual)
            if debug:debug((layer,'input'),normal.float().numpy())
            mixed=(self.gdn(layer,normal,active) if kind=='linear_attention'
                   else self.attention(layer,normal,active))
            residual=(residual+mixed.bfloat16()).bfloat16()
            normal=self.norm(root+'post_attention_layernorm',residual)
            if debug:debug((layer,'post_attention'),normal.float().numpy())
            ffn=self.mlp(layer,normal)
            if debug:debug((layer,'mlp'),ffn.numpy())
            residual=(residual+ffn.bfloat16()).bfloat16()
            if progress:progress(f'CPU oracle layer {layer+1}/40')
        normal=self.norm('model.language_model.norm',residual)
        if debug:debug((40,'final'),normal.float().numpy())
        logits=self.linear('lm_head',normal)
        self.position+=active.numpy().astype('int32')
        return logits.numpy()
