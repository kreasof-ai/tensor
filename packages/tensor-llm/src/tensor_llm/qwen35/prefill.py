"""Framework-free chunked prefill sharing a Qwen batch's resident resources."""
import ctypes as ct
import hashlib,json
from pathlib import Path
import numpy as np
from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall
from .artifacts import identity, requirements


class Qwen35Prefill:
    def __init__(self,model,bundle):
        model._check();self.model=model;self.device=model.device
        self.closed=False;self.graph=None;self.buffers={};self.kernels={};self.plan=[]
        bundle=Path(bundle).resolve();manifest=json.loads((bundle/'prefill.json').read_text())
        if (manifest['schema']!='tensor.qwen35-prefill.v1' or manifest['slots']!=model.slots
                or manifest['context']!=model.context
                or manifest.get('kv_dtype','bfloat16')!=model.kv_dtype):raise ValueError('prefill bundle capacity or KV precision mismatch')
        self.chunk=manifest['chunk'];self.rows=model.slots*self.chunk
        try:
            for key,row in manifest['kernels'].items():
                path=(bundle/row['path']).resolve()
                if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('prefill artifact checksum mismatch')
                self.kernels[key]=self.device.load(path)
            self._allocate();self._plan()
            self.graph=CudaGraph(self.device,self._submit,resources=(*model.weights.values(),
                *model.buffers.values(),*(b for state in model.states.values() for b in state),
                *self.buffers.values(),*self.kernels.values(),*model.kernels.values()))
            if hasattr(model,'prefills'):model.prefills.add(self)
        except BaseException:self.close();raise

    def _allocate(self):
        d=self.device;r=self.rows
        # These activations are independent of decode workspaces; checkpoint,
        # KV and recurrent tensors are shared by both execution phases.
        skip={'positions','active','logits','att_parts','att_stats','experts','routes'}
        for name,buffer in self.model.buffers.items():
            if name in skip or name.startswith('_tune_'):continue
            self.buffers[name]=d.empty((r,*buffer.shape[1:]),buffer.dtype)
        for name in ('flat_positions','flat_active'):
            self.buffers[name]=d.empty((r,),'int32')
        self.buffers['lengths']=d.empty((self.model.slots,),'int32')
        self.buffers['counts']=d.empty((256,),'int32')
        self.buffers['routes']=d.empty((256,r),'int32')
        self.quantized={}
        for name in ('normal','gdn_normal','att_out','shared_activation','expert_activation'):
            x=self.buffers[name]
            bits=d.empty(x.shape,'uint8');scales=d.empty((*x.shape[:-1],x.shape[-1]//128))
            self.quantized[name]=(bits,scales)
            self.buffers['quantized_'+name]=bits;self.buffers['scales_'+name]=scales

    def _plan(self):
        model=self.model;b=self.buffers;w=model.weights;d=self.device;r=self.rows;s=model.slots
        self.plan=[];valid=set()
        def call(kind,p,**bindings):
            kernel=self.kernels[identity(kind,p)]
            args=tuple(bindings[a['name']] for a in kernel.manifest['arguments'])
            values,symbols,launch=kernel._bind(args,{},include_outputs=True)
            self.plan.append((kernel,BoundCall(d,kernel.manifest,values,symbols,launch,validated=True)))
            if 'out' in bindings:
                for name in valid.copy():
                    if b[name] is bindings['out']:valid.remove(name)
        def small(kind,**bindings):call(kind,dict(slots=s,chunk=self.chunk),**bindings)
        def quantize(name):
            if name not in valid:
                x=b[name];p=dict(r=r,k=x.shape[-1])
                if len(x.shape)==3:p['top']=8
                bits,scales=self.quantized[name]
                call('quantize',p,x=x,out=bits,scales=scales);valid.add(name)
            return self.quantized[name]
        def fp8(name,x,out):
            ww=w[name+'.weight'];o,k=ww.shape;xx,scale=quantize(x)
            call('fp8_linear',dict(r=r,k=k,o=o),x=xx,w=ww,scales=w[name+'.weight_scale_inv'],activation_scales=scale,out=b[out])
        def bf16(name,x,out):
            ww=w[name+'.weight'];o,k=ww.shape
            call('bf16_linear',dict(r=r,k=k,o=o),x=b[x],w=ww,out=b[out])
        def norm(name,x=None):
            args=dict(residual=b['residual'],w=w[name+'.weight'],out=b['normal'])
            if x:args['x']=b[x]
            call('add_rms' if x else 'rms',dict(r=r,c=2048,eps=model.config.epsilon),**args)
        def experts(name,x,out,o,k,routed):
            xx,scale=quantize(x)
            call('fp8_experts',dict(r=r,experts=256,top=8,o=o,k=k,routed_input=routed),
                x=xx,activation_scales=scale,w=w[name+'.weight'],scales=w[name+'.weight_scale_inv'],
                counts=b['counts'],routes=b['routes'],out=b[out])
        small('controls',position=model.buffers['positions'],lengths=b['lengths'],
              flat_position=b['flat_positions'],active=b['flat_active'])
        call('embedding',dict(r=r,c=2048,vocab=model.config.vocab),ids=b['tokens'],
             w=w['model.language_model.embed_tokens.weight'],out=b['residual'])
        for layer,kind in enumerate(model.config.layers):
            root=f'model.language_model.layers.{layer}.'
            norm(root+'input_layernorm',None if layer==0 else 'ffn')
            if kind=='linear_attention':
                prefix=root+'linear_attn.'
                for name,dest in (('in_proj_qkv','qkv'),('in_proj_z','z')):fp8(prefix+name,'normal',dest)
                for name in ('a','b'):bf16(prefix+'in_proj_'+name,'normal',name)
                state,history=model.states[layer]
                small('gdn_conv',x=b['qkv'],w=w[prefix+'conv1d.weight'],lengths=b['lengths'],state=history,out=b['conv'])
                call('gdn_prepare',dict(r=r),qkv=b['conv'],a=b['a'],b=b['b'],dt=w[prefix+'dt_bias'],A=w[prefix+'A_log'],
                     q=b['gdn_q'],k=b['gdn_k'],v=b['gdn_v'],g=b['gdn_decay'],beta=b['gdn_beta'])
                small('gdn_scan',q=b['gdn_q'],k=b['gdn_k'],v=b['gdn_v'],g=b['gdn_decay'],beta=b['gdn_beta'],
                      lengths=b['lengths'],state=state,out=b['gdn_out'])
                call('gdn_norm',dict(r=r,eps=model.config.epsilon),x=b['gdn_out'],z=b['z'],w=w[prefix+'norm.weight'],out=b['gdn_normal'])
                fp8(prefix+'out_proj','gdn_normal','mixed')
            else:
                prefix=root+'self_attn.'
                for name,dest in (('q','attq'),('k','attk'),('v','attv')):fp8(prefix+name+'_proj','normal',dest)
                kc,vc,*scales=model.states[layer]
                kv = dict(kv_dtype='fp8') if model.kv_dtype=='fp8' else {}
                cache = dict(ks=scales[0],vs=scales[1]) if scales else {}
                p=dict(r=r,eps=model.config.epsilon,capacity=model.context,theta=model.config.rope_theta)
                call('attention_q',p,projection=b['attq'],w=w[prefix+'q_norm.weight'],positions=b['flat_positions'],
                     active=b['flat_active'],q=b['attq_normal'])
                p=dict(slots=s,chunk=self.chunk,capacity=model.context,eps=model.config.epsilon,theta=model.config.rope_theta,**kv)
                call('attention_kv',p,projection=b['attk'],values=b['attv'],w=w[prefix+'k_norm.weight'],positions=b['flat_positions'],
                     active=b['flat_active'],kc=kc,vc=vc,**cache)
                call('attention',dict(slots=s,chunk=self.chunk,capacity=model.context,**kv),q=b['attq_normal'],kc=kc,vc=vc,**cache,
                     projection=b['attq'],positions=model.buffers['positions'],lengths=b['lengths'],out=b['att_out'])
                fp8(prefix+'o_proj','att_out','mixed')
            norm(root+'post_attention_layernorm','mixed')
            prefix=root+'mlp.'
            bf16(prefix+'gate','normal','router');bf16(prefix+'shared_expert_gate','normal','shared_gate')
            call('router',dict(r=r,experts=256,top=8),logits=b['router'],ids=b['ids'],weights=b['router_weights'])
            small('expert_routes',ids=b['ids'],active=b['flat_active'],counts=b['counts'],routes=b['routes'])
            for name,out in (('gate_proj','expert_g'),('up_proj','expert_u')):
                experts(prefix+'experts.'+name,'normal',out,512,2048,False)
            call('swiglu_experts',dict(r=r,top=8,c=512),g=b['expert_g'],u=b['expert_u'],out=b['expert_activation'])
            experts(prefix+'experts.down_proj','expert_activation','expert_out',2048,512,True)
            for name,out in (('gate_proj','shared_g'),('up_proj','shared_u')):fp8(prefix+'shared_expert.'+name,'normal',out)
            call('swiglu',dict(r=r,c=512),g=b['shared_g'],u=b['shared_u'],out=b['shared_activation'])
            fp8(prefix+'shared_expert.down_proj','shared_activation','shared_out')
            call('moe_combine',dict(r=r,top=8,c=2048),experts=b['expert_out'],weights=b['router_weights'],
                 shared=b['shared_out'],gate=b['shared_gate'],out=b['ffn'])
        norm('model.language_model.norm','ffn')
        call('last_rows',dict(slots=s,chunk=self.chunk,width=2048),x=b['normal'],lengths=b['lengths'],out=model.buffers['normal'])
        # Preserve the decode executor's validated final projection/argmax
        # artifacts; they consume only the final prompt row per request.
        for kernel,bound,label in model.plan[-2:]:self.plan.append((kernel,bound))
        small('advance',position=model.buffers['positions'],lengths=b['lengths'],active=model.buffers['active'])

    def _submit(self):
        for kernel,call in self.plan:self.device._launch(kernel,call)

    def forward(self,tokens,lengths=None,*,read_logits=False):
        self.model._check()
        if self.closed:raise RuntimeError('prefill is closed')
        tokens=np.asarray(tokens)
        s,c=self.model.slots,self.chunk
        if tokens.shape!=(s,c) or tokens.dtype.kind not in 'iu':raise ValueError('prefill requires [slots,chunk] token IDs')
        if lengths is None:lengths=np.full(s,c,'int32')
        lengths=np.asarray(lengths,dtype='int32')
        if lengths.shape!=(s,) or np.any(lengths<0) or np.any(lengths>c):raise ValueError('invalid prefill chunk lengths')
        mask=np.arange(c)[None,:]<lengths[:,None]
        if np.any(tokens[mask]<0) or np.any(tokens[mask]>=self.model.config.vocab):raise ValueError('invalid prefill token ID')
        if np.any(self.model.position+lengths>self.model.context):raise ValueError('Qwen context capacity exhausted')
        data=np.where(mask,tokens,-1).astype('int32').reshape(-1)
        for name,value in (('tokens',data),('lengths',lengths)):
            value=np.ascontiguousarray(value);buffer=self.buffers[name]
            self.device.driver.call('cuMemcpyHtoD_v2',buffer.pointer,ct.c_void_p(value.ctypes.data),value.nbytes)
        self.device.driver.call('cuStreamSynchronize',None)
        self.model.active=(lengths>0).astype('int32');self.model._write('active',self.model.active)
        self.graph.launch();self.model.position+=lengths
        result=self.model.buffers['tokens'].to_numpy()
        return (result,self.model.buffers['logits'].to_numpy()) if read_logits else result

    def close(self):
        if self.closed:return
        if self.graph:self.graph.close();self.graph=None
        for buffer in self.buffers.values():buffer.release()
        for kernel in self.kernels.values():kernel.release()
        self.buffers,self.kernels={},{};self.closed=True
        if hasattr(self.model,'prefills'):self.model.prefills.discard(self)
