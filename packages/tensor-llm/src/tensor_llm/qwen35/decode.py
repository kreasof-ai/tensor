"""Experimental native Qwen3.5 MoE CUDA batch executor.

One checkpoint and one set of workspaces serve every slot. All request state
stays on the GPU. This initial executor uses a token-at-a-time prefill; a full
length throughput claim requires the optimized prefill and independent oracle
qualification, not just a successful decode microbenchmark.
"""
from __future__ import annotations

import ctypes as ct
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from threading import get_ident

import numpy as np

from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall
from .checkpoint import Qwen35Checkpoint
from ..common.artifacts import identity
from .artifacts import requirements


def implementation_hashes():
    import tensor.providers.cuda
    import tensor.providers.cuda_graph
    import tensor.runtime.abi
    from ..common import artifacts as common_artifacts
    from . import artifacts, checkpoint
    from .kernels import decode, fp8_kv
    from . import pipeline
    modules = (common_artifacts, artifacts, checkpoint, decode, pipeline, fp8_kv, tensor.providers.cuda, tensor.providers.cuda_graph,
               tensor.runtime.abi)
    result = {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in modules}
    result[__name__] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


class Qwen35Batch:
    """Owner-thread batch executor with independent recurrent and KV slots."""

    def __init__(self, checkpoint, bundle, device, *, graphs=True, progress=None):
        self.device, self.thread = device, get_ident()
        self.closed = False
        self.weights, self.kernels, self.buffers, self.states = {}, {}, {}, {}
        self.graph = None
        self.prefills = set()
        self.checkpoint = Qwen35Checkpoint(checkpoint)
        self.config = self.checkpoint.config
        self.bundle = Path(bundle).resolve()
        value = json.loads((self.bundle/'inference.json').read_text())
        if (value.get('schema') != 'tensor.qwen35-batch.v1'
                or value.get('implementation') != implementation_hashes()
                or value.get('config') != json.loads(json.dumps(asdict(self.config)))
                or value.get('target') != device.info['arch']):
            raise ValueError('Qwen bundle configuration, source or target mismatch; rebuild it')
        self.slots, self.context, self.splits = value['slots'], value['context'], value['splits']
        self.kv_dtype = value.get('kv_dtype','bfloat16')
        if self.kv_dtype not in ('bfloat16','fp8'):raise ValueError('unsupported KV cache precision')
        if self.slots not in (1,2,4,8,16,32,64) or not 1 <= self.context <= self.config.context_limit:
            raise ValueError('unsupported Qwen batch capacity')
        self.position = np.zeros(self.slots,dtype='int32')
        self.active = np.zeros(self.slots,dtype='int32')
        self._generation = device._generation
        try:
            for key, (kind,p) in requirements(self.config,self.slots,self.context,splits=self.splits,kv_dtype=self.kv_dtype).items():
                row = value['kernels'][key]
                descriptor = row.get('logical',row)
                artifact = (self.bundle/descriptor['path']).resolve()
                if (not artifact.is_relative_to(self.bundle)
                        or hashlib.sha256(artifact.read_bytes()).hexdigest() != descriptor['sha256']
                        or row['kind'] != kind or row['parameters'] != p):
                    raise ValueError('Qwen artifact identity mismatch')
                self.kernels[key] = device.load(artifact)
            if progress:progress('uploading native checkpoint weights')
            self.weights = self.checkpoint.upload(device,pack_experts=True)
            if progress:progress(f'uploaded {sum(b.nbytes for b in self.weights.values())} bytes')
            self._allocate()
            self.plan = self._plan()
            if any(row.get('schedule') for row in value['kernels'].values()):
                from .pipeline import replace
                replace(self,self.bundle)
            if graphs:
                self.graph = CudaGraph(device, self._submit,
                    resources=(*self.weights.values(),*self.buffers.values(),
                               *(b for state in self.states.values() for b in state),*self.kernels.values()))
        except BaseException:
            self.close()
            raise

    def _check(self):
        if self.closed or self._generation != self.device._generation:
            raise RuntimeError('Qwen batch is closed or its device session has expired')
        if self.thread != get_ident():
            raise RuntimeError('Qwen batch must execute on its owner thread')
        self.device._check()

    def _allocate(self):
        r, c = self.slots, self.config.width
        def alloc(name,shape,dtype='float32',zero=False):
            self.buffers[name] = (self.device.from_numpy(np.zeros(shape,dtype=dtype)) if zero
                                  else self.device.empty(shape,dtype))
        for name in ('tokens','positions','active'):
            alloc(name,(r,),'int32',True)
        for name in ('residual','normal'):
            alloc(name,(r,c),'bfloat16')
        for name,columns in (('mixed',c),('ffn',c),('qkv',8192),('z',4096),('a',32),('b',32),
                ('conv',8192),('attq',8192),('attk',512),('attv',512),('router',256),
                ('shared_gate',1),('shared_g',512),('shared_u',512),('shared_out',2048)):
            alloc(name,(r,columns))
        alloc('attq_normal',(r,16,256),'bfloat16')
        alloc('att_parts',(r,2,self.splits,8,256))
        alloc('att_stats',(r,2,self.splits,8,2))
        alloc('att_out',(r,4096),'bfloat16')
        for name in ('gdn_q','gdn_k','gdn_v','gdn_out'):
            alloc(name,(r,32,128))
        for name in ('gdn_decay','gdn_beta'):
            alloc(name,(r,32))
        alloc('gdn_normal',(r,4096),'bfloat16')
        alloc('ids',(r,8),'int32')
        alloc('routes',(r*8,r),'int32')
        alloc('experts',(r*8,),'int32')
        alloc('router_weights',(r,8))
        for name,columns in (('expert_g',512),('expert_u',512),('expert_out',2048)):
            alloc(name,(r,8,columns))
        alloc('expert_activation',(r,8,512),'bfloat16')
        alloc('shared_activation',(r,512),'bfloat16')
        alloc('logits',(r,self.config.vocab))
        self._allocate_states()

    def _allocate_states(self):
        r=self.slots
        for layer,kind in enumerate(self.config.layers):
            values = []
            self.states[layer] = values
            if kind=='linear_attention':
                for shape in ((r,32,128,128),(r,8192,3)):
                    values.append(self.device.from_numpy(np.zeros(shape,dtype='float32')))
            else:
                for _ in range(2):
                    values.append(self.device.empty((r,2,self.context,256),'uint8' if self.kv_dtype=='fp8' else 'bfloat16'))
                if self.kv_dtype=='fp8':
                    for _ in range(2):values.append(self.device.empty((r,2,self.context,2),'float32'))

    def reset_cache_profile(self,bundle):
        """Select an AOT cache profile between requests, retaining the weights.

        All slots must have been reset and prefill graphs closed. A failed
        rebuild closes the executor, since cache replacement releases the old
        allocation before creating the new one to stay within GPU capacity.
        """
        self._check()
        if self.prefills or self.position.any() or self.active.any():
            raise ValueError('reset all requests and close prefill graphs before changing KV profile')
        bundle=Path(bundle).resolve();value=json.loads((bundle/'inference.json').read_text())
        if (value.get('schema')!='tensor.qwen35-batch.v1'
                or value.get('implementation')!=implementation_hashes()
                or value.get('config')!=json.loads(json.dumps(asdict(self.config)))
                or value.get('target')!=self.device.info['arch']
                or (value['slots'],value['context'],value['splits'])!=(self.slots,self.context,self.splits)
                or value.get('kv_dtype','bfloat16') not in ('bfloat16','fp8')):
            raise ValueError('replacement cache profile configuration/source mismatch')
        had_graph=self.graph is not None
        if self.graph:self.graph.close();self.graph=None
        try:
            for kernel in self.kernels.values():kernel.release()
            self.kernels={}
            for name in list(self.buffers):
                if name.startswith('_tune_'):self.buffers.pop(name).release()
            for state in self.states.values():
                for buffer in state:buffer.release()
            self.states={}
            if hasattr(self,'_logical_plan'):del self._logical_plan
            self.kv_dtype=value.get('kv_dtype','bfloat16');self.bundle=bundle
            for key,(kind,p) in requirements(self.config,self.slots,self.context,splits=self.splits,kv_dtype=self.kv_dtype).items():
                row=value['kernels'][key];descriptor=row.get('logical',row)
                artifact=(bundle/descriptor['path']).resolve()
                if (not artifact.is_relative_to(bundle)
                        or hashlib.sha256(artifact.read_bytes()).hexdigest()!=descriptor['sha256']
                        or row['kind']!=kind or row['parameters']!=p):raise ValueError('cache profile artifact mismatch')
                self.kernels[key]=self.device.load(artifact)
            self._allocate_states();self.plan=self._plan()
            if any(row.get('schedule') for row in value['kernels'].values()):
                from .pipeline import replace
                replace(self,bundle)
            if had_graph:
                self.graph=CudaGraph(self.device,self._submit,resources=(*self.weights.values(),*self.buffers.values(),
                    *(b for state in self.states.values() for b in state),*self.kernels.values()))
        except BaseException:
            self.close();raise
        return self

    def _plan(self):
        w, b, r = self.weights,self.buffers,self.slots
        calls = []
        def add(kind,p,*args,label=None):
            kernel = self.kernels[identity(kind,p)]
            values,symbols,launch = kernel._bind(args,{},include_outputs=True)
            calls.append((kernel,BoundCall(self.device,kernel.manifest,values,symbols,launch,validated=True),label))
        def fp8(name,x,out):
            weight = w[name+'.weight']
            o,k = weight.shape
            add('fp8_linear',dict(r=r,k=k,o=o),x,weight,w[name+'.weight_scale_inv'],out)
        def bf16(name,x,out):
            weight = w[name+'.weight']
            o,k = weight.shape
            add('bf16_linear',dict(r=r,k=k,o=o),x,weight,out)
        def norm(name,x=None,label=None):
            args = (b['residual'],w[name+'.weight'],b['normal'])
            add('rms' if x is None else 'add_rms',dict(r=r,c=2048,eps=self.config.epsilon),
                *((x,*args) if x is not None else args),label=label)
        add('embedding',dict(r=r,c=2048,vocab=self.config.vocab),b['tokens'],w['model.language_model.embed_tokens.weight'],b['residual'])
        for layer,kind in enumerate(self.config.layers):
            root = f'model.language_model.layers.{layer}.'
            norm(root+'input_layernorm',None if layer==0 else b['ffn'],label=(layer,'input'))
            if kind=='linear_attention':
                prefix=root+'linear_attn.'
                for name,dest in (('in_proj_qkv','qkv'),('in_proj_z','z')):
                    fp8(prefix+name,b['normal'],b[dest])
                for name in ('a','b'):
                    bf16(prefix+'in_proj_'+name,b['normal'],b[name])
                state,history = self.states[layer]
                add('gdn_conv',dict(r=r),b['qkv'],w[prefix+'conv1d.weight'],b['active'],history,b['conv'])
                add('gdn_prepare',dict(r=r),b['conv'],b['a'],b['b'],w[prefix+'dt_bias'],w[prefix+'A_log'],
                    b['gdn_q'],b['gdn_k'],b['gdn_v'],b['gdn_decay'],b['gdn_beta'])
                add('gdn_recurrent',dict(slots=r,heads=32,key=128,value=128),b['gdn_q'],b['gdn_k'],b['gdn_v'],
                    b['gdn_decay'],b['gdn_beta'],b['active'],state,b['gdn_out'])
                add('gdn_norm',dict(r=r,eps=self.config.epsilon),b['gdn_out'],b['z'],w[prefix+'norm.weight'],b['gdn_normal'])
                fp8(prefix+'out_proj',b['gdn_normal'],b['mixed'])
            else:
                prefix=root+'self_attn.'
                for name,dest in (('q','attq'),('k','attk'),('v','attv')):
                    fp8(prefix+name+'_proj',b['normal'],b[dest])
                kc,vc,*scales = self.states[layer]
                p=dict(r=r,eps=self.config.epsilon,capacity=self.context,theta=self.config.rope_theta)
                add('attention_q',p,b['attq'],w[prefix+'q_norm.weight'],b['positions'],b['active'],b['attq_normal'])
                kv = dict(kv_dtype='fp8') if self.kv_dtype=='fp8' else {}
                add('attention_kv',dict(p,**kv),b['attk'],b['attv'],w[prefix+'k_norm.weight'],b['positions'],b['active'],kc,vc,*scales)
                add('attention_partial',dict(r=r,capacity=self.context,splits=self.splits,**kv),b['attq_normal'],kc,vc,*scales,
                    b['positions'],b['active'],b['att_parts'],b['att_stats'])
                add('attention_merge',dict(r=r,splits=self.splits),b['att_parts'],b['att_stats'],b['attq'],b['active'],b['att_out'])
                fp8(prefix+'o_proj',b['att_out'],b['mixed'])
            norm(root+'post_attention_layernorm',b['mixed'],label=(layer,'post_attention'))
            prefix=root+'mlp.'
            bf16(prefix+'gate',b['normal'],b['router'])
            bf16(prefix+'shared_expert_gate',b['normal'],b['shared_gate'])
            add('router',dict(r=r,experts=256,top=8),b['router'],b['ids'],b['router_weights'])
            add('moe_groups',dict(r=r,top=8),b['ids'],b['experts'],b['routes'])
            for name,dest in (('gate_proj','expert_g'),('up_proj','expert_u')):
                add('fp8_experts',dict(r=r,experts=256,top=8,o=512,k=2048,routed_input=False),
                    b['normal'],w[prefix+'experts.'+name+'.weight'],w[prefix+'experts.'+name+'.weight_scale_inv'],
                    b['experts'],b['routes'],b[dest])
            add('swiglu_experts',dict(r=r,top=8,c=512),b['expert_g'],b['expert_u'],b['expert_activation'])
            add('fp8_experts',dict(r=r,experts=256,top=8,o=2048,k=512,routed_input=True),
                b['expert_activation'],w[prefix+'experts.down_proj.weight'],w[prefix+'experts.down_proj.weight_scale_inv'],
                b['experts'],b['routes'],b['expert_out'])
            for name,dest in (('gate_proj','shared_g'),('up_proj','shared_u')):
                fp8(prefix+'shared_expert.'+name,b['normal'],b[dest])
            add('swiglu',dict(r=r,c=512),b['shared_g'],b['shared_u'],b['shared_activation'])
            fp8(prefix+'shared_expert.down_proj',b['shared_activation'],b['shared_out'])
            add('moe_combine',dict(r=r,top=8,c=2048),b['expert_out'],b['router_weights'],b['shared_out'],b['shared_gate'],b['ffn'],label=(layer,'mlp'))
        norm('model.language_model.norm',b['ffn'],label=(40,'final'))
        bf16('lm_head',b['normal'],b['logits'])
        add('argmax',dict(r=r,vocab=self.config.vocab),b['logits'],b['tokens'],b['positions'],b['active'])
        return calls

    def _submit(self):
        for kernel,call,_ in self.plan:
            self.device._launch(kernel,call)

    def _write(self,name,value):
        buffer = self.buffers[name]
        array = np.ascontiguousarray(value,dtype='int32')
        if array.shape != buffer.shape:
            raise ValueError('control buffer shape mismatch')
        self.device.driver.call('cuMemcpyHtoD_v2',buffer.pointer,ct.c_void_p(array.ctypes.data),array.nbytes)
        self.device.driver.call('cuStreamSynchronize',None)

    def forward(self,tokens,*,read_logits=False,debug=None):
        self._check()
        value = np.asarray(tokens)
        if value.shape != (self.slots,) or value.dtype.kind not in 'iu':
            raise ValueError('one integer token ID per slot is required; -1 marks an inactive slot')
        if np.any(value < -1) or np.any(value >= self.config.vocab):
            raise ValueError('invalid Qwen token ID')
        active = (value >= 0).astype('int32')
        if np.any(self.position[active!=0] >= self.context):
            raise ValueError('Qwen context capacity exhausted')
        self._write('tokens',value)
        self._write('active',active)
        self.active = active
        return self.step(read_logits=read_logits,debug=debug)

    def step(self,*,read_logits=False,debug=None):
        self._check()
        if np.any(self.position[self.active!=0] >= self.context):
            raise ValueError('Qwen context capacity exhausted')
        if debug is not None:
            for kernel,call,label in self.plan:
                self.device._launch(kernel,call)
                if label is not None:
                    debug(label,self.buffers['ffn' if label[1]=='mlp' else 'normal'].to_numpy())
        elif self.graph is not None:
            self.graph.launch()
        else:
            self._submit()
        self.position += self.active
        result = self.buffers['tokens'].to_numpy()
        return (result,self.buffers['logits'].to_numpy()) if read_logits else result

    def reset(self,slots=None):
        self._check()
        indices = list(range(self.slots)) if slots is None else list(slots)
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0<=i<self.slots for i in indices):
            raise ValueError('invalid Qwen reset slots')
        self.device.synchronize()
        for layer,kind in enumerate(self.config.layers):
            if kind != 'linear_attention':continue
            for buffer in self.states[layer]:
                size = buffer.nbytes // self.slots
                zeros = np.zeros(size//4,dtype='float32')
                for index in indices:
                    self.device.driver.call('cuMemcpyHtoD_v2',buffer.pointer+index*size,
                        ct.c_void_p(zeros.ctypes.data),size)
                self.device.driver.call('cuStreamSynchronize',None)
        self.position[indices]=0
        self.active[indices]=0
        self._write('positions',self.position)
        self._write('active',self.active)

    @property
    def allocated_bytes(self):
        return sum(b.nbytes for b in (*self.weights.values(),*self.buffers.values(),
            *(b for state in self.states.values() for b in state)))

    def close(self):
        if self.closed:return
        for prefill in tuple(self.prefills):prefill.close()
        if self.graph is not None:self.graph.close();self.graph=None
        for buffer in (*self.weights.values(),*self.buffers.values(),
                       *(b for state in self.states.values() for b in state)):
            buffer.release()
        for kernel in self.kernels.values():kernel.release()
        self.weights,self.buffers,self.states,self.kernels={},{},{},{}
        self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
