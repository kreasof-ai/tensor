"""Runtime-only binding of AOT projection pipelines to resident model buffers."""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall
from tensor.runtime.signature import resolve_shape


def replace(model,bundle):
    model._check()
    if getattr(model,'prefills',()):raise ValueError('close prefill graphs before replacing decoder pipelines')
    bundle=Path(bundle).resolve()
    manifest=json.loads((bundle/'inference.json').read_text())
    if (manifest['schema']!='tensor.qwen35-batch.v1'
            or manifest['config']!=json.loads(json.dumps(asdict(model.config)))
            or manifest['slots']!=model.slots or manifest['context']!=model.context
            or manifest.get('kv_dtype','bfloat16')!=model.kv_dtype
            or manifest['target']!=model.device.info['arch']):
        raise ValueError('replacement bundle capacity/target mismatch')
    if not hasattr(model,'_logical_plan'):
        keys={kernel:key for key,kernel in model.kernels.items()}
        logical=[]
        for kernel,call,label in model.plan:
            abi=kernel.manifest.get('abi',kernel.manifest['arguments'])
            if len(abi)!=len(call.storage):raise ValueError('tuning supports buffer-only calls')
            logical.append((keys[kernel],{item['name']:buffer for item,buffer in zip(abi,call.storage)},label))
        model._logical_plan=logical
    new_kernels={};new_plan=[];scratch={}
    def load(key,row):
        path=(bundle/row['path']).resolve()
        if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('replacement artifact checksum mismatch')
        kernel=model.device.load(path);new_kernels[key]=kernel
        return kernel
    def bind(kernel,bindings,label):
        arguments=tuple(bindings[item['name']] for item in kernel.manifest['arguments'])
        values,symbols,launch=kernel._bind(arguments,{},include_outputs=True)
        new_plan.append((kernel,BoundCall(model.device,kernel.manifest,values,symbols,launch,validated=True),label))
    try:
        for key in {entry[0] for entry in model._logical_plan}:
            row=manifest['kernels'][key]
            kernel=load(key,row)
            if row.get('quantize'):load(key+'-quantize',row['quantize'])
            if row.get('merge'):
                descriptor=next(a for a in kernel.manifest['arguments'] if a['name']=='out')
                scratch[key]=model.device.empty(resolve_shape(descriptor['shape'],{}),descriptor['dtype'])
                load(key+'-merge',row['merge'])
        quantized={};valid=set()
        for key,bindings,label in model._logical_plan:
            kernel=new_kernels[key]
            row=manifest['kernels'][key]
            if row.get('quantize'):
                x=bindings['x'];identity=id(x)
                quantizer=new_kernels[key+'-quantize']
                if identity not in quantized:
                    allocated={}
                    for name in ('out','scales'):
                        descriptor=next(a for a in quantizer.manifest['arguments'] if a['name']==name)
                        allocated[name]=model.device.empty(resolve_shape(descriptor['shape'],{}),descriptor['dtype'])
                        scratch[f'quantize-{identity}-{name}']=allocated[name]
                    quantized[identity]=allocated
                allocated=quantized[identity]
                if identity not in valid:
                    bind(quantizer,dict(x=x,**allocated),None)
                    valid.add(identity)
                bindings={**bindings,'x':allocated['out'],'activation_scales':allocated['scales']}
            if key in scratch:
                bind(kernel,{**bindings,'out':scratch[key]},None)
                bind(new_kernels[key+'-merge'],dict(partial=scratch[key],out=bindings['out']),label)
            else:bind(kernel,bindings,label)
            # An operation that overwrites an input workspace invalidates its
            # cached quantization. Readers sharing that workspace reuse one
            # quantization until the next producer writes it.
            if 'out' in bindings:valid.discard(id(bindings['out']))
    except BaseException:
        for kernel in new_kernels.values():kernel.release()
        for buffer in scratch.values():buffer.release()
        raise
    old_plan,old_kernels,old_buffers=model.plan,model.kernels,model.buffers
    had_graph=model.graph is not None
    if model.graph:model.graph.close();model.graph=None
    model.plan,model.kernels=new_plan,new_kernels
    model.buffers={**{n:b for n,b in old_buffers.items() if not n.startswith('_tune_')},
                   **{'_tune_'+key:buffer for key,buffer in scratch.items()}}
    try:
        if had_graph:
            model.graph=CudaGraph(model.device,model._submit,
                resources=(*model.weights.values(),*model.buffers.values(),
                    *(b for state in model.states.values() for b in state),*new_kernels.values()))
    except BaseException:
        model.plan,model.kernels,model.buffers=old_plan,old_kernels,old_buffers
        for kernel in new_kernels.values():kernel.release()
        for buffer in scratch.values():buffer.release()
        if had_graph:
            model.graph=CudaGraph(model.device,model._submit,
                resources=(*model.weights.values(),*model.buffers.values(),
                    *(b for state in model.states.values() for b in state),*old_kernels.values()))
        raise
    for kernel in old_kernels.values():kernel.release()
    for name,buffer in old_buffers.items():
        if name.startswith('_tune_'):buffer.release()
    return model
