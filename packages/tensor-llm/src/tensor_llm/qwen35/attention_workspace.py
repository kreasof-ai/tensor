"""Own a transient BF16 attention workspace while retaining FP8 KV storage."""
import hashlib
import json
from pathlib import Path
from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall
from tensor.runtime.cuda_target import matches_device


def implementation_hashes():
    return {__name__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def install(executor,bundle):
    executor.model._check()
    if executor.closed or getattr(executor,'pending_verification',False):
        raise RuntimeError('executor must be idle')
    if getattr(executor,'_attention_workspace_installed',False):
        raise ValueError('attention workspace already installed')
    bundle=Path(bundle).resolve()
    manifest=json.loads((bundle/'attention-workspace.json').read_text())
    if manifest['implementation']!=implementation_hashes():
        raise ValueError('attention workspace implementation mismatch; rebuild bundle')
    if (manifest['slots'],manifest['chunk'],manifest['capacity'])!=(
            executor.model.slots,executor.chunk,executor.model.context):
        raise ValueError('attention workspace capacity mismatch')
    if not matches_device(manifest['target'],executor.device.info['arch']):
        raise ValueError('attention workspace target mismatch')
    d=executor.device;loaded={};buffers={}
    try:
        for name,row in manifest['kernels'].items():
            path=(bundle/row['path']).resolve()
            if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('attention workspace artifact checksum mismatch')
            loaded[name]=d.load(path)
        shape=(manifest['slots'],2,manifest['capacity'],256)
        buffers={name:d.empty(shape,'bfloat16') for name in ('decoded_k','decoded_v')}
        def bind(kernel,values):
            args=tuple(values[a['name']] for a in kernel.manifest['arguments'])
            storage,symbols,launch=kernel._bind(args,{},include_outputs=True)
            return kernel,BoundCall(d,kernel.manifest,storage,symbols,launch,validated=True)
        plan=[];replaced=0
        for kernel,bound in executor.plan:
            identity=kernel.manifest['files']['kernel.cubin']
            if identity!=manifest['original_cubin_sha256']:
                plan.append((kernel,bound));continue
            values=dict(zip((a['name'] for a in kernel.manifest['abi']),bound.storage))
            plan.append(bind(loaded['decode'],dict(values,**buffers)))
            plan.append(bind(loaded['attention'],dict(values,kc=buffers['decoded_k'],vc=buffers['decoded_v'])))
            replaced+=1
        if replaced!=manifest['expected_attention_calls']:
            raise ValueError('attention workspace source plan mismatch')
        graph=CudaGraph(d,lambda:[d._launch(k,v) for k,v in plan],
            resources=(*executor.graph.resources,*loaded.values(),*buffers.values()))
    except BaseException:
        for kernel in loaded.values():kernel.release()
        for buffer in buffers.values():buffer.release()
        raise
    executor.graph.close();executor.graph=graph;executor.plan=plan
    executor.kernels.update({'_attention_workspace_'+k:v for k,v in loaded.items()})
    executor.buffers.update({'_attention_workspace_'+k:v for k,v in buffers.items()})
    executor._attention_workspace_installed=True
