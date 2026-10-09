"""Install split-context attention in an existing native chunk execution graph."""
import hashlib
import json
from pathlib import Path
from tensor.runtime.abi import BoundCall
from tensor.providers.cuda_graph import CudaGraph
from ..artifacts import identity


def implementation_hashes():
    from ..kernels import speculative_attention
    return {m.__name__:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (speculative_attention,)} | {__name__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def install(prefill, bundle):
    prefill.model._check()
    if prefill.closed or getattr(prefill,'pending_verification',False):
        raise RuntimeError('split attention needs an idle live executor')
    bundle=Path(bundle).resolve();manifest=json.loads((bundle/'prefill.json').read_text())
    if manifest.get('split_attention_implementation')!=implementation_hashes():
        raise ValueError('split attention implementation mismatch')
    s,c,cap=prefill.model.slots,prefill.chunk,prefill.model.context
    if (manifest['slots'],manifest['chunk'],manifest['context'],manifest.get('kv_dtype'))!=(s,c,cap,'fp8'):
        raise ValueError('split attention capacity or precision mismatch')
    splits=manifest['split_attention']['splits'];d=prefill.device;b=prefill.buffers
    if '_split_parts' in b:raise RuntimeError('split attention is already installed')
    params=dict(slots=s,chunk=c,capacity=cap,splits=splits)
    loaded=[];allocated=[]
    try:
        for kind in ('split_attention','split_merge'):
            key=identity(kind,params);row=manifest['kernels'][key];path=(bundle/row['path']).resolve()
            if not path.is_relative_to(bundle) or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('split attention checksum mismatch')
            kernel=d.load(path);loaded.append((key,kernel))
        parts=d.empty((s,2,splits,c*8,256));allocated.append(parts)
        stats=d.empty((s,2,splits,c*8,2));allocated.append(stats)
        old=prefill.kernels[identity('attention',dict(slots=s,chunk=c,capacity=cap,kv_dtype='fp8'))]
        plan=[]
        def bind(kernel,values):
            args=tuple(values[a['name']] for a in kernel.manifest['arguments'])
            values,symbols,launch=kernel._bind(args,{},include_outputs=True)
            return kernel,BoundCall(d,kernel.manifest,values,symbols,launch,validated=True)
        for kernel,bound in prefill.plan:
            if kernel is not old:plan.append((kernel,bound));continue
            values=dict(zip((a['name'] for a in kernel.manifest.get('abi',kernel.manifest['arguments'])),bound.storage))
            out=values.pop('out');projection=values.pop('projection')
            values.update(out=parts,stats=stats)
            plan.append(bind(loaded[0][1],values))
            plan.append(bind(loaded[1][1],dict(partial=parts,stats=stats,projection=projection,
                active=b['flat_active'],out=out)))
        if len(plan)==len(prefill.plan):raise ValueError('no chunk attention calls found')
        resources=(*prefill.graph.resources,*allocated,*(k for _,k in loaded))
        graph=CudaGraph(d,lambda:[d._launch(k,v) for k,v in plan],resources=resources)
    except BaseException:
        for value in allocated:value.release()
        for _,kernel in loaded:kernel.release()
        raise
    prefill.graph.close();prefill.graph=graph;prefill.plan=plan
    b['_split_parts'],b['_split_stats']=parts,stats
    prefill.kernels.update(loaded)
