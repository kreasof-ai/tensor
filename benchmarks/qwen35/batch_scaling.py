"""Memory accounting and independent-slot qualification for larger batches."""
import ctypes as ct
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def workload(slots, path='/workspace/workload-c64.json.gz'):
    from benchmarks.llm_serving.workload import digest,validate
    with gzip.open(path,'rt') as source:value=json.load(source)
    validate(value)
    if slots not in (8,16,32,64):raise ValueError('unsupported scaling point')
    value['requests']=value['requests'][:slots]
    value['sha256']=digest({k:v for k,v in value.items() if k!='sha256'})
    return validate(value)


def allocation_plan(prepared):
    """Execute allocation shapes on a counting device; no GPU storage is created."""
    from tensor_llm.qwen35.checkpoint import Qwen35Checkpoint
    from tensor_llm.qwen35.decode import Qwen35Batch
    from tensor_llm.qwen35.prefill import Qwen35Prefill
    from tensor_llm.qwen35.speculative.recompute import Qwen35RecomputeVerifier
    from tensor_llm.qwen35.speculative.verifier import Qwen35Verifier
    sizes={'float32':4,'int32':4,'bfloat16':2,'uint8':1}
    class Device:
        def empty(self,shape,dtype='float32'):
            return SimpleNamespace(shape=tuple(shape),dtype=dtype,nbytes=int(np.prod(shape))*sizes[dtype])
        def from_numpy(self,value):return self.empty(value.shape,str(value.dtype))
    device=Device(); checkpoint=Qwen35Checkpoint(prepared['base']['checkpoint'])
    config=checkpoint.config;slots=prepared['slots']
    def model(c):
        value=Qwen35Batch.__new__(Qwen35Batch)
        value.device=device;value.config=c;value.slots=slots;value.context=48000
        value.splits=16;value.kv_dtype='fp8';value.buffers={};value.states={}
        value._allocate()
        return value
    target=model(config);draft=model(replace(config,layers=('full_attention',)))
    for name,columns,dtype in [('mtp_hidden',2048,'bfloat16'),('mtp_join',4096,'bfloat16'),('mtp_fc',2048,'float32')]:
        draft.buffers[name]=device.empty((slots,columns),dtype)
    def context(owner,bundle,recomputed=False,snapshots=False,extra_workspace=None):
        manifest=json.loads((Path(bundle)/'prefill.json').read_text())
        cls=Qwen35RecomputeVerifier if recomputed else Qwen35Verifier if snapshots else Qwen35Prefill
        value=cls.__new__(cls)
        value.model=owner;value.device=device;value.chunk=manifest['chunk']
        value.rows=slots*manifest['chunk'];value.buffers={}
        cls._allocate(value)
        count=sum(b.nbytes for b in value.buffers.values())
        count+=sum(int(np.prod(row['partial_shape']))*4 for row in manifest.get('split_linear',{}).values()
                   if 'partial_shape' in row)
        if 'split_attention' in manifest:
            splits=manifest['split_attention']['splits']
            count+=slots*2*splits*manifest['chunk']*8*(256+2)*4
        if 'compact_experts' in manifest:count+=8*manifest['compact_experts']['max_tiles']
        if 'attention_workspace' in manifest or extra_workspace:
            count+=2*slots*2*48000*256*2
        return count
    paths=prepared['base']['paths']
    owned_mtp=Qwen35Checkpoint(prepared['base']['checkpoint'],branch='mtp').weight_bytes
    core=checkpoint.weight_bytes+owned_mtp
    core+=sum(b.nbytes for m in (target,draft) for b in m.buffers.values())
    core+=sum(b.nbytes for m in (target,draft) for state in m.states.values() for b in state)
    prefix=context(target,prepared['prefill'],extra_workspace=prepared['attention_workspace'])
    draft_prefix=context(draft,paths['draft_prefill'])
    verify=context(target,paths['verify'],recomputed=True)
    repair=context(draft,paths['repair'])
    adaptive_verify=adaptive_repair=0
    manifest=json.loads((Path(paths['verify'])/'prefill.json').read_text())
    for row in manifest.get('adaptive_verification',{}).get('profiles',[]):
        adaptive_verify+=context(target,Path(paths['verify'])/row['verify'],snapshots=True)
        adaptive_repair+=context(draft,Path(paths['verify'])/row['repair'])
    decoding=verify+repair+adaptive_verify+adaptive_repair
    # Nonresident serving releases both prefix contexts before capturing the
    # paired verification pool; no precision or request-state storage changes.
    total=core+(prefix+draft_prefix+decoding if prepared['resident_speculative_graphs']
                else max(prefix+draft_prefix,decoding))
    return dict(core_bytes=core,prefill_bytes=prefix,draft_prefill_bytes=draft_prefix,
                verification_bytes=verify,repair_bytes=repair,
                adaptive_verification_bytes=adaptive_verify,adaptive_repair_bytes=adaptive_repair,
                resident_speculative_graphs=prepared['resident_speculative_graphs'],known_peak_bytes=total,
                reserve_bytes=6*2**30,required_with_reserve_bytes=total+6*2**30,
                exclusions=['CUDA modules/graphs/driver overhead covered by reserve',
                            'small decoder Split-K and quantization scratch covered by reserve'])


def install_prefill(model,prepared):
    from tensor_llm import Qwen35Prefill
    from tensor_llm.qwen35.compact_prefill import install as compact
    from tensor_llm.qwen35.hopper import install as hopper
    from tensor_llm.qwen35.attention_workspace import install as workspace
    value=Qwen35Prefill(model,prepared['prefill'])
    try:
        compact(value,prepared['prefill']);hopper(value,prepared['hopper'])
        workspace(value,prepared['attention_workspace'])
        return value
    except BaseException:value.close();raise


def digest(value):return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def state_records(model,workers,*,cache_start=0):
    """Hash each slot separately, excluding undefined future cache rows."""
    result=[{} for _ in range(model.slots)]
    for layer,buffers in model.states.items():
        recurrent=model.config.layers[layer]=='linear_attention'
        for index,buffer in enumerate(buffers):
            values=buffer.to_numpy()
            def one(slot):
                value=values[slot] if recurrent else values[slot,:,cache_start:int(model.position[slot]),:]
                return slot,digest(value)
            for slot,sha in workers.map(one,range(model.slots)):result[slot][f'{layer}:{index}']=sha
    return result


def observe(model,prepared,tokens):
    """Real-prefix forward and commits; no reference logits are used for serving."""
    from benchmarks.qwen35.whole_tune import Snapshot
    from benchmarks.qwen35.spec_run import make_verifier,install_selected
    from tensor.runtime.dtypes import encode_bfloat16
    model.reset();prefill=install_prefill(model,prepared)
    try:
        for offset in range(0,tokens.shape[1],prefill.chunk):
            count=min(prefill.chunk,tokens.shape[1]-offset)
            block=np.zeros((model.slots,prefill.chunk),'int32');block[:,:count]=tokens[:,offset:offset+count]
            prefill.forward(block,np.full(model.slots,count,'int32'))
    finally:prefill.close()
    prefix=Snapshot(model);normal=encode_bfloat16(model.buffers['normal'].to_numpy())
    with ThreadPoolExecutor(max_workers=min(16,model.slots)) as workers:
        logits=model.buffers['logits'].to_numpy()
        records=[dict(prefill_token=int(prefix.tokens[s]),prefill_logits=digest(logits[s]),
                      prefix_position=int(prefix.position[s])) for s in range(model.slots)]
        del logits
        states=state_records(model,workers)
        for record,state in zip(records,states):record['prefix_states']=state
        verifier=make_verifier(model,prepared['base']['paths']['verify'])
        try:
            install_selected(verifier,prepared['base']['paths']['verify'])
            chunk=verifier.chunk
            lengths=np.resize(np.array([chunk,chunk-1,chunk//2+5,0,1,3,chunk,3*chunk//4],'int32'),model.slots)
            proposals=np.repeat(prefix.tokens[:,None],chunk,axis=1)
            predictions,logits=verifier.forward(proposals,lengths,read_logits=True)
            logits=logits.reshape(model.slots,chunk,-1)
            for slot,record in enumerate(records):
                valid=int(lengths[slot]);record['length']=valid
                record['verify_predictions']=digest(predictions[slot,:valid])
                record['verify_logits']=digest(logits[slot,:valid])
                record['finite']=bool(np.isfinite(logits[slot,:valid]).all())
                record['commits']=[]
            del logits
            states=state_records(model,workers,cache_start=tokens.shape[1])
            for record,state in zip(records,states):record['verified_states']=state
            for depth in (1,2,3,17,65,127,128):
                if not verifier.pending_verification:
                    prefix.restore(model)
                    model.device.driver.call('cuMemcpyHtoD_v2',model.buffers['normal'].pointer,
                        ct.c_void_p(normal.ctypes.data),normal.nbytes)
                    model.device.driver.call('cuStreamSynchronize',None)
                    verifier.forward(proposals,lengths)
                verifier.commit(np.minimum(lengths,depth).astype('int32'))
                states=state_records(model,workers,cache_start=tokens.shape[1])
                hidden=model.buffers['normal'].to_numpy()
                for slot,record in enumerate(records):
                    record['commits'].append(dict(depth=depth,position=int(model.position[slot]),
                        states=states[slot],normal=digest(hidden[slot])))
            # Ensure speculative writes did not change the already initialized prefix.
            position=model.position.copy();model.position=prefix.position.copy()
            final_prefix=state_records(model,workers)
            model.position=position
            for slot,record in enumerate(records):
                record['kv_prefix_unchanged']=all(final_prefix[slot][key]==record['prefix_states'][key]
                    for key in final_prefix[slot] if model.config.layers[int(key.split(':')[0])]=='full_attention')
        finally:verifier.close();prefix.restore(model)
    return records


def compare_records(expected,actual):
    if len(expected)!=len(actual):raise ValueError('slot coverage mismatch')
    differences=[dict(slot=s,components=[k for k in a if a[k]!=b.get(k)])
                 for s,(a,b) in enumerate(zip(expected,actual)) if a!=b]
    return dict(slots=len(actual),differences=differences,
                finite=all(r['finite'] for r in actual),
                kv_prefix_unchanged=all(r['kv_prefix_unchanged'] for r in actual),
                passed=not differences and all(r['finite'] and r['kv_prefix_unchanged'] for r in actual),
                canonical_model_qualified=False,model_throughput_qualified=False)
