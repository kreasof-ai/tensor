"""Native fixed-cohort MTP decoding: batched verify, prefix commit, draft repair.

Run on the resident owner thread. All timing includes host control, target
verification, recurrent rollback, and correction of the private MTP cache.
"""
import hashlib
import json
from pathlib import Path
import time
import ctypes as ct
import numpy as np
from tensor_llm.qwen_spec import accepted_prefix


class PrefixSnapshot:
    """Restore a tuning prefix; append-only KV prefixes need no full cache copy."""
    def __init__(self,model,mtp):
        from benchmarks.qwen35.whole_tune import Snapshot
        self.target=Snapshot(model);self.draft=Snapshot(mtp)
        self.normal=np.empty(mtp.buffers['normal'].shape,dtype='uint16')
        mtp.device.synchronize()
        mtp.device.driver.call('cuMemcpyDtoH_v2',ct.c_void_p(self.normal.ctypes.data),
                              mtp.buffers['normal'].pointer,self.normal.nbytes)

    def restore(self,model,mtp):
        self.target.restore(model);self.draft.restore(mtp)
        mtp.device.driver.call('cuMemcpyHtoD_v2',mtp.buffers['normal'].pointer,
                              ct.c_void_p(self.normal.ctypes.data),self.normal.nbytes)
        mtp.device.driver.call('cuStreamSynchronize',None)


def initialize(model,mtp,target_prefill,draft_prefill,tokens):
    """Initialize matched target/MTP prefix states and cache the first draft."""
    tokens=np.asarray(tokens)
    if (tokens.ndim!=2 or tokens.shape[1]!=model.slots or len(tokens)<1
            or target_prefill.chunk!=draft_prefill.chunk):
        raise ValueError('expected [steps,slots] prefix and equal prefill chunks')
    started=time.perf_counter();model.reset();mtp.reset();chunk=target_prefill.chunk
    for start in range(0,len(tokens),chunk):
        count=min(chunk,len(tokens)-start)
        inputs=np.zeros((model.slots,chunk),'int32');inputs[:,:count]=tokens[start:start+count].T
        lengths=np.full(model.slots,count,'int32')
        pending=target_prefill.forward(inputs,lengths)
        shifted=np.zeros_like(inputs)
        if count>1:shifted[:,:count-1]=inputs[:,1:count]
        shifted[:,count-1]=tokens[start+count] if start+count<len(tokens) else pending
        draft_prefill.forward(shifted,target_prefill.buffers['normal'],lengths)
        if (start//chunk+1)%16==0:print('Target/MTP stress prefix',start+count,flush=True)
    target_prefill.close();draft_prefill.close()
    return time.perf_counter()-started


def run(model, mtp, verifier, repair, out, *, output_tokens=16000, prefill_seconds=0.,
        initial_output=True, progress_every=128, output_lookup=False):
    model._check(); mtp._check()
    if (verifier.model is not model or repair.model is not mtp or verifier.chunk!=repair.chunk
            or np.any(model.position!=mtp.position) or np.any(model.active!=1)
            or output_tokens<1 or np.any(model.position+output_tokens>model.context)):
        raise ValueError('speculative cohort needs initialized equal target/draft prefixes and context room')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    slots,chunk=model.slots,verifier.chunk
    pending=model.buffers['tokens'].to_numpy()
    generated=np.full(slots,1 if initial_output else 0,'int32')
    streams=[[int(pending[s])] if initial_output else [] for s in range(slots)]
    from tensor_llm.spec_lookup import OutputLookup
    lookups=[OutputLookup() for _ in range(slots)] if output_lookup else None
    if lookups and initial_output:
        for slot in range(slots):lookups[slot].append(pending[slot])
    records=[];started=time.perf_counter()
    while np.any(generated<output_tokens):
        round_start=time.perf_counter();base=model.position.copy()
        lengths=np.minimum(chunk,output_tokens-generated).astype('int32')
        inputs=np.zeros((slots,chunk),'int32');inputs[:,0]=pending
        inputs[:,1]=mtp.buffers['tokens'].to_numpy()
        lookup_mask=np.zeros(slots,bool)
        if lookups:
            for slot,length in enumerate(lengths):
                if length<=1:continue
                proposals=lookups[slot].propose(int(length)-1)
                if proposals is not None:
                    lookup_mask[slot]=True;inputs[slot,1:length]=proposals
        draft_start=time.perf_counter()
        for i in range(2,chunk):
            use=(lengths>i)&~lookup_mask
            if np.any(use):
                prediction=mtp.draft(np.where(use,inputs[:,i-1],-1),mtp.buffers['normal'])
                inputs[use,i]=prediction[use]
        draft_seconds=time.perf_counter()-draft_start
        verify_start=time.perf_counter();predictions=verifier.forward(inputs,lengths)
        verify_seconds=time.perf_counter()-verify_start
        counts,outputs=accepted_prefix(inputs,predictions,lengths)
        commit_start=time.perf_counter();verifier.commit(counts)
        commit_seconds=time.perf_counter()-commit_start
        # Replace approximate draft hidden states with the exact verified target
        # states, including the corrected/bonus token at each accepted prefix.
        repair_start=time.perf_counter()
        shifted=np.zeros_like(inputs)
        for slot,count in enumerate(counts):
            if count:
                shifted[slot,:count-1]=inputs[slot,1:count]
                pending[slot]=predictions[slot,count-1]
                shifted[slot,count-1]=pending[slot]
        mtp.position=base.copy();mtp._write('positions',mtp.position)
        repair.forward(shifted,verifier.buffers['normal'],counts)
        repair_seconds=time.perf_counter()-repair_start
        generated+=counts
        for slot,values in enumerate(outputs):
            streams[slot].extend(values)
            if lookups:
                for token in values:lookups[slot].append(token)
        records.append(dict(round=len(records),position=model.position.tolist(),accepted_inputs=counts.tolist(),
            draft_seconds=draft_seconds,verify_seconds=verify_seconds,commit_seconds=commit_seconds,
            repair_seconds=repair_seconds,seconds=time.perf_counter()-round_start))
        records[-1]['lookup_requests']=int(lookup_mask.sum())
        records[-1]['active_requests']=int((lengths>0).sum())
        records[-1]['output_tokens']=int(counts.sum())
        records[-1]['ended_seconds']=time.perf_counter()-started
        if len(records)%progress_every==0:
            elapsed=time.perf_counter()-started
            print('MTP batched replay',generated.tolist(),'decode tok/s',
                  int(generated.sum()-slots)/elapsed,flush=True)
    decode_seconds=time.perf_counter()-started
    model._write('tokens',pending)
    elapsed=prefill_seconds+decode_seconds
    concurrent=[r for r in records if r['active_requests']==slots]
    concurrent_seconds=concurrent[-1]['ended_seconds'] if concurrent else 0.
    report=dict(scope='native fixed cohort; exact output lengths; batched greedy MTP verification',
        slots=slots,output_tokens_per_request=output_tokens,total_output_tokens=int(generated.sum()),
        elapsed_seconds=elapsed,prefill_seconds=prefill_seconds,decode_seconds=decode_seconds,
        aggregate_output_tokens_per_second=int(generated.sum())/elapsed,
        decode_output_tokens_per_second=(int(generated.sum())-(slots if initial_output else 0))/decode_seconds,
        all_requests_active_decode_seconds=concurrent_seconds,
        all_requests_active_output_tokens=sum(r['output_tokens'] for r in concurrent),
        all_requests_active_decode_tokens_per_second=sum(r['output_tokens'] for r in concurrent)/concurrent_seconds if concurrent_seconds else None,
        rounds=len(records),mean_committed_inputs=float(np.mean([r['accepted_inputs'] for r in records])),
        mean_verify_seconds=float(np.mean([r['verify_seconds'] for r in records])),
        mean_draft_seconds=float(np.mean([r['draft_seconds'] for r in records])),
        mean_commit_seconds=float(np.mean([r['commit_seconds'] for r in records])),
        mean_repair_seconds=float(np.mean([r['repair_seconds'] for r in records])),
        end_positions=model.position.tolist(),kv_dtype=model.kv_dtype,weight_dtype='official FP8',
        draft='embedded MTP with output-history lookup' if lookups else 'embedded MTP',
        output_lookup=bool(lookups),lookup_request_rounds=sum(r['lookup_requests'] for r in records),
        proposals=chunk-1,speculative_verification=True,
        model_throughput_qualified=False,full_stress_target_reached=False,
        output_token_ids_sha256=hashlib.sha256(np.asarray(streams,dtype='int32').tobytes()).hexdigest())
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    (out/'rounds.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    np.save(out/'output-ids.npy',np.asarray(streams,dtype='int32'))
    print('Native MTP batched result',report,flush=True)
    return report
