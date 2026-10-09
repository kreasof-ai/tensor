"""Native speculative scheduling over the shared target decoder and its MTP head."""
import time
import numpy as np
from ...speculative.acceptance import accepted_prefix
from ...speculative.lookup import OutputLookup


class Qwen35Speculative:
    """Borrow resident executors and commit independently for every request slot.

    The caller initializes both prefixes, then seeds the already emitted first
    token. Returned tokens have all passed target greedy verification. Optional
    output lookup is a draft heuristic; it never bypasses target computation.
    """
    def __init__(self,model,mtp,verifier,repair,*,output_lookup=False,fallback_proposals=None):
        model._check();mtp._check()
        if (verifier.model is not model or repair.model is not mtp or mtp.target is not model
                or verifier.chunk!=repair.chunk or verifier.chunk<2):
            raise ValueError('speculative executors must share target, slots and verification length')
        self.model,self.mtp,self.verifier,self.repair=model,mtp,verifier,repair
        self.chunk=verifier.chunk;self.slots=model.slots
        self.fallback_proposals=min(3,self.chunk-1) if fallback_proposals is None else fallback_proposals
        if not 1<=self.fallback_proposals<self.chunk:raise ValueError('invalid MTP fallback depth')
        self.lookups=[OutputLookup() for _ in range(self.slots)] if output_lookup else None

    def seed(self,pending,slots=None):
        self.model._check()
        pending=np.asarray(pending)
        if pending.shape!=(self.slots,) or pending.dtype.kind not in 'iu':raise ValueError('one pending token per slot is required')
        if np.any(pending<-1) or np.any(pending>=self.model.config.vocab):raise ValueError('invalid pending token')
        indices=list(range(self.slots)) if slots is None else list(slots)
        if len(set(indices))!=len(indices) or any(type(s) is not int or not 0<=s<self.slots for s in indices):
            raise ValueError('invalid request slots')
        if self.lookups:
            for slot in indices:
                self.lookups[slot]=OutputLookup()
                if pending[slot]>=0:self.lookups[slot].append(pending[slot])

    def step(self,pending,lengths=None):
        model,mtp=self.model,self.mtp;model._check();mtp._check()
        if np.any(model.position!=mtp.position):raise ValueError('target and MTP prefixes must be synchronized')
        pending=np.asarray(pending)
        if pending.shape!=(self.slots,) or pending.dtype.kind not in 'iu':raise ValueError('one pending token per slot is required')
        if lengths is None:lengths=np.full(self.slots,self.chunk,'int32')
        lengths=np.asarray(lengths)
        if (lengths.shape!=(self.slots,) or lengths.dtype.kind not in 'iu'
                or np.any(lengths<0) or np.any(lengths>self.chunk)):
            raise ValueError('invalid verification lengths')
        lengths=lengths.astype('int32',copy=True);base=model.position.copy()
        inputs=np.zeros((self.slots,self.chunk),'int32');inputs[:,0]=pending
        inputs[:,1]=mtp.buffers['tokens'].to_numpy()
        lookup_mask=np.zeros(self.slots,bool)
        if self.lookups:
            for slot,length in enumerate(lengths):
                if length<=1:continue
                proposals=self.lookups[slot].propose(int(length)-1)
                if proposals is not None:
                    lookup_mask[slot]=True;inputs[slot,1:length]=proposals
                else:lengths[slot]=min(length,self.fallback_proposals+1)
        draft_start=time.perf_counter()
        for i in range(2,self.chunk):
            use=(lengths>i)&~lookup_mask
            if np.any(use):
                prediction=mtp.draft(np.where(use,inputs[:,i-1],-1),mtp.buffers['normal'])
                inputs[use,i]=prediction[use]
        draft_seconds=time.perf_counter()-draft_start
        verify_start=time.perf_counter();predictions=self.verifier.forward(inputs,lengths)
        verify_seconds=time.perf_counter()-verify_start
        counts,outputs=accepted_prefix(inputs,predictions,lengths)
        commit_start=time.perf_counter();self.verifier.commit(counts)
        commit_seconds=time.perf_counter()-commit_start
        repair_start=time.perf_counter();next_pending=pending.astype('int32',copy=True)
        shifted=np.zeros_like(inputs)
        for slot,count in enumerate(counts):
            if count:
                shifted[slot,:count-1]=inputs[slot,1:count]
                next_pending[slot]=predictions[slot,count-1]
                shifted[slot,count-1]=next_pending[slot]
        mtp.position=base.copy();mtp._write('positions',mtp.position)
        self.repair.forward(shifted,self.verifier.buffers['normal'],counts)
        repair_seconds=time.perf_counter()-repair_start
        if self.lookups:
            for slot,values in enumerate(outputs):
                for token in values:self.lookups[slot].append(token)
        return dict(pending=next_pending,counts=counts,outputs=outputs,lengths=lengths,
                    lookup_requests=int(lookup_mask.sum()),draft_seconds=draft_seconds,
                    verify_seconds=verify_seconds,commit_seconds=commit_seconds,repair_seconds=repair_seconds)
