"""Small verification graphs must preserve slots and the matching repair buffer."""
from types import SimpleNamespace
import numpy as np
import pytest
from tensor_llm.qwen35.speculative.graph_pool import VerifierPool,RepairPool


class Context:
    def __init__(self,model,chunk):
        self.model=model;self.chunk=chunk;self.closed=False;self.pending_verification=False
        self.buffers={'normal':object()};self.calls=[]

    def forward(self,tokens,lengths=None,*,read_logits=False):
        self.calls.append((tokens.copy(),np.asarray(lengths).copy()))
        self.pending_verification=True
        prediction=(tokens+1)%self.model.config.vocab
        logits=np.repeat(tokens[...,None],self.model.config.vocab,axis=2).astype('float32')
        return (prediction,logits.reshape(-1,self.model.config.vocab)) if read_logits else prediction

    def commit(self,counts):
        if not self.pending_verification:raise RuntimeError('not pending')
        self.pending_verification=False;self.counts=np.array(counts)

    def close(self):self.closed=True


def model():
    return SimpleNamespace(slots=2,device=None,config=SimpleNamespace(vocab=5),_check=lambda:None)


@pytest.mark.parametrize('lengths,width',[([3,0],8),([8,2],8),([9,2],128),([128,127],128)])
def test_pool_preserves_slot_order_and_repair_context(lengths,width):
    target=model();draft=model();contexts={c:Context(target,c) for c in (8,128)}
    pool=VerifierPool(contexts)
    class Repair:
        def __init__(self,chunk):self.chunk=chunk;self.model=draft;self.closed=False
        def forward(self,tokens,hidden,counts):
            assert hidden is contexts[self.chunk].buffers['normal']
            self.tokens=tokens.copy();self.counts=np.array(counts)
        def close(self):self.closed=True
    repairs={c:Repair(c) for c in contexts};repair=RepairPool(pool,repairs)
    tokens=np.arange(256,dtype='int32').reshape(2,128)%5
    prediction,logits=pool.forward(tokens,lengths,read_logits=True)
    assert pool.active is contexts[width] and pool.pending_verification
    np.testing.assert_array_equal(prediction[:,:width],(tokens[:,:width]+1)%5)
    np.testing.assert_array_equal(logits.reshape(2,128,5)[:,:width],np.repeat(tokens[:,:width,None],5,axis=2))
    if width<128:assert not prediction[:,width:].any()
    with pytest.raises(RuntimeError):pool.forward(tokens,lengths)
    pool.commit(lengths);assert not pool.pending_verification
    repair.forward(tokens,pool.buffers['normal'],lengths)
    np.testing.assert_array_equal(repairs[width].tokens,tokens[:,:width])
    np.testing.assert_array_equal(repairs[width].counts,lengths)
    # Switch back to the maximum graph without changing owners or slot order.
    pool.forward(tokens,[128,128]);assert pool.active is contexts[128]
    pool.commit([128,128]);repair.close();pool.close();pool.close()
    assert all(context.closed for context in contexts.values())


def test_pool_rejects_mismatched_owner():
    with pytest.raises(ValueError,match='same live model'):
        VerifierPool({8:Context(model(),8),128:Context(model(),128)})
