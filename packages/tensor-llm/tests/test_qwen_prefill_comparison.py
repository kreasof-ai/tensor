"""The wider-batch gate must compare each slot's last real token."""
import numpy as np


def test_split_control_preserves_early_finishing_and_inactive_slots():
    from benchmarks.qwen35.prefill_comparison import forward
    class Control:
        chunk=4
        def __init__(self):self.counts=[];self.positions=np.zeros(4,'int32')
        def forward(self,tokens,lengths,*,read_logits):
            assert read_logits
            self.counts.append(lengths.copy());self.positions+=lengths
            last=np.full(4,-1,'int32')
            for slot,count in enumerate(lengths):
                if count:last[slot]=tokens[slot,count-1]
            # Inactive calls clear temporary logits, just like the native head.
            logits=np.stack([last,np.maximum(last,0)],axis=1).astype('float32')
            return last,logits
    tokens=np.arange(32,dtype='int32').reshape(4,8)
    control=Control();result,logits=forward(control,tokens,np.array([8,5,2,0],'int32'))
    np.testing.assert_array_equal(result,[7,12,17,-1])
    np.testing.assert_array_equal(logits,[[7,7],[12,12],[17,17],[-1,0]])
    np.testing.assert_array_equal(control.positions,[8,5,2,0])
    np.testing.assert_array_equal(control.counts,[[4,4,2,0],[4,1,0,0]])


def test_cache_hashes_detect_valid_changes_and_ignore_unused_capacity():
    from types import SimpleNamespace
    from benchmarks.qwen35.prefill_comparison import cache_prefix_hashes
    cache=np.arange(128,dtype='uint8').reshape(2,2,8,4)
    buffer=SimpleNamespace(to_numpy=lambda:cache)
    model=SimpleNamespace(states={0:(buffer,)},position=np.array([3,0]),
                          config=SimpleNamespace(layers=['full_attention']))
    before=cache_prefix_hashes(model)
    cache[0,:,3:]=127;cache[1]=255
    assert cache_prefix_hashes(model)==before
    cache[0,1,2,1]+=1
    assert cache_prefix_hashes(model)!=before
