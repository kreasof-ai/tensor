"""FP16 experiment preserves packed storage and checks arithmetic contracts."""
import os
from pathlib import Path
import numpy as np
import pytest


def test_half2_tail_and_unknown_schedule_rejected():
    from benchmarks.lfm2.fp16_decode import linear_source
    with pytest.raises(ValueError,match='divisible by 64'):
        linear_source({'r':1,'k':96,'o':64,'type':1},'fp16_half2')
    with pytest.raises(ValueError,match='unknown'):
        linear_source({'r':1,'k':64,'o':64,'type':1},'not-a-schedule')


@pytest.mark.skipif(os.environ.get('TENSOR_LFM2_CUDA')!='1',reason='set TENSOR_LFM2_CUDA=1')
@pytest.mark.parametrize('mode',['fp16_cast','fp16_half2','fp16_mma32'])
def test_small_mixed_encoding_fp16_plan_against_independent_operators(tmp_path,mode):
    import tensor
    from tensor_llm import LFM2
    from benchmarks.lfm2.fp16_decode import bundle
    from benchmarks.lfm2.torch_reference import Reference
    root=Path(__file__).resolve().parents[2]
    model=root/'build/lfm2-diagnostic.gguf';base=root/'build/lfm2-diagnostic'
    bundle(base,tmp_path,mode)
    reference=Reference(model,decode_mode='half2' if mode=='fp16_half2' else 'fp16')
    with tensor.Device() as device,LFM2(model,tmp_path,device,context=384) as network:
        for tokens in ([256]+[i%256 for i in range(128)],[65],[66]):
            actual=network.forward(tokens)
            for start in range(0,len(tokens),128):expected=reference.forward(tokens[start:start+128])
            assert np.isfinite(actual).all()
            assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<.01
        network.reset();reference.reset()
        np.testing.assert_allclose(network.forward([256,65,66]),reference.forward([256,65,66]),rtol=.01,atol=.002)
