"""Independent operator checks for partitioned cached decoding."""
import os
from pathlib import Path
import numpy as np
import pytest


def test_invalid_decode_schedule_rejected():
    from benchmarks.lfm2.decode_optimization import bundle,warp_partial_source
    with pytest.raises(ValueError,match='positive split'):
        warp_partial_source({'h':4,'kh':2,'d':64,'cap':512},0)
    with pytest.raises(ValueError,match='unknown projection'):
        bundle('unused','unused',projection='bad')


@pytest.mark.skipif(os.environ.get('TENSOR_LFM2_CUDA')!='1',reason='set TENSOR_LFM2_CUDA=1')
@pytest.mark.parametrize('splits',[4,16,32])
def test_split_decode_against_independent_reference(tmp_path,splits):
    import tensor
    from tensor_llm import LFM2
    from benchmarks.lfm2.decode_optimization import bundle,OptimizedLFM2
    from benchmarks.lfm2.torch_reference import Reference
    root=Path(__file__).resolve().parents[2]
    model=root/'build/lfm2-diagnostic.gguf';base=root/'build/lfm2-diagnostic'
    bundle(base,tmp_path,splits=splits)
    reference=Reference(model)
    with tensor.Device() as device,OptimizedLFM2(model,tmp_path,device,context=384) as network:
        with LFM2(model,base,device,context=384) as baseline:
            for tokens in ([256]+[i%256 for i in range(128)],[65],[66]):
                actual=network.forward(tokens);original=baseline.forward(tokens)
                for start in range(0,len(tokens),128):expected=reference.forward(tokens[start:start+128])
                assert np.isfinite(actual).all()
                assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<.01
                np.testing.assert_allclose(actual,original,rtol=.002,atol=.0002)
            assert len(network.caches)==len(baseline.caches)==1
            assert network.allocated_bytes-baseline.allocated_bytes==4*splits*66*4
            assert len(network.plans[1])==len(baseline.plans[1])+1
        network.reset();reference.reset()
        tokens=[256,65,66];graph=network.forward(tokens);expected=reference.forward(tokens)
        np.testing.assert_allclose(graph,expected,rtol=.01,atol=.002)
        network.reset();network.graphs_enabled=False
        np.testing.assert_array_equal(graph,network.forward(tokens))
