"""Opt-in regression for causal short blocks and partial attention tiles."""
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TENSOR_ATTENTION_CUDA') != '1',
                                reason='set TENSOR_ATTENTION_CUDA=1')


@pytest.mark.parametrize('length,causal', [(128, True), (129, False), (129, True), (257, True)])
def test_zero_logits_attention_handles_causal_and_tail_tiles(tmp_path, length, causal):
    import torch
    import tensor as tx
    from tools.flash_attention_demo import specialize
    shape = (2, 2, length, 64)
    source = tmp_path / 'attention.py'
    source.write_text(specialize(shape, causal))
    artifact = tmp_path / 'attention.tbin'
    with tx.Device() as device:
        target = device.info['arch']
    tx.build(source, artifact, target=target, compiler='nvrtc', cache_dir=tmp_path / 'compiler')
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream), tx.Device(stream=stream.cuda_stream) as device:
        q, k = (torch.zeros(shape, dtype=torch.float16, device='cuda') for _ in range(2))
        # Distinct values across batches/heads expose incorrect address binding.
        v = ((torch.arange(q.numel(), device='cuda') % 17 - 8) / 8).half().reshape(shape)
        output = torch.full_like(q, float('nan'))
        borrowed = tuple(device.from_dlpack(t) for t in (q, k, v, output))
        device.load(artifact).launch(*borrowed)
        if causal:
            expected = (v.float().cumsum(2) / torch.arange(1, length + 1, device='cuda').reshape(1, 1, length, 1)).half()
        else:
            expected = v.float().mean(2, keepdim=True).expand(shape).half()
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-3)
