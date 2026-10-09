"""Independent arithmetic checks for MTP input normalization and joining."""
import os
import numpy as np
import pytest


@pytest.mark.skipif(os.environ.get('TENSOR_QWEN_CUDA') != '1', reason='requires native CUDA')
def test_mtp_join_and_cast_match_independent_bf16_reference(tmp_path):
    import tensor
    import torch
    from tensor.compiler.build import build_artifact
    from tensor.compiler.entry import export_source
    torch.set_num_threads(1)
    r, c, eps = 8, 2048, 1e-6
    rng = np.random.default_rng(1792)
    arrays = [torch.from_numpy(rng.normal(size=shape).astype('float32')).bfloat16().float()
              for shape in ((r, c), (r, c), (c,), (c,))]
    arrays[0][0] = 0; arrays[1][1] = 0
    expected = torch.cat([x * torch.rsqrt(x.square().mean(-1, keepdim=True)+eps) * (1+w)
                          for x, w in zip(arrays[:2], arrays[2:])], dim=-1).bfloat16().float().numpy()
    artifacts = []
    for kind, p in (('mtp_join', dict(r=r, c=c, eps=eps)), ('mtp_cast', dict(r=r, c=c))):
        entry = tmp_path/(kind+'.py')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.mtp', 'make_kernel', kind, p,
                                      dependencies=('tensor.compiler.entry',)))
        artifact = entry.with_suffix('.tbin')
        build_artifact(entry, artifact, target=os.environ.get('TENSOR_QWEN_TARGET','sm_89'), compiler='nvrtc', nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
        artifacts.append(artifact)
    with tensor.Device() as d:
        buffers = [d.from_numpy(a.numpy(), dtype='bfloat16') for a in arrays]
        out = d.empty((r, 2*c), 'bfloat16')
        d.load(artifacts[0]).launch(*buffers, out)
        actual = out.to_numpy()
        assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < .001
        np.testing.assert_allclose(actual, expected, rtol=.008, atol=.002)
        values = rng.normal(size=(r, c)).astype('float32')
        raw = d.from_numpy(values); cast = d.empty((r, c), 'bfloat16')
        d.load(artifacts[1]).launch(raw, cast)
        np.testing.assert_array_equal(cast.to_numpy(), torch.from_numpy(values).bfloat16().float().numpy())
        copied = d.empty(cast.shape, 'bfloat16')
        d.driver.call('cuMemcpyDtoD_v2', copied.pointer, cast.pointer, cast.nbytes)
        d.driver.call('cuStreamSynchronize', None)
        np.testing.assert_array_equal(copied.to_numpy(), cast.to_numpy())
