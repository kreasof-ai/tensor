"""Capture owns fixed resources and recovers when the submission callback fails."""
import os
from pathlib import Path
import numpy as np
import pytest
import tensor as tx
from tensor.providers.cuda_graph import CudaGraph
from tensor.providers.cuda import CudaError

pytestmark=pytest.mark.skipif(os.environ.get('TENSOR_P2_CUDA')!='1',reason='set TENSOR_P2_CUDA=1')


def test_graph_replay_resource_lifetime_and_capture_recovery(tmp_path):
    artifact=tmp_path/'elementwise.tbin'
    tx.build(Path(__file__).resolve().parents[2]/'examples/elementwise.py',artifact,compiler='nvrtc')
    with tx.Device() as device:
        kernel=device.load(artifact);a=device.arange(129);b=device.ones((129,));out=device.zeros((129,))
        def submit():kernel.launch(a,b,out)
        with CudaGraph(device,submit,resources=(kernel,a,b,out)) as graph:
            for _ in range(3):graph.launch()
            np.testing.assert_array_equal(out.to_numpy(),2*np.arange(129)+1)
            a.release()
            with pytest.raises(CudaError,match='released resource'):graph.launch()
        with pytest.raises(CudaError,match='released'):graph.launch()
        a=device.arange(129)
        def failed():
            kernel.launch(a,b,out)
            raise ValueError('submission failed')
        with pytest.raises(ValueError,match='submission failed'):CudaGraph(device,failed)
        with CudaGraph(device,submit,resources=(kernel,a,b,out)) as recovered:
            recovered.launch()
            np.testing.assert_array_equal(out.to_numpy(),2*np.arange(129)+1)
