"""A faster invalid schedule must never win the kernel search."""
import numpy as np
import pytest
from tensor.tuning import tune


class Output:
    def __init__(self):self.value=None
    def to_numpy(self):return self.value


class Candidate:
    def __init__(self,result,latency,*,saved=None):self.result=result;self.device=latency;self.saved=saved
    def launch(self,*arguments):
        arguments[-1].value=np.asarray(self.result,dtype=np.float32)
        if self.saved is not None:arguments[-2].value=np.asarray(self.saved,dtype=np.float32)


def test_tuning_rejects_fast_wrong_results_and_wrong_saved_intermediates(monkeypatch):
    monkeypatch.setattr('tensor.tuning.measure_cuda',lambda device,callback:{'median_gpu_seconds':device})
    output,saved=Output(),Output()
    candidates={'wrong_output':Candidate([9,9],0.01,saved=[1,2]),
                'wrong_saved':Candidate([1,2],0.02,saved=[9,9]),
                'correct':Candidate([1,2],1.0,saved=[1,2])}
    result=tune(candidates,[saved],output,np.array([1,2]),checks=[(saved,np.array([1,2]))])
    assert result['selected']=='correct'
    assert result['candidates']['wrong_output']['status']=='rejected'
    assert result['candidates']['wrong_saved']['status']=='rejected'


def test_tuning_requires_finite_reference_and_correct_candidates(monkeypatch):
    monkeypatch.setattr('tensor.tuning.measure_cuda',lambda device,callback:{'median_gpu_seconds':device})
    with pytest.raises(ValueError,match='finite'):tune({'bad':Candidate([0],1)},[],Output(),[np.nan])
    with pytest.raises(ValueError,match='all tuning'):tune({'bad':Candidate([0],1)},[],Output(),[10])
