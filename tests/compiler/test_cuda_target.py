import pytest
from tensor.runtime.cuda_target import TARGET, matches_device


@pytest.mark.parametrize('target,arch,expected', [
    ('sm_90a','sm_90',True), ('sm_90a','sm_89',False),
    ('sm_90a','sm_100',False), ('sm_90','sm_90',True),
    ('sm_89','sm_90',False), ('sm_90','sm_100',False),
])
def test_exact_architecture_compatibility(target,arch,expected):
    assert matches_device(target,arch) is expected


@pytest.mark.parametrize('target', ['sm_89a','sm_100a','sm_90a_suffix','sm_90\n'])
def test_only_hopper_accelerated_target_is_supported(target):
    assert not TARGET.fullmatch(target)
