"""Compiler ablations must preserve the benchmark and numerical contract."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from benchmarks.inference.webgpu_gemm_accumulation_comparison import compare


def evidence():
    root=Path(__file__).resolve().parents[2]/'docs/research/data'
    report=json.loads((root/'clblast-rx6700xt-stock.json').read_text())
    suite=json.loads((root/'clblast-rx6700xt-suite.json').read_text())
    return report,deepcopy(report),suite,deepcopy(suite)


def test_identical_complete_sweep_is_a_valid_control():
    result=compare(*evidence())
    assert result['status']=='passed' and len(result['cases'])==32
    assert all(row['same_output'] and not row['wgsl_changed'] for row in result['cases'])


@pytest.mark.parametrize('change',['input','output','backend','shape','samples','tolerance','fp32_gate','storage'])
def test_ablation_rejects_changed_conditions(change):
    before,after,old_suite,new_suite=evidence()
    if change=='input':after['cases'][0]['inputs_sha256'][0]='changed'
    elif change=='output':after['cases'][0]['tensor']['correctness']['output_sha256']='changed'
    elif change=='backend':after['webgpu']['adapter']['backend_type']='D3D12'
    elif change=='shape':after['cases'][0]['m']+=1
    elif change=='samples':after['cases'][0]['tensor']['allocating']['samples_ms'].pop()
    elif change=='tolerance':after['cases'][0]['tensor']['correctness']['rtol']=1
    elif change=='fp32_gate':after['cases'][0]['clblast']['correctness']['passed']=False
    else:new_suite['cases'][0]['workgroup_storage_bytes']+=4
    with pytest.raises(ValueError):compare(before,after,old_suite,new_suite)
