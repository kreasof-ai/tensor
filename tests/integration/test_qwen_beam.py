"""Keep incomplete, changed-load and incorrectly counted runs out of the beam."""
from copy import deepcopy
import pytest


def result():
    return dict(status='measured-experimental',device=dict(name='NVIDIA H200'),
        kernel_tests_exit_code=0,same_state_quality=[dict(passed=True),dict(passed=True)],
        model_throughput_qualified=False,verification_serial_quality=dict(passed=False),
        client_report=dict(workload_sha256='6e012fe014f8fc86d58d0065862c62e77e1374fccba612a7ab5d54c1679db44a',
            servers=[dict(points=[dict(concurrency=8,summary=dict(completed=8,failed=0,
                prompt_tokens=256000,output_tokens=128000,elapsed_seconds=64.,
                output_tokens_per_second=2000.))])]))


def test_experimental_timing_does_not_require_or_imply_model_qualification():
    from benchmarks.qwen35.modal_beam import assess
    value=result()
    assert assess(value)==64.
    assert not value['model_throughput_qualified']
    assert not value['verification_serial_quality']['passed']


@pytest.mark.parametrize('field,value',[
    ('completed',7),('failed',1),('prompt_tokens',128000),('output_tokens',127999),
    ('elapsed_seconds',float('nan')),('elapsed_seconds',0.),
    ('output_tokens_per_second',6000.)])
def test_changed_or_miscounted_client_result_is_rejected(field,value):
    from benchmarks.qwen35.modal_beam import assess
    report=result()
    report['client_report']['servers'][0]['points'][0]['summary'][field]=value
    assert assess(report) is None


def test_changed_workload_device_or_failed_kernel_gate_is_rejected():
    from benchmarks.qwen35.modal_beam import assess
    original=result()
    changed=deepcopy(original);changed['device']['name']='NVIDIA L40S'
    assert assess(changed) is None
    changed=deepcopy(original);changed['client_report']['workload_sha256']='other'
    assert assess(changed) is None
    changed=deepcopy(original);changed['same_state_quality'][1]['passed']=False
    assert assess(changed) is None
    changed=deepcopy(original);changed['kernel_tests_exit_code']=None
    assert assess(changed) is None


def test_qualification_proof_rejects_changed_source_and_tampered_log(tmp_path):
    import hashlib,json
    from benchmarks.qwen35.qualification_cache import read_proof
    fingerprint={'source':'pinned','driver':'H200'};log='4 passed\n'
    (tmp_path/'kernel-tests.log').write_text(log)
    (tmp_path/'proof.json').write_text(json.dumps(dict(fingerprint=fingerprint,
        exit_code=0,log_sha256=hashlib.sha256(log.encode()).hexdigest())))
    assert read_proof(tmp_path,fingerprint) is not None
    assert read_proof(tmp_path,{'source':'changed','driver':'H200'}) is None
    (tmp_path/'kernel-tests.log').write_text('unverified replacement')
    assert read_proof(tmp_path,fingerprint) is None
    (tmp_path/'proof.json').write_text('[]')
    assert read_proof(tmp_path,fingerprint) is None
