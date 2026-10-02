"""Independent fixture replay must reject a different model or token sequence."""
import hashlib,json
import numpy as np
import pytest
from benchmarks.lfm2.webgpu_run import cached_reference


def fixture(tmp_path):
    model=tmp_path/'model.gguf';model.write_bytes(b'model identity')
    cases=[{'name':'chat','tokens':[1,2],'reset':True}]
    logits=np.array([1.,2.],np.float32);np.save(tmp_path/'0-numpy-logits.npy',logits)
    report={'status':'passed','model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
            'protocol':{'context':512},'validation':[{**cases[0],'numpy':{
                'relative_rms':.001,'cosine':.99999,'argmax':[1,1]}}]}
    (tmp_path/'report.json').write_text(json.dumps(report))
    return model,cases,logits,report


def test_cached_reference_records_independent_array_hashes(tmp_path):
    model,cases,logits,_=fixture(tmp_path)
    arrays,info=cached_reference(tmp_path,model,cases)
    np.testing.assert_array_equal(arrays[0],logits)
    assert info['logits_sha256']==[hashlib.sha256((tmp_path/'0-numpy-logits.npy').read_bytes()).hexdigest()]
    assert info['mode']=='cached independent NumPy logits'


@pytest.mark.parametrize('change',['model','tokens','context','status','accuracy','nonfinite'])
def test_cached_reference_rejects_mismatch_or_failed_evidence(tmp_path,change):
    model,cases,_,report=fixture(tmp_path)
    if change=='model':model.write_bytes(b'different model')
    elif change=='tokens':cases[0]['tokens']=[1,3]
    elif change=='context':report['protocol']['context']=1024
    elif change=='status':report['status']='failed'
    elif change=='accuracy':report['validation'][0]['numpy']['relative_rms']=.011
    else:np.save(tmp_path/'0-numpy-logits.npy',np.array([1.,np.nan],np.float32))
    (tmp_path/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):cached_reference(tmp_path,model,cases)
