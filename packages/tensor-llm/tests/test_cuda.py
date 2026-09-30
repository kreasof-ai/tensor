"""Opt-in real checkpoint checks against retained independent-reference evidence."""
import json
import os
from pathlib import Path
import numpy as np
import pytest
import tensor
from tensor_llm import LFM2

pytestmark=pytest.mark.skipif(os.environ.get('TENSOR_LFM2_CUDA')!='1',reason='set TENSOR_LFM2_CUDA=1 and prepare the Q4_0 bundle')
ROOT=Path(__file__).resolve().parents[3]
MODEL=ROOT/'build/lfm2-models/LFM2.5-2.6B-Q4_0.gguf'
BUNDLE=ROOT/'build/lfm2-q4_0'
REFERENCE=ROOT/'build/lfm2-q4_0-validation-final'


@pytest.mark.parametrize('graphs',[False,True])
def test_real_model_graph_eager_reset_and_capacity(graphs):
    spec=json.loads((REFERENCE/'reference-spec.json').read_text())
    prompt=spec['validation'][0]['tokens'];token=spec['validation'][1]['tokens']
    with tensor.Device() as device,LFM2(MODEL,BUNDLE,device,context=128,graphs=graphs) as model:
        first=model.forward(prompt)
        np.testing.assert_allclose(first,np.load(REFERENCE/'tensor-0-logits.npy'),rtol=1e-6,atol=1e-6)
        second=model.forward(token)
        np.testing.assert_allclose(second,np.load(REFERENCE/'tensor-1-logits.npy'),rtol=1e-6,atol=1e-6)
        model.reset();np.testing.assert_array_equal(model.forward(prompt),first)
        for tokens,message in (([], 'nonempty'),([128000],'vocabulary'),([1]*129,'capacity')):
            with pytest.raises(ValueError,match=message):model.forward(tokens)
    with pytest.raises(RuntimeError,match='closed'):model.forward(prompt)


def test_manifest_implementation_and_artifact_checksum_rejected(tmp_path):
    manifest=json.loads((BUNDLE/'inference.json').read_text())
    manifest['implementation']['tensor_llm.model']='bad'
    (tmp_path/'inference.json').write_text(json.dumps(manifest))
    with tensor.Device() as device:
        with pytest.raises(ValueError,match='implementation mismatch'):LFM2(MODEL,tmp_path,device)
    manifest=json.loads((BUNDLE/'inference.json').read_text())
    for record in manifest['kernels'].values():
        target=tmp_path/record['artifact'];target.parent.mkdir(exist_ok=True)
        target.write_bytes(b'corrupt')
    (tmp_path/'inference.json').write_text(json.dumps(manifest))
    with tensor.Device() as device:
        with pytest.raises(ValueError,match='checksum mismatch'):LFM2(MODEL,tmp_path,device)


def test_small_mixed_encoding_model_against_independent_operators():
    from benchmarks.lfm2.torch_reference import Reference
    path=ROOT/'build/lfm2-diagnostic.gguf';bundle=ROOT/'build/lfm2-diagnostic'
    reference=Reference(path)
    with tensor.Device() as device,LFM2(path,bundle,device,context=384) as model:
        for ids in ([256]+[j%256 for j in range(128)],[65],[66],[67]):
            actual=model.forward(ids)
            for start in range(0,len(ids),128):expected=reference.forward(ids[start:start+128])
            assert np.linalg.norm(actual-expected)/np.linalg.norm(expected)<.01
        model.reset();reference.reset()
        np.testing.assert_allclose(model.forward([256,65,66]),reference.forward([256,65,66]),rtol=.01,atol=.002)
