"""Opt-in NVRTC module selection, TIRx retargeting and source fallback."""
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

import tensor as tx
from tensor.modules import ModuleError, Project, install

pytestmark = pytest.mark.skipif(os.environ.get('TENSOR_P3_CUDA')!='1',reason='set TENSOR_P3_CUDA=1')
ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('example',['dynamic_affine','gemm_relu'])
def test_wrong_sm_binary_retargets_portable_ir_and_reuses_cached_image(tmp_path,example):
    lib=tmp_path/'lib';lib.mkdir()
    carrier=lib/'sm80.tbin'
    tx.build(ROOT/f'examples/{example}.py',carrier,target='sm_80',compiler='nvrtc',cache_dir=tmp_path/'compiler')
    (lib/'tensor.json').write_text(json.dumps({'formatVersion':1,'name':'ops','version':'0.1.0','tensorAbi':1,
        'exports':{'affine':{'artifacts':['sm80.tbin'],'portable':'sm80.tbin'}}}))
    cache=tmp_path/'modules';install(lib,cache_dir=cache)
    mod=Project(lib,cache_dir=cache).module()
    with tx.Device() as d:
        if d.info['arch']=='sm_80':pytest.skip('test needs a device different from sm_80')
        with pytest.raises(ModuleError,match='--compile'):
            mod.load('affine',d)
        kernel=mod.load('affine',d,compile=True,cache_dir=tmp_path/'compiler')
        if example=='dynamic_affine':
            for size in (1,127,128,129,1025):
                tx.assert_close(kernel(d.arange(size),d.ones((size,)),scale=2.5),2.5*np.arange(size,dtype='float32')+1)
        else:
            tx.assert_close(kernel(d.ones((64,64),'float16'),d.ones((64,64),'float16'),d.ones((64,),'float16')),
                            np.full((64,64),65,dtype='float16'))
        assert mod.resolve('affine',target=d.info['arch'])['selection']=='cached'
        kernel.release()
        with mod.load('affine',d) as cached:
            if example=='dynamic_affine':
                tx.assert_close(cached(d.arange(129),d.ones((129,)),scale=2.),2*np.arange(129,dtype='float32')+1)
            else:
                tx.assert_close(cached(d.ones((64,64),'float16'),d.ones((64,64),'float16'),d.ones((64,),'float16')),
                                np.full((64,64),65,dtype='float16'))


def test_source_only_module_uses_nvrtc_once_and_runs_by_cli_reference(tmp_path,capsys):
    from tensor.cli import main
    lib=tmp_path/'lib';lib.mkdir()
    shutil.copyfile(ROOT/'examples/elementwise.py',lib/'kernel.py')
    (lib/'tensor.json').write_text(json.dumps({'formatVersion':1,'name':'ops','version':'0.1.0','tensorAbi':1,
        'exports':{'elementwise':'kernel.py'}}))
    cache=tmp_path/'modules';install(lib,cache_dir=cache)
    mod=Project(lib,cache_dir=cache).module()
    with tx.Device() as d:
        result=mod.resolve('elementwise',target=d.info['arch'],compile=True,cache_dir=tmp_path/'compiler')
        assert result['selection']=='compiled' and result['build']['compiler']=='nvrtc'
    np.save(tmp_path/'a.npy',np.arange(129,dtype='float32'))
    np.save(tmp_path/'b.npy',np.ones(129,dtype='float32'))
    common=['--project',str(lib),'--module-cache',str(cache)]
    assert main(['run','ops::elementwise',*common,'--input',f'a={tmp_path}/a.npy','--input',f'b={tmp_path}/b.npy','--out-dir',str(tmp_path/'out')])==0
    np.testing.assert_array_equal(np.load(tmp_path/'out/c.npy'),2*np.arange(129,dtype='float32')+1)
    assert json.loads(capsys.readouterr().out)['module']['selection']=='cached'
