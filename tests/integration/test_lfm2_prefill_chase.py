"""Experimental chunk profiles preserve decode and mixed-precision prefill."""
import os
import numpy as np
import pytest
from tensor_llm.model import valid_rows,webgpu_parameters


@pytest.mark.parametrize('chunk',[32,64,128])
def test_experimental_chunk_is_explicit_and_decode_parameters_match(chunk):
    for profile in ('prefill_unrolled','prefill_outer','prefill_chunked'):
        assert valid_rows('webgpu',(1,chunk),profile)
        for kind,p in [('linear',dict(r=1,k=1024,o=2560,type=1)),
                       ('ffn',dict(r=1,k=1024,o=2560,type=1)),
                       ('attention',dict(r=1,h=16,kh=8,d=64,cap=576))]:
            assert webgpu_parameters(kind,p,profile)==webgpu_parameters(kind,p,'decode_searched')
    if chunk!=32:assert not valid_rows('webgpu',(1,chunk),'decode_searched')
    p=webgpu_parameters('ffn',dict(r=chunk,k=1024,o=2560,type=1),'prefill_unrolled')
    assert p['r']==chunk and p['schedule']=='partitioned'
    assert p['explicit_unroll'] and p['unroll']==16


def test_adaptive_rows_require_an_experimental_profile_and_sorted_unique_chunks():
    assert valid_rows('webgpu',(1,32,128),'prefill_outer')
    assert valid_rows('webgpu',(1,32,64,128),'prefill_chunked')
    for rows in ((1,128,32),(1,32,32),(1,33),(True,32),(1,)):
        assert not valid_rows('webgpu',rows,'prefill_outer')
    assert not valid_rows('webgpu',(1,32,128),'decode_searched')
    assert not valid_rows('cuda',(1,32,128),'prefill_outer')


def test_selected_outer_configuration_is_not_shared_with_callers():
    p=dict(r=32,k=1024,o=1024,type=1)
    first=webgpu_parameters('linear',p,'prefill_outer')
    first['outer']['tile_m']=1
    assert webgpu_parameters('linear',p,'prefill_outer')['outer']['tile_m']==32
    ffn=webgpu_parameters('ffn',dict(r=128,k=1024,o=2560,type=1),'prefill_outer')
    assert ffn['schedule']=='partitioned' and not ffn.get('explicit_unroll',False)


@pytest.mark.parametrize('rows,count,expected',[
    ((1,32),33,[32,1]),((1,32),17,[32]),((1,128),129,[128,1]),
    ((1,32,128),129,[128,1]),((1,32,128),159,[128,32]),
    ((1,32,128),384,[128,128,128]),
])
def test_forward_chunk_controls_and_last_readback(rows,count,expected):
    from types import SimpleNamespace
    from tensor_llm.model import LFM2
    launches=[];writes=[]
    class Plan:
        def __init__(self,r):self.r=r
        def launch(self,*,readback):
            launches.append((self.r,readback))
            return np.array([4],np.int32) if readback is not None else None
    engine=SimpleNamespace(closed=False,position=0,context=512,rows=rows,provider='webgpu',
        config=SimpleNamespace(vocab=8),logits='logits',control='control',
        workspaces={r:{'tokens':'tokens'+str(r)} for r in rows},
        prepared={r:Plan(r) for r in rows},_write=lambda buffer,value:writes.append((buffer,value.copy())))
    result=LFM2.forward(engine,np.full(count,4,np.int32))
    assert [r for r,_ in launches]==expected
    assert all(readback is None for _,readback in launches[:-1])
    assert launches[-1][1]=='logits' and result[0]==4
    controls=[value for buffer,value in writes if buffer=='control']
    assert sum(int(v[1]) for v in controls)==count
    assert [int(v[0]) for v in controls]==[sum(int(w[1]) for w in controls[:i]) for i in range(len(controls))]
    assert engine.position==count


@pytest.mark.skipif(os.environ.get('TENSOR_WEBGPU')!='1',reason='requires native WebGPU')
@pytest.mark.parametrize('kind',['linear','ffn'])
@pytest.mark.parametrize('layout,fma',[('km',True),('mk',False)])
def test_mixed_activation_outer_product_fusion_and_tails(tmp_path,kind,layout,fma):
    import tensor
    from tensor_llm.webgpu_kernels import source
    from benchmarks.lfm2.prefill_chase_search import reference
    r,k,o=35,129,33
    p=dict(r=r,k=k,o=o,type=1,schedule='outer',outer=dict(
        tile_m=16,tile_n=32,tile_k=16,micro_m=2,micro_n=4,threads=64,
        lhs_layout=layout,fma=fma,unroll=8,explicit_unroll=True))
    path=tmp_path/'outer.py';artifact=path.with_suffix('.tbin');path.write_text(source(kind,p))
    tensor.build(path,artifact,provider='webgpu')
    rng=np.random.default_rng(903)
    weights=[rng.normal(size=(o,k)).astype(np.float16) for _ in range(2 if kind=='ffn' else 1)]
    with tensor.Device(provider='webgpu') as device:
        kernel=device.load(artifact);ws=[device.from_numpy(w.ravel()) for w in weights]
        output=device.full(r*o,np.nan)
        for scale in (1,.01,1e-5):
            x=(rng.normal(size=(r,k))*scale).astype(np.float32)
            expected,bound=reference(x,weights)
            inp=device.from_numpy(x.ravel());device.write(output,np.full(r*o,np.nan,np.float32))
            kernel.launch(inp,*ws,output);actual=output.to_numpy().reshape(r,o)
            assert np.isfinite(actual).all()
            assert np.all(np.abs(actual-expected)<=bound)
            inp.release()
