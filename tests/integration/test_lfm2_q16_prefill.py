"""Two-component integer projections: packed fields and independent dot oracle."""
import os
import numpy as np
import pytest
from tensor.compiler.webgpu_lowering import packed_integer_matmul_schedule


@pytest.mark.parametrize('change',[
    {'depth':33},{'micro_m':3},{'threads':64},{'owner_axis':'invalid'},
    {'group_order':'invalid'},{'rhs_words':[]},{'rhs_scales':[]},
    {'tile_k':33},{'depth':288,'tile_k':64},
    {'fixed_residual':1},
    {'signed_rhs':1},
    {'tile_m':256,'tile_n':64,'micro_m':16,'micro_n':16,'threads':64},
])
def test_integer_matmul_legality(change):
    with pytest.raises(ValueError):
        packed_integer_matmul_schedule(**{'rows':35,'depth':256,'columns':34,
            'rhs_words':['word'],'rhs_scales':['scale'],**change})


@pytest.mark.skipif(os.environ.get('TENSOR_WEBGPU')!='1',reason='requires native WebGPU')
@pytest.mark.parametrize('kind',['linear_q16','ffn_q16'])
@pytest.mark.parametrize('subgroup',[False,True])
@pytest.mark.parametrize('tile_k',[32,128])
@pytest.mark.parametrize('fixed_residual',[False,True])
@pytest.mark.parametrize('prepacked',[False,True])
def test_two_component_projection_fields_and_precision(tmp_path,kind,subgroup,tile_k,fixed_residual,prepacked):
    import tensor
    from tensor_llm.gguf import dequantize,prepack_q4_0
    from tensor_llm.webgpu_kernels import source
    r,k,o=35,256,34;rng=np.random.default_rng(1462);raw=[];weights=[]
    for _ in range(2 if kind=='ffn_q16' else 1):
        blocks=rng.integers(0,256,(o*k//32,18),dtype=np.uint8)
        factors=np.resize(np.array([0.,1.,-1.,.001,2**-24,-2**-24],np.float16),len(blocks))
        blocks[:,:2]=factors.view(np.uint8).reshape(-1,2)
        raw.append(prepack_q4_0(blocks) if prepacked else blocks.ravel().view(np.uint32));weights.append(dequantize(blocks.ravel(),2).reshape(o,k).astype(np.float64))
    paths=[]
    for name in ('quantize_q16',kind):
        path=tmp_path/(name+'.py');path.write_text(source(name,dict(r=r,k=k,o=o,sg=subgroup,fixed_residual=fixed_residual,q4_prepacked=prepacked,
            integer=dict(tile_k=tile_k,fixed_residual=fixed_residual))))
        artifact=path.with_suffix('.tbin');tensor.build(path,artifact,provider='webgpu');paths.append(artifact)
    with tensor.Device(provider='webgpu') as device:
        quantizer,projection=[device.load(p) for p in paths]
        packed=device.zeros(r*k//2,dtype='uint32');scales=device.zeros(r*k//16);sums=device.zeros(r*k//16,dtype='int32')
        uploaded=[device.from_numpy(v) for v in raw];output=device.full(r*o,np.nan)
        for magnitude in (.01,1.,1e-5):
            x=(rng.normal(size=(r,k))*magnitude).astype(np.float32);x[0]=0
            inp=device.from_numpy(x.ravel());quantizer.launch(inp,packed,scales,sums)
            q=packed.to_numpy().view(np.int8).reshape(2,r,k//32,32)
            factors=scales.to_numpy().reshape(2,r,k//32)
            np.testing.assert_array_equal(sums.to_numpy().reshape(2,r,k//32),q.astype(np.int32).sum(axis=-1))
            approximate=np.sum(q.astype(np.float64)*factors[:,:,:,None],axis=0).reshape(r,k)
            rounded=x.astype(np.float16).astype(np.float64)
            maximum=np.max(np.abs(rounded.reshape(r,k//32,32)),axis=-1)
            assert np.all(np.abs(approximate-rounded)<=np.repeat(maximum/127**2*.51+1e-12,32,axis=1))
            values=[approximate@w.T for w in weights]
            bounds=[np.abs(approximate)@np.abs(w).T*3e-6+1e-10 for w in weights]
            if len(weights)==1:expected,bound=values[0],bounds[0]
            else:
                g,u=values;gb,ub=bounds;silu=g/(1+np.exp(-g));expected=silu*u
                bound=np.abs(u)*1.1*gb+np.abs(silu)*ub+1.1*gb*ub+np.abs(expected)*2e-6+1e-10
            device.write(output,np.full(r*o,np.nan,np.float32))
            projection.launch(packed,scales,sums,*uploaded,output)
            actual=output.to_numpy().reshape(r,o)
            assert np.isfinite(actual).all() and np.all(np.abs(actual-expected)<=bound)
            inp.release()
