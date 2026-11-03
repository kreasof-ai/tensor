"""Hopper shared-copy/GEMM lowering must fit Tensor's pointer/scalar ABI."""
import os
import pytest


@pytest.mark.skipif(os.environ.get('TENSOR_NVRTC') != '1', reason='pinned NVRTC producer check; GPU-free')
@pytest.mark.parametrize('rows', [16, 64])
@pytest.mark.parametrize('target', ['sm_90','sm_90a'])
@pytest.mark.parametrize('mode', ['auto', 'mma'])
def test_hopper_gemm_compiles_without_tensor_map_arguments(tmp_path, rows, target, mode):
    pytest.importorskip('tilelang')
    from tensor.compiler.build import build_artifact
    from tensor.artifacts.format import read_artifact
    source = tmp_path/'gemm.py'
    source.write_text(f'''
import tilelang.language as T

def tensor_export():
    @T.prim_func
    def kernel(a: T.Tensor(({rows}, 128), 'bfloat16'),
               b: T.Tensor((128, 128), 'bfloat16'),
               out: T.Tensor(({rows}, 128), 'float32')):
        with T.Kernel(1, threads=128):
            lhs = T.alloc_shared(({rows}, 128), 'bfloat16')
            rhs = T.alloc_shared((128, 128), 'bfloat16')
            result = T.alloc_fragment(({rows}, 128), 'float32')
            T.copy(a, lhs)
            T.copy(b, rhs)
            T.gemm(lhs, rhs, result, transpose_B=True, clear_accum=True)
            T.copy(result, out)
    return {{'kernel': kernel.with_attr('tensor.cuda.gemm_mode', '{mode}')}}
''')
    artifact = tmp_path/'gemm.tbin'
    build_artifact(source, artifact, target=target, compiler='nvrtc',
                   nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME', 'build/nvrtc-12.9'))
    manifest, _ = read_artifact(artifact)
    assert manifest['target'] == target
    assert {a['name'] for a in manifest['abi']} == {'a', 'b', 'out'}
    assert all(a['kind'] == 'buffer' for a in manifest['abi'])
    assert manifest['compiler']['pass_config']['tl.disable_tma_lower']
    assert manifest['compiler']['pass_config']['tl.disable_wgmma'] is (target=='sm_90' or mode=='mma')


def test_gemm_mode_rejects_unknown_instruction_choice():
    from types import SimpleNamespace
    from tensor.compiler.cuda.lowering import lowering_options
    with pytest.raises(ValueError, match="gemm_mode"):
        lowering_options("sm_90a", SimpleNamespace(attrs={"tensor.cuda.gemm_mode": "typo"}))


@pytest.mark.skipif(os.environ.get('TENSOR_NVRTC') != '1', reason='GPU-free Hopper compiler check')
def test_tma_descriptor_metadata_preserves_frontend_bindings(tmp_path):
    from tensor.compiler.build import build_artifact
    from tensor.artifacts.format import read_artifact
    source=tmp_path/'tma.py'
    source.write_text('''
import tilelang.language as T

def tensor_export():
    @T.prim_func
    def kernel(a:T.Tensor((128,256),'float16'),b:T.Tensor((128,256),'float16'),out:T.Tensor((128,128),'float32')):
        with T.Kernel(1,threads=128):
            lhs=T.alloc_shared((128,64),'float16')
            rhs=T.alloc_shared((128,64),'float16')
            result=T.alloc_fragment((128,128),'float32')
            T.clear(result)
            for tile in T.Pipelined(4,num_stages=2):
                T.copy(a[:,tile*64:tile*64+64],lhs)
                T.copy(b[:,tile*64:tile*64+64],rhs)
                T.gemm(lhs,rhs,result,transpose_B=True)
            T.copy(result,out)
    return {'kernel':kernel.with_attr('tensor.cuda.tma',1)}
''')
    artifact=tmp_path/'tma.tbin'
    build_artifact(source,artifact,target='sm_90a',compiler='nvrtc',
        nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME','build/nvrtc-12.9'))
    manifest,_=read_artifact(artifact)
    assert [a['name'] for a in manifest['abi']]==['a','b','out']
    assert set(manifest['cuda_tensor_maps'])=={'a','b'}
    assert manifest['cuda_tensor_maps']['a']['global_dims']==[256,128]
    assert manifest['launch']['block']==[256,1,1]
    assert 'cuda_tensor_maps' in manifest['runtime_abi']['required_capabilities']


@pytest.mark.parametrize('value',[1.5,'yes',2])
def test_tma_option_requires_boolean(value):
    from types import SimpleNamespace
    from tensor.compiler.cuda.lowering import lowering_options
    with pytest.raises(ValueError,match='boolean'):
        lowering_options('sm_90a',SimpleNamespace(attrs={'tensor.cuda.tma':value}))
