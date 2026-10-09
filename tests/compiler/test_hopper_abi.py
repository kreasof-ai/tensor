"""Hopper shared-copy/GEMM lowering must fit Tensor's pointer/scalar ABI."""
import os
import pytest


@pytest.mark.skipif(os.environ.get('TENSOR_NVRTC') != '1', reason='pinned NVRTC producer check; GPU-free')
@pytest.mark.parametrize('rows', [16, 64])
def test_hopper_gemm_compiles_without_tensor_map_arguments(tmp_path, rows):
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
    return {{'kernel': kernel}}
''')
    artifact = tmp_path/'gemm.tbin'
    build_artifact(source, artifact, target='sm_90', compiler='nvrtc',
                   nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME', 'build/nvrtc-12.9'))
    manifest, _ = read_artifact(artifact)
    assert manifest['target'] == 'sm_90'
    assert {a['name'] for a in manifest['abi']} == {'a', 'b', 'out'}
    assert all(a['kind'] == 'buffer' for a in manifest['abi'])
    assert manifest['compiler']['pass_config']['tl.disable_tma_lower']
    assert manifest['compiler']['pass_config']['tl.disable_wgmma']
