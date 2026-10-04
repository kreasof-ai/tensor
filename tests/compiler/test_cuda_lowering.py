"""Hardware helper lowering leaves ordinary kernels untouched."""
from tensor.compiler.cuda_lowering import lower_cuda_intrinsics


def test_ordinary_cuda_has_no_helper_or_header_dependency():
    source='extern "C" __global__ void plain(float* out) { out[0] = 1.0f; }'
    assert lower_cuda_intrinsics(source)==source


def test_only_referenced_typed_helpers_are_emitted():
    load=lower_cuda_intrinsics('auto value = tensor_load_u16(ptr);')
    assert 'unsigned short' in load
    assert 'tensor_pack_f16x2' not in load
    pair=lower_cuda_intrinsics('auto value = tensor_pack_f16x2(lo, hi);')
    assert '__floats2half2_rn' in pair
    assert 'tensor_load_u16' not in pair
    both=lower_cuda_intrinsics('auto value = tensor_pack_f16x2(tensor_load_u16(ptr), hi);')
    assert both.count('__device__ __forceinline__')==2
