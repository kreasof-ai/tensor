"""Typed CUDA operations used by inspectable TileLang/TIRx algorithms.

Only scalar loads and FP16 pair conversion need helpers. Loops, quantization,
reductions, fusion and attention remain in frontend IR. TileLang already owns
warp shuffles, vectorized loads and IEEE FP32 FMA lowering.
"""


def lower_cuda_intrinsics(source):
    helpers = {
        'tensor_load_u16': '''__device__ __forceinline__ unsigned int tensor_load_u16(const void* ptr) {
    return *reinterpret_cast<const unsigned short*>(ptr);
}''',
        'tensor_pack_f16x2': '''__device__ __forceinline__ unsigned int tensor_pack_f16x2(float lo, float hi) {
    __half2_raw bits = __floats2half2_rn(lo, hi);
    return unsigned(bits.x) | (unsigned(bits.y) << 16);
}''',
    }
    used = [definition for name, definition in helpers.items() if name + '(' in source]
    if not used:
        return source
    return '#include <cuda_fp16.h>\n' + '\n'.join(used) + '\n' + source
