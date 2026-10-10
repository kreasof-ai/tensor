"""Packed, exact FP8-to-FP16 operand conversion for paired Hopper MMA."""

CUDA_SOURCE=r'''
__device__ __forceinline__ unsigned int tensor_widen_e4m3x2(unsigned short bits) {
    unsigned int packed;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(packed) : "h"(bits));
    return packed;
}
'''
