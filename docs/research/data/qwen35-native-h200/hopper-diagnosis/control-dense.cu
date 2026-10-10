#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/instruction/mma.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/cuda_fp8.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void kernel_kernel(const float* __restrict__ activation_scales, float* __restrict__ out, const bfloat16_t* __restrict__ scales, const uchar* __restrict__ weights, const uchar* __restrict__ x);
extern "C" __global__ void __launch_bounds__(256, 1) kernel_kernel(const float* __restrict__ activation_scales, float* __restrict__ out, const bfloat16_t* __restrict__ scales, const uchar* __restrict__ weights, const uchar* __restrict__ x) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  void* rhs = ((void*)((char*)buf_dyn_shmem + 0));
  void* lhs = ((void*)((char*)buf_dyn_shmem + 16384));
  void* activation_scales_1 = ((void*)((char*)buf_dyn_shmem + 24576));
  float total[16];
  uchar bits = (uchar)0;
  float block[16];
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    float broadcast_var = 0x0p+0f/*0.000000e+00*/;
    *(float4*)(total + (i * 4)) = make_float4(broadcast_var, broadcast_var, broadcast_var, broadcast_var);
  }
  #pragma unroll
  for (int i_1 = 0; i_1 < 2; ++i_1) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)rhs)[(((((i_1 * 4096) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(((fp8_e4_t*)weights)[((((((int)blockIdx.x) * 131072) + (i_1 * 65536)) + ((((int)threadIdx.x) >> 3) * 2048)) + ((((int)threadIdx.x) & 7) * 16))])));
  }
  tl::cp_async_commit();
  #pragma unroll
  for (int i_2 = 0; i_2 < 2; ++i_2) {
    tl::cp_async_gs<16>((&(((fp8_e4_t*)rhs)[((((((i_2 * 4096) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192)])), (&(((fp8_e4_t*)weights)[(((((((int)blockIdx.x) * 131072) + (i_2 * 65536)) + ((((int)threadIdx.x) >> 3) * 2048)) + ((((int)threadIdx.x) & 7) * 16)) + 128)])));
  }
  tl::cp_async_commit();
  for (int local_tile = 0; local_tile < 14; ++local_tile) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      ((float*)activation_scales_1)[((int)threadIdx.x)] = 0x1p+0f/*1.000000e+00*/;
      ((float*)activation_scales_1)[((int)threadIdx.x)] = activation_scales[(((((int)blockIdx.y) * 1024) + (((int)threadIdx.x) * 16)) + local_tile)];
    }
    __syncthreads();
    #pragma unroll
    for (int i_3 = 0; i_3 < 32; ++i_3) {
      bits = (uchar)0;
      bits = x[(((((((int)blockIdx.y) * 131072) + (i_3 * 4096)) + ((((int)threadIdx.x) >> 7) * 2048)) + (local_tile * 128)) + (((int)threadIdx.x) & 127))];
      uchar _reinterpret_tmp = bits;
      ((fp8_e4_t*)lhs)[((((((i_3 * 256) + ((((int)threadIdx.x) >> 7) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((i_3 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + (i_3 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((int)threadIdx.x) & 15))] = (*(fp8_e4_t *)(&(_reinterpret_tmp)));
    }
    tl::cp_async_wait<1>();
    __syncthreads();
    {
      fp8_e4_t A_local[32];
      fp8_e4_t B_local[16];
      #pragma unroll
      for (int i_4 = 0; i_4 < 4; ++i_4) {
        float broadcast_var_1 = 0x0p+0f/*0.000000e+00*/;
        *(float4*)(block + (i_4 * 4)) = make_float4(broadcast_var_1, broadcast_var_1, broadcast_var_1, broadcast_var_1);
      }
      for (int ki = 0; ki < 4; ++ki) {
        #pragma unroll
        for (int i_5 = 0; i_5 < 2; ++i_5) {
          tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)lhs)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_5 * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A_local[(i_5 * 16)])));
        }
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)rhs)[((((((((local_tile & 1) * 8192) + ((((int)threadIdx.x) >> 6) * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 1024)) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B_local[0])));
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(block + (i_6 * 8)), reinterpret_cast<const unsigned*>(A_local + (i_6 * 16)), reinterpret_cast<const unsigned*>(B_local + 0));
          tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(block + ((i_6 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local + (i_6 * 16)), reinterpret_cast<const unsigned*>(B_local + 8));
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for (int i_7 = 0; i_7 < 2; ++i_7) {
      tl::cp_async_gs<16>((&(((fp8_e4_t*)rhs)[(((((((local_tile & 1) * 8192) + (i_7 * 4096)) + ((((int)threadIdx.x) >> 3) * 128)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) & 1) * 64)) + (((((((int)threadIdx.x) & 31) >> 4) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(((fp8_e4_t*)weights)[((((((((int)blockIdx.x) * 131072) + (i_7 * 65536)) + ((((int)threadIdx.x) >> 3) * 2048)) + (local_tile * 128)) + ((((int)threadIdx.x) & 7) * 16)) + 256)])));
    }
    tl::cp_async_commit();
    __syncthreads();
    #pragma unroll
    for (int i_8 = 0; i_8 < 8; ++i_8) {
      float activation_scales_local_cast[2];
      *(float2*)(activation_scales_local_cast + 0) = make_float2(((float*)activation_scales_1)[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_8 >> 2) * 16)) + ((i_8 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)activation_scales_1)[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_8 >> 2) * 16)) + ((i_8 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
      float2 __1;
        float2 __2;
          float2 __3;
            float2 v_ = *(float2*)(block + (i_8 * 2));
            float2 v__1 = *(float2*)(activation_scales_local_cast + 0);
            __3.x = (v_.x*v__1.x);
            __3.y = (v_.y*v__1.y);
          float2 v__2 = make_float2(((float)scales[(((((int)blockIdx.x) >> 1) * 16) + local_tile)]), ((float)scales[(((((int)blockIdx.x) >> 1) * 16) + local_tile)]));
          __2.x = (__3.x*v__2.x);
          __2.y = (__3.y*v__2.y);
        float2 v__3 = *(float2*)(total + (i_8 * 2));
        __1.x = (__2.x+v__3.x);
        __1.y = (__2.y+v__3.y);
      *(float2*)(total + (i_8 * 2)) = __1;
    }
  }
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((float*)activation_scales_1)[((int)threadIdx.x)] = 0x1p+0f/*1.000000e+00*/;
    ((float*)activation_scales_1)[((int)threadIdx.x)] = activation_scales[(((((int)blockIdx.y) * 1024) + (((int)threadIdx.x) * 16)) + 14)];
  }
  __syncthreads();
  #pragma unroll
  for (int i_9 = 0; i_9 < 32; ++i_9) {
    bits = (uchar)0;
    bits = x[(((((((int)blockIdx.y) * 131072) + (i_9 * 4096)) + ((((int)threadIdx.x) >> 7) * 2048)) + (((int)threadIdx.x) & 127)) + 1792)];
    uchar _reinterpret_tmp_1 = bits;
    ((fp8_e4_t*)lhs)[((((((i_9 * 256) + ((((int)threadIdx.x) >> 7) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((i_9 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + (i_9 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((int)threadIdx.x) & 15))] = (*(fp8_e4_t *)(&(_reinterpret_tmp_1)));
  }
  tl::cp_async_wait<1>();
  __syncthreads();
  {
    fp8_e4_t A_local_1[32];
    fp8_e4_t B_local_1[16];
    #pragma unroll
    for (int i_10 = 0; i_10 < 4; ++i_10) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(block + (i_10 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)lhs)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_11 * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A_local_1[(i_11 * 16)])));
      }
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)rhs)[(((((((((int)threadIdx.x) >> 6) * 2048) + (((((int)threadIdx.x) & 31) >> 4) * 1024)) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_1 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_1 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(B_local_1[0])));
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(block + (i_12 * 8)), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 16)), reinterpret_cast<const unsigned*>(B_local_1 + 0));
        tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(block + ((i_12 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_1 + (i_12 * 16)), reinterpret_cast<const unsigned*>(B_local_1 + 8));
      }
    }
  }
  #pragma unroll
  for (int i_13 = 0; i_13 < 8; ++i_13) {
    float activation_scales_local_cast_1[2];
    *(float2*)(activation_scales_local_cast_1 + 0) = make_float2(((float*)activation_scales_1)[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_13 >> 2) * 16)) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)activation_scales_1)[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_13 >> 2) * 16)) + ((i_13 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
    float2 __4;
      float2 __5;
        float2 __6;
          float2 v__4 = *(float2*)(block + (i_13 * 2));
          float2 v__5 = *(float2*)(activation_scales_local_cast_1 + 0);
          __6.x = (v__4.x*v__5.x);
          __6.y = (v__4.y*v__5.y);
        float2 v__6 = make_float2(((float)scales[(((((int)blockIdx.x) >> 1) * 16) + 14)]), ((float)scales[(((((int)blockIdx.x) >> 1) * 16) + 14)]));
        __5.x = (__6.x*v__6.x);
        __5.y = (__6.y*v__6.y);
      float2 v__7 = *(float2*)(total + (i_13 * 2));
      __4.x = (__5.x+v__7.x);
      __4.y = (__5.y+v__7.y);
    *(float2*)(total + (i_13 * 2)) = __4;
  }
  __syncthreads();
  if (((int)threadIdx.x) < 64) {
    ((float*)activation_scales_1)[((int)threadIdx.x)] = 0x1p+0f/*1.000000e+00*/;
    ((float*)activation_scales_1)[((int)threadIdx.x)] = activation_scales[(((((int)blockIdx.y) * 1024) + (((int)threadIdx.x) * 16)) + 15)];
  }
  __syncthreads();
  #pragma unroll
  for (int i_14 = 0; i_14 < 32; ++i_14) {
    bits = (uchar)0;
    bits = x[(((((((int)blockIdx.y) * 131072) + (i_14 * 4096)) + ((((int)threadIdx.x) >> 7) * 2048)) + (((int)threadIdx.x) & 127)) + 1920)];
    uchar _reinterpret_tmp_2 = bits;
    ((fp8_e4_t*)lhs)[((((((i_14 * 256) + ((((int)threadIdx.x) >> 7) * 128)) + (((((((int)threadIdx.x) & 127) >> 6) + ((i_14 & 3) >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 63) >> 5) + (i_14 & 1)) & 1) * 32)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 31) >> 4)) & 1) * 16)) + (((int)threadIdx.x) & 15))] = (*(fp8_e4_t *)(&(_reinterpret_tmp_2)));
  }
  tl::cp_async_wait<0>();
  __syncthreads();
  {
    fp8_e4_t A_local_2[32];
    fp8_e4_t B_local_2[16];
    #pragma unroll
    for (int i_15 = 0; i_15 < 4; ++i_15) {
      float broadcast_var_3 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(block + (i_15 * 4)) = make_float4(broadcast_var_3, broadcast_var_3, broadcast_var_3, broadcast_var_3);
    }
    for (int ki_2 = 0; ki_2 < 4; ++ki_2) {
      #pragma unroll
      for (int i_16 = 0; i_16 < 2; ++i_16) {
        tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)lhs)[((((((((((int)threadIdx.x) & 63) >> 5) * 4096) + (i_16 * 2048)) + ((((int)threadIdx.x) & 15) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 16))])), (&(A_local_2[(i_16 * 16)])));
      }
      tl::ptx_ldmatrix_x4((&(((fp8_e4_t*)rhs)[((((((((((int)threadIdx.x) >> 6) * 2048) + (((((int)threadIdx.x) & 31) >> 4) * 1024)) + ((((int)threadIdx.x) & 7) * 128)) + (((((((int)threadIdx.x) & 7) >> 2) + (ki_2 >> 1)) & 1) * 64)) + (((((((int)threadIdx.x) & 3) >> 1) + (ki_2 & 1)) & 1) * 32)) + (((((((int)threadIdx.x) & 15) >> 3) + (((int)threadIdx.x) & 1)) & 1) * 16)) + 8192)])), (&(B_local_2[0])));
      for (int i_17 = 0; i_17 < 2; ++i_17) {
        tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(block + (i_17 * 8)), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 16)), reinterpret_cast<const unsigned*>(B_local_2 + 0));
        tl::mma_sync<tl::DataType::kFloat8_e4m3, tl::DataType::kFloat8_e4m3, tl::DataType::kFloat32, 16, 8, 32, false, true>(reinterpret_cast<float*>(block + ((i_17 * 8) + 4)), reinterpret_cast<const unsigned*>(A_local_2 + (i_17 * 16)), reinterpret_cast<const unsigned*>(B_local_2 + 8));
      }
    }
  }
  #pragma unroll
  for (int i_18 = 0; i_18 < 8; ++i_18) {
    float activation_scales_local_cast_2[2];
    *(float2*)(activation_scales_local_cast_2 + 0) = make_float2(((float*)activation_scales_1)[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_18 >> 2) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))], ((float*)activation_scales_1)[((((((((int)threadIdx.x) & 63) >> 5) * 32) + ((i_18 >> 2) * 16)) + ((i_18 & 1) * 8)) + ((((int)threadIdx.x) & 31) >> 2))]);
    float2 __7;
      float2 __8;
        float2 __9;
          float2 v__8 = *(float2*)(block + (i_18 * 2));
          float2 v__9 = *(float2*)(activation_scales_local_cast_2 + 0);
          __9.x = (v__8.x*v__9.x);
          __9.y = (v__8.y*v__9.y);
        float2 v__10 = make_float2(((float)scales[(((((int)blockIdx.x) >> 1) * 16) + 15)]), ((float)scales[(((((int)blockIdx.x) >> 1) * 16) + 15)]));
        __8.x = (__9.x*v__10.x);
        __8.y = (__9.y*v__10.y);
      float2 v__11 = *(float2*)(total + (i_18 * 2));
      __7.x = (__8.x+v__11.x);
      __7.y = (__8.y+v__11.y);
    *(float2*)(total + (i_18 * 2)) = __7;
  }
  #pragma unroll
  for (int i_19 = 0; i_19 < 8; ++i_19) {
    *(float2*)(out + (((((((((((int)blockIdx.y) * 524288) + (((((int)threadIdx.x) & 63) >> 5) * 262144)) + ((i_19 >> 2) * 131072)) + ((i_19 & 1) * 65536)) + (((((int)threadIdx.x) & 31) >> 2) * 8192)) + (((int)blockIdx.x) * 64)) + ((((int)threadIdx.x) >> 6) * 16)) + (((i_19 & 3) >> 1) * 8)) + ((((int)threadIdx.x) & 3) * 2))) = *(float2*)(total + (i_19 * 2));
  }
}
