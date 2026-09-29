/* Tensor runtime call ABI 1.0. Compiler IR is never part of this contract. */
#ifndef TENSOR_ABI_H
#define TENSOR_ABI_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif

#define TENSOR_ABI_VERSION 1u
#define TENSOR_DEVICE_CPU 1u
#define TENSOR_DEVICE_CUDA 2u
#define TENSOR_ARG_BUFFER 1u
#define TENSOR_ARG_SCALAR 2u
#define TENSOR_OK 0
#define TENSOR_ERROR_ABI 1
#define TENSOR_ERROR_ARGUMENT 2
#define TENSOR_ERROR_PROVIDER 3

/* Numeric IDs are permanent. Scalar payloads use their native little-endian
 * representation in the first sizeof(dtype) bytes; remaining bytes are zero.
 * ABI 1 supports 64-bit little-endian hosts and contiguous positive extents.
 */
enum TensorDTypeV1 {
  TENSOR_BOOL = 1, TENSOR_INT8 = 2, TENSOR_UINT8 = 3,
  TENSOR_INT16 = 4, TENSOR_UINT16 = 5, TENSOR_INT32 = 6,
  TENSOR_UINT32 = 7, TENSOR_INT64 = 8, TENSOR_UINT64 = 9,
  TENSOR_FLOAT16 = 10, TENSOR_FLOAT32 = 11, TENSOR_FLOAT64 = 12
};

typedef struct TensorBufferV1 {
  uint64_t address;
  uint64_t byte_size;
  const int64_t *shape;
  const int64_t *strides; /* bytes, never elements */
  uint32_t rank;
  uint32_t dtype;
  uint32_t device_type;
  int32_t device_ordinal;
} TensorBufferV1;

typedef struct TensorArgumentV1 {
  uint32_t kind;
  uint32_t dtype;
  TensorBufferV1 buffer; /* zero for a scalar */
  uint64_t scalar;       /* zero for a buffer */
} TensorArgumentV1;

typedef struct TensorStreamV1 {
  uint32_t device_type;
  int32_t device_ordinal;
  uint64_t handle; /* provider-specific opaque token; never cross-provider */
} TensorStreamV1;

typedef struct TensorCallV1 {
  uint32_t abi_version;
  uint32_t struct_size;
  const TensorArgumentV1 *arguments;
  uint32_t argument_count;
  uint32_t flags; /* zero in ABI 1.0 */
  uint32_t grid[3];
  uint32_t block[3];
  uint64_t shared_memory_bytes;
  TensorStreamV1 stream;
} TensorCallV1;

typedef struct TensorErrorV1 {
  int32_t code;
  char message[508]; /* UTF-8, always NUL terminated */
} TensorErrorV1;

/* CPU images export this function as tensor_kernel_v1. A CUDA provider consumes
 * the identical call descriptor and binds arguments to the image's CUDA ABI.
 * Calls borrow all descriptors and buffers. Descriptor arrays live through
 * return; device storage must live until stream completion. No ownership
 * transfers through the ABI. Providers report errors, never C++ exceptions.
 */
typedef int32_t (*TensorKernelV1)(const TensorCallV1 *, TensorErrorV1 *);

#ifdef __cplusplus
}
static_assert(sizeof(TensorBufferV1) == 48, "Tensor requires a 64-bit ABI");
static_assert(sizeof(TensorArgumentV1) == 64, "Tensor argument layout");
static_assert(sizeof(TensorStreamV1) == 16, "Tensor stream layout");
static_assert(sizeof(TensorCallV1) == 72, "Tensor call layout");
static_assert(sizeof(TensorErrorV1) == 512, "Tensor error layout");
#endif
#endif
