// P0-only wrapper around the measured TileLang CPU kernel.
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/function.h>
#include <stdexcept>

extern "C" int32_t elementwise_kernel(float*, float*, float*);

void Run(tvm::ffi::TensorView a, tvm::ffi::TensorView b, tvm::ffi::TensorView c) {
  for (auto tensor : {a, b, c}) {
    auto dtype = tensor.dtype();
    if (tensor.device().device_type != kDLCPU || tensor.ndim() != 1 ||
        tensor.shape()[0] != 129 || dtype.code != kDLFloat || dtype.bits != 32 ||
        dtype.lanes != 1 || !tensor.IsContiguous() || tensor.byte_offset() != 0) {
      throw std::invalid_argument("P0 run requires contiguous CPU float32[129] at offset zero");
    }
  }
  if (elementwise_kernel(static_cast<float*>(a.data_ptr()), static_cast<float*>(b.data_ptr()),
                         static_cast<float*>(c.data_ptr())) != 0) {
    throw std::runtime_error("CPU kernel failed");
  }
}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, Run);
