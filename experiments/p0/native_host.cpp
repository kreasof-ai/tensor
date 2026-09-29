// Standalone C++ process: no Python embedding or PyTorch.
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/function.h>
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "run") {
      auto module = tvm::ffi::Module::LoadFromFile(argv[2]);
      auto run = module->GetFunction("run").value();
      int64_t size = 129;
      std::vector<float> a(size), b(size, 0.25f), c(size, std::numeric_limits<float>::quiet_NaN());
      for (int i = 0; i < size; ++i) a[i] = i / 8.0f - 12.0f;
      auto tensor = [&](float* data) {
        return DLTensor{data, DLDevice{kDLCPU, 0}, 1, DLDataType{kDLFloat, 32, 1}, &size, nullptr, 0};
      };
      auto aa = tensor(a.data()), bb = tensor(b.data()), cc = tensor(c.data());
      run(&aa, &bb, &cc);
      for (int i = 0; i < size; ++i) {
        if (!std::isfinite(c[i]) || c[i] != std::max(2*a[i]+b[i], 0.0f)) {
          throw std::runtime_error("native CPU numerics mismatch");
        }
      }
      aa.dtype.bits = 16;
      bool rejected = false;
      try { run(&aa, &bb, &cc); } catch (const std::exception&) { rejected = true; }
      if (!rejected) throw std::runtime_error("wrong dtype accepted");
      std::cout << "{\"status\":\"passed\",\"size\":129,\"max_abs_error\":0,\"wrong_dtype_rejected\":true}\n";
    } else if (argc == 5 && std::string(argv[1]) == "ir") {
      for (int i = 2; i <= 3; ++i) {
        if (!dlopen(argv[i], RTLD_NOW | RTLD_GLOBAL)) throw std::runtime_error(dlerror());
      }
      std::ifstream input(argv[4]);
      std::string json((std::istreambuf_iterator<char>(input)), {});
      if (json.empty()) throw std::runtime_error("empty IR input");
      auto load = tvm::ffi::Function::GetGlobalRequired("node.LoadJSON");
      auto save = tvm::ffi::Function::GetGlobalRequired("node.SaveJSON");
      auto ir = load(tvm::ffi::String(json));
      auto encoded = save(ir).cast<tvm::ffi::String>();
      if (std::string(encoded) != json) throw std::runtime_error("native IR roundtrip changed bytes");
      std::cout << "{\"status\":\"passed\",\"byte_identical\":true,\"ir_bytes\":" << json.size() << "}\n";
    } else {
      throw std::runtime_error("usage: native_host run module.so | ir compiler.so tilelang.so ir.json");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
