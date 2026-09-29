// Native ABI 1 host. No Python, TVM FFI, CUDA headers or compiler dependencies.
// Artifact verification/extraction is a producer-side step; this host receives
// the verified kernel image and its entrypoint. Linux x86-64 validation profile.
#include <tensor/abi.h>
#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

template<class Fn> Fn symbol(void* library, const char* name) {
  auto fn = reinterpret_cast<Fn>(dlsym(library, name));
  if (!fn) throw std::runtime_error(std::string("missing symbol ")+name);
  return fn;
}
void check(int status) { if(status) throw std::runtime_error("CUDA status "+std::to_string(status)); }

struct Executable {
  static uint64_t identity() { static uint64_t next=1;return next++; }
  TensorExecutableV1 descriptor;
  void* function;
  Executable(unsigned device, void* fn): descriptor{TENSOR_ABI_VERSION,sizeof(TensorExecutableV1),
      device,0,identity(),identity(),3,device==TENSOR_DEVICE_CUDA?TENSOR_EXEC_ASYNC:0,{0,1,device,0,0}},function(fn) {}
  void* lookup(const TensorExecutableV1& snapshot) {
    if(!function || snapshot.abi_version!=TENSOR_ABI_VERSION || snapshot.struct_size<sizeof(snapshot) ||
       snapshot.device_type!=descriptor.device_type || snapshot.device_ordinal!=descriptor.device_ordinal ||
       snapshot.session!=descriptor.session || snapshot.handle!=descriptor.handle ||
       snapshot.argument_count!=descriptor.argument_count || snapshot.flags!=descriptor.flags ||
       std::memcmp(&snapshot.workspace,&descriptor.workspace,sizeof(snapshot.workspace)))
      throw std::runtime_error("invalid or released executable descriptor");
    return function;
  }
  bool rejects(const TensorExecutableV1& snapshot) {
    try { lookup(snapshot);return false; } catch(const std::runtime_error&) { return true; }
  }
  void validate_contract() {
    auto changed=descriptor;changed.workspace.byte_size=16;
    if(!rejects(changed)) throw std::runtime_error("workspace rejection failed");
    changed=descriptor;changed.session++;
    if(!rejects(changed)) throw std::runtime_error("session rejection failed");
  }
  void release() {
    function=nullptr;
    if(!rejects(descriptor)) throw std::runtime_error("released executable accepted");
  }
};

int main(int argc, char** argv) {
  try {
    if(argc<3) throw std::runtime_error("usage: host cpu kernel.so | cuda kernel.cubin entrypoint");
    bool cuda=std::string(argv[1])=="cuda";
    if(!cuda && std::string(argv[1])!="cpu") throw std::runtime_error("unknown provider");
    alignas(64) float a[129],b[129],out[129];
    for(int i=0;i<129;i++) { a[i]=float(i)-64; b[i]=1; out[i]=NAN; }
    int64_t shape=129,stride=4;
    TensorArgumentV1 args[3]{};
    float* pointers[]={a,b,out};
    for(int i=0;i<3;i++) {
      args[i].kind=TENSOR_ARG_BUFFER; args[i].dtype=TENSOR_FLOAT32;
      args[i].buffer={reinterpret_cast<uint64_t>(pointers[i]),sizeof(a),&shape,&stride,1,TENSOR_FLOAT32,
                      cuda?TENSOR_DEVICE_CUDA:TENSOR_DEVICE_CPU,0};
    }
    TensorCallV1 call{TENSOR_ABI_VERSION,sizeof(TensorCallV1),args,3,0,{2,1,1},{128,1,1},0,
                      {cuda?TENSOR_DEVICE_CUDA:TENSOR_DEVICE_CPU,0,0}};
    bool rejected=false;
    if(!cuda) {
      void* library=dlopen(argv[2],RTLD_NOW|RTLD_LOCAL);
      if(!library) throw std::runtime_error(dlerror());
      Executable executable(TENSOR_DEVICE_CPU,symbol<void*>(library,"tensor_kernel_v1"));
      executable.validate_contract();
      auto run=reinterpret_cast<TensorKernelV1>(executable.lookup(executable.descriptor));
      TensorErrorV1 error{};
      int status=run(&call,&error);
      if(status) throw std::runtime_error(error.message);
      args[0].dtype=TENSOR_FLOAT64;
      rejected=run(&call,&error)==TENSOR_ERROR_ARGUMENT;
      args[0].dtype=TENSOR_FLOAT32;
      call.abi_version=2;
      rejected=rejected && run(&call,&error)==TENSOR_ERROR_ABI;
      if(!rejected) throw std::runtime_error("ABI/dtype rejection failed");
      executable.release();
      dlclose(library);
    } else {
      if(argc!=4) throw std::runtime_error("CUDA mode needs an entrypoint");
      void* driver=dlopen("libcuda.so.1",RTLD_NOW|RTLD_LOCAL);
      if(!driver) throw std::runtime_error(dlerror());
      check(symbol<int(*)(unsigned)>(driver,"cuInit")(0));
      int device=0; check(symbol<int(*)(int*,int)>(driver,"cuDeviceGet")(&device,0));
      void *context=nullptr,*previous=nullptr,*stream=nullptr,*module=nullptr,*function=nullptr;
      check(symbol<int(*)(void**)>(driver,"cuCtxGetCurrent")(&previous));
      check(symbol<int(*)(void**,int)>(driver,"cuDevicePrimaryCtxRetain")(&context,device));
      check(symbol<int(*)(void*)>(driver,"cuCtxSetCurrent")(context));
      check(symbol<int(*)(void**,unsigned)>(driver,"cuStreamCreate")(&stream,1));
      std::ifstream input(argv[2],std::ios::binary);
      if(!input) throw std::runtime_error("kernel image unavailable");
      std::vector<char> image((std::istreambuf_iterator<char>(input)),{});
      check(symbol<int(*)(void**,const void*)>(driver,"cuModuleLoadData")(&module,image.data()));
      check(symbol<int(*)(void**,void*,const char*)>(driver,"cuModuleGetFunction")(&function,module,argv[3]));
      Executable executable(TENSOR_DEVICE_CUDA,function);
      executable.validate_contract();
      for(auto& arg:args) check(symbol<int(*)(uint64_t*,size_t)>(driver,"cuMemAlloc_v2")(&arg.buffer.address,arg.buffer.byte_size));
      for(int i=0;i<2;i++) check(symbol<int(*)(uint64_t,const void*,size_t)>(driver,"cuMemcpyHtoD_v2")(
                                args[i].buffer.address,pointers[i],sizeof(a)));
      check(symbol<int(*)(void*)>(driver,"cuStreamSynchronize")(nullptr));
      call.stream.handle=reinterpret_cast<uint64_t>(stream);
      void* parameters[3]={&args[0].buffer.address,&args[1].buffer.address,&args[2].buffer.address};
      using Launch=int(*)(void*,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,void*,void**,void**);
      check(symbol<Launch>(driver,"cuLaunchKernel")(executable.lookup(executable.descriptor),call.grid[0],call.grid[1],call.grid[2],
        call.block[0],call.block[1],call.block[2],unsigned(call.shared_memory_bytes),
        reinterpret_cast<void*>(call.stream.handle),parameters,nullptr));
      check(symbol<int(*)(void*)>(driver,"cuStreamSynchronize")(stream));
      check(symbol<int(*)(void*,uint64_t,size_t)>(driver,"cuMemcpyDtoH_v2")(out,args[2].buffer.address,sizeof(out)));
      for(auto& arg:args) check(symbol<int(*)(uint64_t)>(driver,"cuMemFree_v2")(arg.buffer.address));
      executable.release();
      check(symbol<int(*)(void*)>(driver,"cuModuleUnload")(module));
      check(symbol<int(*)(void*)>(driver,"cuStreamDestroy_v2")(stream));
      check(symbol<int(*)(int)>(driver,"cuDevicePrimaryCtxRelease_v2")(device));
      check(symbol<int(*)(void*)>(driver,"cuCtxSetCurrent")(previous));
      void* restored=nullptr;check(symbol<int(*)(void**)>(driver,"cuCtxGetCurrent")(&restored));
      if(restored!=previous) throw std::runtime_error("previous context not restored");
      dlclose(driver);
    }
    for(int i=0;i<129;i++) if(out[i]!=std::max(2*a[i]+b[i],0.f)) throw std::runtime_error("native numerics mismatch");
    std::cout<<"{\"status\":\"passed\",\"provider\":\""<<(cuda?"cuda":"cpu")
             <<"\",\"abi\":1,\"minor\":1,\"workspace_bytes\":0,\"elements\":129,\"negative_checks\":"<<(cuda?3:5)<<"}\n";
    return 0;
  } catch(const std::exception& error) { std::cerr<<error.what()<<"\n";return 1; }
}
