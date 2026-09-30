// CPU C++ only: no CUDA toolkit headers or libraries. Torch owns storage and
// current streams; the installed NVIDIA driver supplies cuLaunchKernel.
#include <ATen/ops/empty_strided.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/impl/VirtualGuardImpl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <pybind11/stl.h>
#include <array>
#include <cstring>
#include <mutex>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#define CUDA_CALL __stdcall
#else
#include <dlfcn.h>
#define CUDA_CALL
#endif

using Launch = int (CUDA_CALL *)(void*, unsigned, unsigned, unsigned,
    unsigned, unsigned, unsigned, unsigned, void*, void**, void**);

static Launch driver_launch() {
    static Launch launch = [] {
#ifdef _WIN32
        auto driver = GetModuleHandleA("nvcuda.dll");
        auto entry = driver ? reinterpret_cast<Launch>(GetProcAddress(driver, "cuLaunchKernel")) : nullptr;
#else
        auto driver = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        auto entry = driver ? reinterpret_cast<Launch>(dlsym(driver, "cuLaunchKernel")) : nullptr;
#endif
        if (!entry) throw std::runtime_error("CUDA driver launch entry point unavailable");
        return entry;
    }();
    return launch;
}

struct Spec {
    std::vector<int64_t> sizes, strides;
    at::ScalarType dtype;
    c10::Device device;
    void* pointer;

    explicit Spec(const at::Tensor& t) : sizes(t.sizes().vec()), strides(t.strides().vec()),
        dtype(t.scalar_type()), device(t.device()), pointer(t.data_ptr()) {}

    bool matches(const at::Tensor& t, bool fixed) const {
        return t.device() == device && t.scalar_type() == dtype &&
            t.sizes().equals(sizes) && t.strides().equals(strides) &&
            (!fixed || t.data_ptr() == pointer);
    }
};

static std::vector<at::Tensor> unpack(py::handle sequence) {
    std::vector<at::Tensor> result;
    for (auto value : py::reinterpret_borrow<py::sequence>(sequence)) {
        if (!THPVariable_CheckExact(value.ptr()))
            throw py::type_error("native executor requires ordinary PyTorch tensors");
        result.push_back(THPVariable_Unpack(value.ptr()));
    }
    return result;
}

struct Slot {
    alignas(16) std::array<unsigned char, 16> value{};
    int tensor = -1;
    uint64_t alignment = 1;
};

class Plan {
    void* function_;
    std::array<unsigned, 7> launch_;
    Launch submit_;
    bool fixed_;
    c10::Device device_;
    std::vector<Spec> inputs_, outputs_;
    std::vector<at::Tensor> retained_;
    std::vector<Slot> slots_;
    std::vector<void*> parameters_;
    std::mutex mutex_;

public:
    Plan(uint64_t function, std::array<unsigned, 7> launch, py::tuple inputs,
         py::tuple outputs, py::list bindings, bool fixed) :
        function_(reinterpret_cast<void*>(function)), launch_(launch), submit_(driver_launch()),
        fixed_(fixed), device_(c10::DeviceType::CUDA, 0) {
        auto in = unpack(inputs), out = unpack(outputs);
        if (in.empty() || out.empty()) throw py::value_error("native plan needs inputs and outputs");
        device_ = in.front().device();
        if (!device_.is_cuda()) throw py::value_error("native plan requires CUDA tensors");
        for (const auto& t : in) {
            if (t.device() != device_) throw py::value_error("native inputs must share a device");
            inputs_.emplace_back(t);
        }
        for (const auto& t : out) {
            if (t.device() != device_) throw py::value_error("native outputs must share a device");
            outputs_.emplace_back(t);
        }
        if (fixed_) {
            retained_ = std::move(in);
            retained_.insert(retained_.end(), out.begin(), out.end());
        }
        slots_.reserve(bindings.size());
        for (auto binding : bindings) {
            Slot slot;
            if (py::isinstance<py::bytes>(binding)) {
                auto bytes = py::cast<std::string>(binding);
                if (bytes.empty() || bytes.size() > slot.value.size()) throw py::value_error("invalid scalar size");
                std::memcpy(slot.value.data(), bytes.data(), bytes.size());
            } else {
                auto pair = py::cast<std::pair<int, uint64_t>>(binding);
                slot.tensor = pair.first;
                slot.alignment = pair.second;
                if (slot.tensor < 0 || static_cast<size_t>(slot.tensor) >= inputs_.size() + outputs_.size() ||
                    !slot.alignment) throw py::value_error("invalid native tensor binding");
            }
            slots_.push_back(slot);
        }
        for (auto& slot : slots_) parameters_.push_back(slot.value.data());
    }

    py::object call(py::args args) {
        if (fixed_ && !args.empty()) throw py::type_error("prepared native calls take no arguments");
        if (!fixed_ && args.size() != inputs_.size()) throw py::value_error("wrong native input count");
        auto tensors = fixed_ ? retained_ : unpack(args);
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if (!inputs_[i].matches(tensors[i], fixed_))
                throw py::value_error(fixed_ ? "prepared tensor metadata or storage changed" : "native input metadata changed");
        }
        if (fixed_) {
            for (size_t i = 0; i < outputs_.size(); ++i)
                if (!outputs_[i].matches(tensors[inputs_.size()+i], true))
                    throw py::value_error("prepared tensor metadata or storage changed");
        }
        // Reject alignment before allocating, recording streams, or submitting.
        for (const auto& slot : slots_)
            if (slot.tensor >= 0 && static_cast<size_t>(slot.tensor) < inputs_.size() &&
                reinterpret_cast<uintptr_t>(tensors[slot.tensor].data_ptr()) % slot.alignment)
                return py::none();
        {
            py::gil_scoped_release release;
            // Unlock before reacquiring the GIL so concurrent callers cannot deadlock.
            std::lock_guard<std::mutex> lock(mutex_);
            c10::DeviceGuard guard(device_);
            c10::impl::VirtualGuardImpl impl(device_.type());
            auto stream = impl.getStream(device_);
            if (!fixed_)
                for (const auto& spec : outputs_)
                    tensors.push_back(at::empty_strided(spec.sizes, spec.strides,
                        at::TensorOptions().dtype(spec.dtype).device(spec.device)));
            for (auto& slot : slots_) {
                if (slot.tensor < 0) continue;
                const auto& tensor = tensors[slot.tensor];
                void* pointer = tensor.data_ptr();
                if (reinterpret_cast<uintptr_t>(pointer) % slot.alignment)
                    throw std::runtime_error("native output has insufficient pointer alignment");
                impl.recordDataPtrOnStream(tensor.storage().data_ptr(), stream);
                std::memcpy(slot.value.data(), &pointer, sizeof(pointer));
            }
            auto code = submit_(function_, launch_[0], launch_[1], launch_[2], launch_[3],
                launch_[4], launch_[5], launch_[6], impl.getStreamNativeHandle(stream), parameters_.data(), nullptr);
            if (code) throw std::runtime_error("cuLaunchKernel failed with CUDA error " + std::to_string(code));
        }
        if (fixed_) return py::none();
        if (outputs_.size() == 1)
            return py::reinterpret_steal<py::object>(THPVariable_Wrap(tensors[inputs_.size()]));
        py::tuple result(outputs_.size());
        for (size_t i = 0; i < outputs_.size(); ++i)
            result[i] = py::reinterpret_steal<py::object>(THPVariable_Wrap(tensors[inputs_.size()+i]));
        return result;
    }
};

PYBIND11_MODULE(TENSOR_EXECUTOR_MODULE, m) {
    py::class_<Plan>(m, "Plan")
        .def(py::init<uint64_t, std::array<unsigned, 7>, py::tuple, py::tuple, py::list, bool>())
        .def("__call__", &Plan::call);
}
