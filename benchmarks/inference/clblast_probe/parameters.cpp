// Benchmark-only access to CLBlast's actual device-selected tuning parameters.
#include <clblast.h>
#include <cstring>
#include <sstream>
#include <unordered_map>

extern "C" __declspec(dllexport) int tensor_clblast_parameters(
    cl_device_id device, const char* kernel, int precision, char* output, size_t capacity) {
  try {
    std::unordered_map<std::string, size_t> parameters;
    auto status = clblast::RetrieveParameters(device, kernel,
        static_cast<clblast::Precision>(precision), parameters);
    if (status != clblast::StatusCode::kSuccess) return static_cast<int>(status);
    std::ostringstream text;
    for (const auto& pair : parameters) text << pair.first << "=" << pair.second << "\n";
    auto value = text.str();
    if (value.size() + 1 > capacity) return -1;
    std::memcpy(output, value.c_str(), value.size() + 1);
    return 0;
  } catch (...) { return -2; }
}
