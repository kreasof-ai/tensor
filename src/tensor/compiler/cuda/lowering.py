"""Typed CUDA operations used by inspectable TileLang/TIRx algorithms.

Only scalar loads and packed floating point conversion need helpers. Loops, quantization,
reductions, fusion and attention remain in frontend IR. TileLang already owns
warp shuffles, vectorized loads and IEEE FP32 FMA lowering.
"""


def lowering_options(target):
    # The pointer/scalar ABI excludes auto-TMA tensor-map parameters. Only
    # sm_90a enables WGMMA; both Hopper paths retain ordinary launch arguments.
    return ({"tl.disable_tma_lower": True, "tl.disable_wgmma": target == "sm_90",
             "tl.disable_warp_specialized": True} if target in ("sm_90", "sm_90a") else {})


def lower_kernel(kernel, target, *, shape_domain=None, symbols=None):
    """Canonical CUDA lowering shared by artifact production and inspection."""
    import tilelang
    from tvm.target import Target
    from tensor.compiler.cuda.prelower import lower_cuda_operations
    from tensor.compiler.frontend.domain import constrain
    from tensor.compiler.frontend.signature import frontend_arguments

    if symbols is None:
        symbols = {}
        frontend_arguments(kernel, symbols, shape_domain=shape_domain)
    kernel, prelowering = lower_cuda_operations(kernel, target)
    kernel = constrain(kernel, shape_domain, symbols)
    resolved = Target({"kind": "cuda", "arch": target})
    config = lowering_options(target)
    with tilelang.transform.PassContext(opt_level=3, config=config), resolved:
        lowered = tilelang.lower(kernel, target=resolved, enable_device_compile=False)
    cuda = lower_cuda_intrinsics(str(lowered.kernel_source))
    return kernel, prelowering, lowered, cuda


def lower_cuda_intrinsics(source):
    helpers = {
        "tensor_load_u16": """__device__ __forceinline__ unsigned int tensor_load_u16(const void* ptr) {
    return *reinterpret_cast<const unsigned short*>(ptr);
}""",
        "tensor_pack_f16x2": """__device__ __forceinline__ unsigned int tensor_pack_f16x2(float lo, float hi) {
    __half2_raw bits = __floats2half2_rn(lo, hi);
    return unsigned(bits.x) | (unsigned(bits.y) << 16);
}""",
        "tensor_decode_e4m3": """__device__ __forceinline__ float tensor_decode_e4m3(unsigned int bits) {
    __nv_fp8_e4m3 value;
    value.__x = static_cast<__nv_fp8_storage_t>(bits);
    return static_cast<float>(value);
}""",
        "tensor_encode_e4m3": """__device__ __forceinline__ unsigned int tensor_encode_e4m3(float value) {
    return static_cast<unsigned int>(__nv_fp8_e4m3(value).__x);
}""",
    }
    used = [definition for name, definition in helpers.items() if name + "(" in source]
    if not used:
        return source
    includes = "#include <cuda_fp16.h>\n"
    if "tensor_decode_e4m3(" in source or "tensor_encode_e4m3(" in source:
        includes += "#include <cuda_fp8.h>\n"
    return includes + "\n".join(used) + "\n" + source
