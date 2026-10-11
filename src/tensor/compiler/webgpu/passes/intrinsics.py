"""Intrinsics implementation for WebGPU kernels."""

from __future__ import annotations


def lower_wgsl_intrinsics(source):
    """Small typed WGSL operations absent from TileLang's scalar codegen.

    Keep packed byte alignment out of TIR's signed/unsigned index arithmetic.
    The WGSL backend then owns exact unsigned shifts and half unpacking.
    """
    helpers = {
        "tensor_byte_align_u32": """fn tensor_byte_align_u32(lo:u32, hi:u32, byte:u32)->u32 {
  let shift=byte*8u;
  return (lo>>shift) | select(0u,hi<<((32u-shift)&31u),shift!=0u);
}""",
        "tensor_unpack_f16": """fn tensor_unpack_f16(bits:u32)->f32 {
  return unpack2x16float(bits).x;
}""",
        "tensor_dot2_f16_add": """fn tensor_dot2_f16_add(a:u32, b:u32, accumulator:f32)->f32 {
  return accumulator + dot(unpack2x16float(a), unpack2x16float(b));
}""",
        "tensor_fma2_f16_vec": """fn tensor_fma2_f16_vec(a:u32, b:u32, accumulator:vec2<f16>)->vec2<f16> {
  return fma(vec2<f16>(unpack2x16float(a)), vec2<f16>(unpack2x16float(b)), accumulator);
}""",
        "tensor_sum2_f16_vec": """fn tensor_sum2_f16_vec(a:vec2<f16>)->f32 {
  return f32(a.x) + f32(a.y);
}""",
    }
    for name, helper in helpers.items():
        if name + "(" in source:
            source += "\n" + helper + "\n"
    if "tensor_fma2_f16_vec(" in source and "enable f16;" not in source:
        source = "enable f16;\n" + source
    if "dot4I8Packed(" in source:
        source = "requires packed_4x8_integer_dot_product;\n" + source
    if "tensor_subgroup_size()" in source:
        source = source.replace(
            "threadIdx : vec3<u32>",
            "threadIdx : vec3<u32>,\n  @builtin(subgroup_size) tensorSubgroupSize : u32",
        )
        source = source.replace("tensor_subgroup_size()", "tensorSubgroupSize")
    return source
