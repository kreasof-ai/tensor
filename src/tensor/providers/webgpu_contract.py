"""Compiler-free reflection of Tensor's deliberately small WGSL envelope."""

import re

TARGET = "webgpu-portable-v1"
TYPES = {"f16": "float16", "f32": "float32", "i32": "int32", "u32": "uint32"}


def reflect(source, entrypoint, abi, launch):
    if not isinstance(source, str) or not source.strip() or "\x00" in source:
        raise ValueError("invalid WGSL source")
    functions = re.findall(r"@compute\s+@workgroup_size\((\d+),\s*(\d+),\s*(\d+)\)\s+fn\s+(\w+)\(", source)
    if len(functions) != 1 or functions[0][3] != entrypoint or list(map(int, functions[0][:3])) != launch["block"]:
        raise ValueError("WGSL entrypoint/workgroup size differs from manifest")
    bindings = re.findall(r"@group\(0\)\s+@binding\((\d+)\)\s+var<storage,\s*(read|read_write)>\s+(\w+)\s*:\s*array<(f16|f32|i32|u32)>\s*;", source)
    buffers = [arg for arg in abi if arg["kind"] == "buffer"]
    if len(bindings) != len(buffers):
        raise ValueError("WGSL storage bindings differ from the lowered ABI")
    layout = []
    for index, ((binding, access, _, dtype), arg) in enumerate(zip(bindings, buffers)):
        if int(binding) != index or TYPES[dtype] != arg["dtype"]:
            raise ValueError("WGSL binding order/dtype differs from the lowered ABI")
        layout.append({"binding": index, "name": arg["name"], "access": access})
    uniform = re.findall(r"@group\(0\)\s+@binding\((\d+)\)\s+var<uniform>\s+\w+\s*:\s*(\w+)\s*;", source)
    if len(uniform) != 1 or int(uniform[0][0]) != len(buffers):
        raise ValueError("WGSL needs one trailing POD uniform binding")
    structures = re.findall(r"struct\s+" + re.escape(uniform[0][1]) + r"\s*\{([^}]*)\}", source)
    if len(structures) != 1:
        raise ValueError("WGSL POD structure is missing")
    fields = re.findall(r"(\w+)\s*:\s*(f32|i32|u32)\s*,?", structures[0])
    scalars = [arg for arg in abi if arg["kind"] == "scalar"]
    if len(fields) != len(scalars)+1 or fields[-1][1] != "u32" or not fields[-1][0].startswith("packGridDimX"):
        raise ValueError("WGSL POD fields differ from the lowered scalar ABI")
    if any(TYPES[field[1]] != arg["dtype"] for field, arg in zip(fields, scalars)):
        raise ValueError("WGSL POD scalar dtype differs from ABI")
    if len(re.findall(r"@binding\(", source)) != len(buffers)+1:
        raise ValueError("unexpected WGSL bindings")
    allocations = re.findall(r"var<workgroup>\s+\w+\s*:\s*array<(f16|f32|i32|u32),\s*(\d+)>\s*;", source)
    if len(allocations) != len(re.findall(r"var<workgroup>", source)):
        raise ValueError("unsupported WGSL workgroup allocation")
    storage = sum(int(size)*(2 if dtype == "f16" else 4) for dtype, size in allocations)
    enables = re.findall(r"\benable\s+(\w+)\s*;", source)
    if set(enables) - {"f16"}:
        raise ValueError("portable WebGPU shaders cannot require subgroup/vendor extensions")
    features = ["shader-f16"] if "f16" in enables else []
    if re.search(r"\bsubgroup\w*\s*\(|@builtin\(subgroup_", source):
        features.append("subgroup")
    return {"bindings": layout, "required_features": features,
            "workgroup_storage_bytes": storage}


def validate_metadata(metadata, abi):
    if not isinstance(metadata, dict) or set(metadata) != {"bindings", "required_features", "workgroup_storage_bytes"}:
        raise ValueError("invalid WebGPU contract")
    features = metadata["required_features"]
    if features not in ([], ["shader-f16"], ["subgroup"], ["shader-f16", "subgroup"]):
        raise ValueError("invalid WebGPU features")
    if any(arg["dtype"] == "float16" for arg in abi) and "shader-f16" not in features:
        raise ValueError("FP16 WebGPU buffers require shader-f16")
    storage = metadata["workgroup_storage_bytes"]
    if type(storage) is not int or not 0 <= storage <= 32768:
        raise ValueError("portable WebGPU profile supports at most 32768 workgroup bytes")
    buffers = [arg for arg in abi if arg["kind"] == "buffer"]
    bindings = metadata["bindings"]
    if not isinstance(bindings, list) or len(bindings) != len(buffers) or len(buffers) > 8:
        raise ValueError("invalid WebGPU binding count")
    for index, (item, arg) in enumerate(zip(bindings, buffers)):
        if (not isinstance(item, dict) or set(item) != {"binding", "name", "access"}
                or type(item["binding"]) is not int or item["binding"] != index
                or item["name"] != arg["name"] or item["access"] not in ("read", "read_write")):
            raise ValueError("invalid WebGPU binding contract")
    if any(arg["dtype"] not in TYPES.values() or (arg["kind"] == "scalar" and arg["dtype"] == "float16") for arg in abi):
        raise ValueError("WebGPU profile accepts FP16/FP32/int32/uint32 buffers and 32-bit scalars")
