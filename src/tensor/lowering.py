"""Translate lowered TileLang metadata into a compiler-independent CUDA ABI."""

from __future__ import annotations

import re

from tensor.signature import INTEGER_TYPES, SCALAR_TYPES


def integer_expression(value, symbols: dict):
    import tvm

    if type(value) is int:
        return value
    ir = tvm.tirx
    if isinstance(value, ir.IntImm):
        return int(value.value)
    if isinstance(value, ir.Var):
        name, dtype = str(value.name), str(value.dtype)
        if dtype not in INTEGER_TYPES:
            raise ValueError(f"dimension {name} needs an integer dtype, received {dtype}")
        if name in symbols and symbols[name] != dtype:
            raise ValueError(f"conflicting dtype for dimension {name}")
        symbols[name] = dtype
        return {"var": name}
    if isinstance(value, ir.Cast) and str(value.dtype) in INTEGER_TYPES:
        return {"cast": str(value.dtype), "value": integer_expression(value.value, symbols)}
    operators = {"Add": "add", "Sub": "sub", "Mul": "mul", "FloorDiv": "floordiv",
                 "FloorMod": "floormod", "Min": "min", "Max": "max"}
    operation = operators.get(type(value).__name__)
    if operation:
        if str(value.dtype) not in INTEGER_TYPES:
            raise ValueError("launch and shape expressions must be integer-valued")
        return {"cast": str(value.dtype),
                "value": {"op": operation, "args": [integer_expression(value.a, symbols),
                                                      integer_expression(value.b, symbols)]}}
    raise ValueError(f"unsupported integer expression: {type(value).__name__}: {value}")


def frontend_arguments(kernel, symbols: dict) -> list[dict]:
    import tvm

    arguments = []
    for parameter in kernel.params:
        if parameter in kernel.buffer_map:
            buffer = kernel.buffer_map[parameter]
            contiguous = True
            if len(buffer.strides):
                expected = 1
                analyzer = tvm.arith.Analyzer()
                for stride, extent in zip(reversed(buffer.strides), reversed(buffer.shape)):
                    contiguous = contiguous and analyzer.can_prove_equal(stride, expected)
                    expected = expected * extent
            if (not contiguous or not isinstance(buffer.elem_offset, tvm.tirx.IntImm)
                    or int(buffer.elem_offset.value) != 0):
                raise ValueError("buffer exports need contiguous storage with zero frontend offset")
            shape = [integer_expression(value, symbols) for value in buffer.shape]
            if not shape or any(type(value) is int and value < 1 for value in shape):
                raise ValueError("buffer shapes must have positive extents")
            arguments.append({"kind": "buffer", "name": str(buffer.name),
                              "dtype": str(buffer.dtype), "shape": shape,
                              "alignment": max(1, int(buffer.data_alignment))})
        else:
            dtype = str(parameter.dtype)
            if dtype not in SCALAR_TYPES:
                raise ValueError(f"unsupported scalar argument {parameter.name}: {dtype}")
            arguments.append({"kind": "scalar", "name": str(parameter.name), "dtype": dtype})
    if not any(item["kind"] == "buffer" for item in arguments):
        raise ValueError("the kernel has no buffer arguments")
    return arguments


def device_signature(kernel, lowered, cuda: str, symbols: dict) -> tuple[str, list[dict], dict]:
    functions = list(lowered.device_mod.functions.items())
    if len(functions) != 1:
        raise ValueError(f"this profile requires one CUDA kernel, found {len(functions)}")
    name, function = functions[0]
    entrypoint = str(name.name_hint)
    definitions = re.findall(
        r'extern\s+"C"\s+__global__\s+void\s+(?:__launch_bounds__\([^)]+\)\s+)?'
        r'([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{', cuda, flags=re.S,
    )
    if len(definitions) != 1 or definitions[0][0] != entrypoint:
        raise ValueError("lowered CUDA entrypoint does not match emitted source")
    parameters = [part.strip() for part in definitions[0][1].split(",") if part.strip()]
    if len(parameters) != len(function.params):
        raise ValueError("lowered argument count does not match emitted CUDA signature")
    buffers = {str(buffer.data.name): buffer for buffer in kernel.buffer_map.values()}
    scalars = {str(parameter.name): str(parameter.dtype) for parameter in kernel.params
               if parameter not in kernel.buffer_map}
    scalar_cuda_types = {
        "bool": {"bool"}, "float32": {"float"}, "float64": {"double"},
        "int8": {"int8_t", "signed char"}, "uint8": {"uint8_t", "unsigned char"},
        "int16": {"int16_t", "short"}, "uint16": {"uint16_t", "unsigned short"},
        "int32": {"int", "int32_t"}, "uint32": {"uint", "uint32_t", "unsigned int"},
        "int64": {"int64_t", "long long"}, "uint64": {"uint64_t", "unsigned long long"},
    }
    abi = []
    for parameter, declaration in zip(function.params, parameters):
        match = re.search(r"([A-Za-z_]\w*)\s*$", declaration)
        parameter_name, dtype = str(parameter.name), str(parameter.dtype)
        if not match or match[1] != parameter_name:
            raise ValueError("lowered argument order/names do not match emitted CUDA signature")
        if dtype == "handle":
            buffer = buffers.get(parameter_name)
            if buffer is None or "*" not in declaration:
                raise ValueError(f"unmapped CUDA buffer argument {parameter_name}")
            element = str(parameter.type_annotation.element_type.dtype)
            if element != str(buffer.dtype):
                raise ValueError(f"CUDA buffer dtype changed for {buffer.name}")
            abi.append({"kind": "buffer", "name": str(buffer.name), "dtype": element})
        else:
            if dtype not in SCALAR_TYPES or declaration[:match.start()].strip() not in scalar_cuda_types[dtype]:
                raise ValueError(f"unsupported emitted CUDA scalar signature: {declaration}")
            if scalars.get(parameter_name, symbols.get(parameter_name)) != dtype:
                raise ValueError(f"unmapped CUDA scalar argument {parameter_name}: {dtype}")
            abi.append({"kind": "scalar", "name": parameter_name, "dtype": dtype})
    attrs = function.attrs
    if "cluster_dims" in attrs or "use_cooperative_groups" in attrs:
        raise ValueError("cluster/cooperative launches need a dedicated CUDA launch adapter")
    extent = attrs.get("thread_extent")
    if extent is None:
        raise ValueError("lowered kernel has no thread extents")
    launch = {
        "grid": [integer_expression(extent.get(f"blockIdx.{axis}", 1), symbols) for axis in "xyz"],
        "block": [integer_expression(extent.get(f"threadIdx.{axis}", 1), symbols) for axis in "xyz"],
        "shared_memory_bytes": integer_expression(attrs.get("dyn_shared_memory_buf", 0), symbols),
    }
    return entrypoint, abi, launch
