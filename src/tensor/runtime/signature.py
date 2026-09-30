"""Compiler-independent scalar, shape, and launch bindings for CUDA artifacts."""

from __future__ import annotations

import ctypes
import math
import numbers

SCALAR_TYPES = {
    "bool": ctypes.c_bool,
    "int8": ctypes.c_int8, "int16": ctypes.c_int16,
    "int32": ctypes.c_int32, "int64": ctypes.c_int64,
    "uint8": ctypes.c_uint8, "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32, "uint64": ctypes.c_uint64,
    "float32": ctypes.c_float, "float64": ctypes.c_double,
}
INTEGER_TYPES = set(SCALAR_TYPES) - {"bool", "float32", "float64"}
OPERATIONS = {"add", "sub", "mul", "floordiv", "floormod", "min", "max"}


def scalar_value(value, dtype: str, name: str):
    """Validate before constructing ctypes values, which otherwise wrap integers."""
    if dtype not in SCALAR_TYPES:
        raise ValueError(f"{name}: unsupported scalar dtype {dtype}")
    if dtype == "bool":
        if not isinstance(value, (bool,)):
            raise ValueError(f"{name} needs a bool")
        return value
    if dtype in INTEGER_TYPES:
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise ValueError(f"{name} needs an integer {dtype}")
        value = int(value)
        bits = ctypes.sizeof(SCALAR_TYPES[dtype]) * 8
        low, high = (0, (1 << bits) - 1) if dtype.startswith("u") else (-(1 << (bits-1)), (1 << (bits-1))-1)
        if not low <= value <= high:
            raise ValueError(f"{name} is outside the {dtype} range")
        return value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} needs a real {dtype}")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite and fit {dtype}") from exc
    converted = SCALAR_TYPES[dtype](value).value
    if not math.isfinite(value) or not math.isfinite(converted):
        raise ValueError(f"{name} must be finite and fit {dtype}")
    return converted


def validate_expression(expression, symbols: dict, *, depth: int = 0) -> set[str]:
    """Only a small integer expression tree is accepted; no Python eval is used."""
    if depth > 32:
        raise ValueError("integer expression exceeds 32 levels")
    if type(expression) is int:
        if not -(1 << 63) <= expression < (1 << 63):
            raise ValueError("integer expression constant exceeds int64")
        return set()
    if not isinstance(expression, dict):
        raise ValueError("invalid integer expression")
    if set(expression) == {"var"}:
        name = expression["var"]
        if not isinstance(name, str) or name not in symbols:
            raise ValueError(f"unknown dimension symbol {name}")
        return {name}
    if (set(expression) == {"cast", "value"} and isinstance(expression["cast"], str)
            and expression["cast"] in INTEGER_TYPES):
        return validate_expression(expression["value"], symbols, depth=depth+1)
    if (set(expression) != {"op", "args"} or not isinstance(expression["op"], str)
            or expression["op"] not in OPERATIONS
            or not isinstance(expression["args"], list) or len(expression["args"]) != 2):
        raise ValueError("invalid integer expression operation")
    result = set()
    for argument in expression["args"]:
        result.update(validate_expression(argument, symbols, depth=depth+1))
    return result


def evaluate(expression, symbols: dict) -> int:
    if type(expression) is int:
        return expression
    if "var" in expression:
        name = expression["var"]
        if name not in symbols:
            raise ValueError(f"cannot bind dimension {name}; supply a matching input or {name}=VALUE")
        return symbols[name]
    if "cast" in expression:
        return scalar_value(evaluate(expression["value"], symbols), expression["cast"], "dimension cast")
    left, right = (evaluate(value, symbols) for value in expression["args"])
    operation = expression["op"]
    if operation in ("floordiv", "floormod") and right == 0:
        raise ValueError("division by zero in integer expression")
    if operation == "add":
        result = left + right
    elif operation == "sub":
        result = left - right
    elif operation == "mul":
        result = left * right
    elif operation == "floordiv":
        result = left // right
    elif operation == "floormod":
        result = left % right
    elif operation == "min":
        result = min(left, right)
    elif operation == "max":
        result = max(left, right)
    else:
        raise ValueError(f"unsupported integer expression {operation}")
    if not -(1 << 63) <= result < (1 << 63):
        raise ValueError("integer expression overflow")
    return result


def buffer_argument(descriptor: dict) -> bool:
    return descriptor.get("kind", "buffer") == "buffer"


def bind_shapes(manifest: dict, supplied: dict, dimensions: dict | None = None) -> tuple[dict, dict]:
    """Infer direct dimension variables, then check every supplied buffer shape."""
    definitions = manifest.get("symbols", {})
    dimensions = dimensions or {}
    unknown = dimensions.keys() - definitions.keys()
    if unknown:
        raise ValueError(f"unknown dimensions: {sorted(unknown)}")
    bound = {name: scalar_value(value, definitions[name], name) for name, value in dimensions.items()}
    values = dict(supplied)
    for descriptor in manifest["arguments"]:
        name = descriptor["name"]
        if name not in values:
            continue
        value = values[name]
        if not buffer_argument(descriptor):
            values[name] = scalar_value(value, descriptor["dtype"], name)
            if name in definitions:
                if name in bound and bound[name] != values[name]:
                    raise ValueError(f"conflicting values for dimension {name}")
                bound[name] = values[name]
            continue
        if not hasattr(value, "shape") or not hasattr(value, "dtype"):
            raise TypeError(f"{name} must be a buffer with shape and dtype")
        if len(value.shape) != len(descriptor["shape"]) or str(value.dtype) != descriptor["dtype"]:
            raise ValueError(f"{name} needs shape {descriptor['shape']} and dtype {descriptor['dtype']}")
        for extent, actual in zip(descriptor["shape"], value.shape):
            if isinstance(extent, dict) and set(extent) == {"var"}:
                symbol = extent["var"]
                actual = scalar_value(actual, definitions[symbol], symbol)
                if symbol in bound and bound[symbol] != actual:
                    raise ValueError(f"{name}: dimension {symbol} mismatch; expected {bound[symbol]}, received {actual}")
                bound[symbol] = actual
    for descriptor in manifest["arguments"]:
        name = descriptor["name"]
        if name in values and buffer_argument(descriptor):
            expected = resolve_shape(descriptor["shape"], bound)
            if tuple(values[name].shape) != expected:
                raise ValueError(f"{name} needs shape {list(expected)} and dtype {descriptor['dtype']}")
    return bound, values


def resolve_shape(shape: list, symbols: dict) -> tuple[int, ...]:
    result = tuple(evaluate(value, symbols) for value in shape)
    if any(value < 1 for value in result):
        raise ValueError("buffer shape must contain positive extents")
    return result


def resolve_launch(launch: dict, symbols: dict) -> dict:
    result = {name: [evaluate(value, symbols) for value in launch[name]] for name in ("grid", "block")}
    result["shared_memory_bytes"] = evaluate(launch["shared_memory_bytes"], symbols)
    for name in ("grid", "block"):
        if any(value < 1 or value > (1 << 31)-1 for value in result[name]):
            raise ValueError(f"launch {name} exceeds positive int32 dimensions")
    if math.prod(result["block"]) > 1024:
        raise ValueError("launch block exceeds 1024 threads")
    if not 0 <= result["shared_memory_bytes"] <= (1 << 31)-1:
        raise ValueError("invalid shared memory size")
    return result
