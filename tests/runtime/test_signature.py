"""Runtime shape/ABI guards are exercised without a compiler or GPU."""

import copy

import numpy as np
import pytest

from tensor.artifacts.format import ArtifactError, validate_manifest
from tensor.runtime.signature import bind_shapes, resolve_launch, scalar_value


def manifest():
    return {
        "format": "tensor.cuda", "format_version": 2, "kind": "cubin", "target": "sm_86",
        "entrypoint": "test", "source_sha256": "0"*64,
        "tilelang_version": "0.1.14", "tvm_ffi_version": "0.1.12", "op_set": [],
        "files": {"kernel.cubin": "0"*64, "kernel.tirx.json": "0"*64},
        "arguments": [
            {"kind": "buffer", "name": name, "dtype": "float32", "alignment": 64,
             "shape": [{"var": "n"}]} for name in ("a", "b", "c")
        ] + [{"kind": "scalar", "name": "scale", "dtype": "float32"}],
        "symbols": {"n": "int32"}, "outputs": ["c"],
        "abi": [{"kind": "buffer", "name": name, "dtype": "float32"} for name in ("a", "b", "c")]
               + [{"kind": "scalar", "name": "scale", "dtype": "float32"},
                  {"kind": "scalar", "name": "n", "dtype": "int32"}],
        "launch": {"grid": [{"op": "floordiv", "args": [{"op": "add", "args": [{"var": "n"}, 127]}, 128]}, 1, 1],
                   "block": [128, 1, 1], "shared_memory_bytes": 0},
    }


def test_shapes_bind_from_inputs_and_launch_geometry_follows():
    specification = validate_manifest(manifest())
    symbols, values = bind_shapes(specification, {"a": np.zeros(129, dtype="float32"),
                                                  "b": np.ones(129, dtype="float32"), "scale": 2.5})
    assert symbols == {"n": 129}
    assert values["scale"] == 2.5
    assert resolve_launch(specification["launch"], symbols)["grid"] == [2, 1, 1]
    with pytest.raises(ValueError, match="dimension n mismatch"):
        bind_shapes(specification, {"a": np.zeros(129, dtype="float32"), "b": np.zeros(128, dtype="float32")})
    with pytest.raises(ValueError, match="dimension n mismatch"):
        bind_shapes(specification, {"a": np.zeros(129, dtype="float32")}, {"n": 130})


@pytest.mark.parametrize("expression", ["n + 127", {"var": "unknown"},
    {"op": "exec", "args": [1, 1]}, {"op": [], "args": [1, 1]},
    {"cast": [], "value": 1}, {"op": "mul", "args": [1]}, True])
def test_manifest_rejects_unsafe_or_ambiguous_integer_expressions(expression):
    specification = manifest()
    specification["launch"]["grid"][0] = expression
    with pytest.raises(ArtifactError):
        validate_manifest(specification)


def test_manifest_rejects_scalar_abi_mismatch_and_output_scalar():
    specification = manifest()
    specification["abi"][-1]["dtype"] = "int64"
    with pytest.raises(ArtifactError, match="unmapped"):
        validate_manifest(specification)
    specification = manifest()
    specification["outputs"] = ["scale"]
    with pytest.raises(ArtifactError, match="output"):
        validate_manifest(specification)


def test_scalar_width_and_finite_float_guards():
    assert scalar_value((1 << 40)+3, "int64", "delta") == (1 << 40)+3
    for value in (1 << 63, -(1 << 63)-1):
        with pytest.raises(ValueError, match="int64 range"):
            scalar_value(value, "int64", "delta")
    with pytest.raises(ValueError, match="uint32 range"):
        scalar_value(-1, "uint32", "size")
    with pytest.raises(ValueError, match="integer"):
        scalar_value(True, "int32", "size")
    with pytest.raises(ValueError, match="finite"):
        scalar_value(1e40, "float32", "scale")


def test_geometry_rejects_bad_division_and_overflow_before_driver_launch():
    launch = manifest()["launch"]
    with pytest.raises(ValueError, match="positive"):
        resolve_launch(launch, {"n": -200})
    invalid = copy.deepcopy(launch)
    invalid["grid"][0]["args"][1] = 0
    with pytest.raises(ValueError, match="division by zero"):
        resolve_launch(invalid, {"n": 129})
    with pytest.raises(ValueError, match="overflow"):
        resolve_launch(launch, {"n": 1 << 63})
