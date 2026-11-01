"""Bounded shapes reject unsafe calls before allocation and preserve old artifacts."""

import copy
import numpy as np
import pytest

import tensor as tx
from tensor.artifacts.format import ArtifactError, validate_manifest
from tensor.runtime.domain import check_domain, validate_domain
from tensor.runtime.signature import bind_shapes
from tests.runtime.test_signature import manifest


def domain():
    return {"version": 1, "bounds": {"n": {"min": 1, "max": 256}}, "relations": []}


def bounded_manifest():
    data = manifest()
    data.update(format="tensor.module", format_version=4, provider="cuda",
                compiler={"name": "nvrtc", "version": "12.9"},
                runtime_abi={"major": 1, "minor": 1, "required_capabilities": [
                    "executable_descriptors", "no_external_workspace", "symbolic_shapes"]},
                workspace={"bytes": 0, "alignment": 1}, shape_domain=domain())
    return data


@pytest.mark.parametrize("size", [1, 3, 7, 31, 32, 33, 127, 128, 129, 255, 256])
def test_domain_binds_live_extents(size):
    data = validate_manifest(bounded_manifest())
    bound, _ = bind_shapes(data, {"a": np.zeros(size, dtype="float32")})
    assert bound == {"n": size}


@pytest.mark.parametrize("size", [0, 257])
def test_domain_rejects_outside_dimensions(size):
    with pytest.raises(ValueError, match="positive|outside shape domain"):
        bind_shapes(bounded_manifest(), {"a": np.zeros(size, dtype="float32")})


def test_relations_and_divisibility_do_not_infer_ambiguous_factors():
    value = {"version": 1, "bounds": {"B": {"min": 1, "max": 256},
             "T": {"min": 1, "max": 4096, "multiple_of": 8}},
             "relations": [{"left": {"var": "B"}, "op": "le", "right": {"var": "T"}}]}
    validate_domain(value, {"B": "int32", "T": "int32"})
    check_domain(value, {"B": 3, "T": 24})
    for symbols, match in [({"B": 32, "T": 24}, "relation"),
                           ({"B": 3, "T": 25}, "divisible"), ({"T": 24}, "cannot bind")]:
        with pytest.raises(ValueError, match=match):
            check_domain(value, symbols)


def test_domain_integer_contracts_reject_constant_errors_and_numpy_overflow():
    with pytest.raises(ValueError):
        validate_domain(domain(), {"n": "float32"})
    for relation in ({"left": 2, "op": "le", "right": 1},
                     {"left": {"op": "floordiv", "args": [1, 0]}, "op": "eq", "right": 1}):
        value = domain()
        value["relations"] = [relation]
        with pytest.raises(ValueError):
            validate_domain(value, {"n": "int32"})
    value = domain()
    value["bounds"]["n"]["max"] = 1 << 62
    value["relations"] = [{"left": {"op": "mul", "args": [{"var": "n"}, 8]},
                            "op": "le", "right": 1}]
    validate_domain(value, {"n": "int64"})
    with pytest.raises(ValueError, match="overflow"):
        check_domain(value, {"n": np.int64(1 << 62)})
    with pytest.raises(ValueError, match="integer"):
        check_domain(domain(), {"n": 1.5})


@pytest.mark.parametrize("change", [
    lambda d: d.update(version=True),
    lambda d: d["bounds"]["n"].update(min=0),
    lambda d: d["bounds"]["n"].update(max=1 << 40),
    lambda d: d["bounds"]["n"].update(min=3, max=3, multiple_of=2),
    lambda d: d["bounds"].update(unknown={"min": 1, "max": 2}),
    lambda d: d["relations"].append({"left": {"var": "unknown"}, "op": "eq", "right": 1}),
    lambda d: d["relations"].append({"left": 1, "op": "exec", "right": 1}),
])
def test_invalid_domains_fail_artifact_validation(change):
    data = bounded_manifest()
    change(data["shape_domain"])
    with pytest.raises(ArtifactError):
        validate_manifest(data)


def test_mandatory_guards_cannot_hide_in_old_artifact_format():
    data = bounded_manifest()
    for revision in (1, 2, 3):
        changed = copy.deepcopy(data)
        changed["format_version"] = revision
        with pytest.raises(ArtifactError):
            validate_manifest(changed)
    del data["shape_domain"]
    with pytest.raises(ArtifactError):
        validate_manifest(data)


def test_capacity_prefix_bounds_logical_reads_and_nested_ranges():
    with tx.Device(provider="cpu") as device:
        base = device.from_numpy(np.arange(256, dtype="float32"))
        view = base.prefix((7, 16))
        assert view.pointer == base.pointer and view.nbytes == 7 * 16 * 4
        np.testing.assert_array_equal(view.to_numpy(), np.arange(112).reshape(7, 16))
        with pytest.raises(ValueError, match="capacity"):
            view.prefix((113,))
        nested = view.prefix((3, 16))
        np.testing.assert_array_equal(nested.to_numpy(), np.arange(48).reshape(3, 16))
        replacement = device.from_numpy(np.full(48, -7, dtype="float32"))
        nested.copy_from(replacement)
        expected = np.arange(256, dtype="float32")
        expected[:48] = -7
        np.testing.assert_array_equal(base.to_numpy(), expected)
        nested.release()
        base.release()
        with pytest.raises(RuntimeError, match="released"):
            view.to_numpy()
        view.release()


def test_public_binding_rejects_invalid_domain_before_output_allocation(monkeypatch):
    from tensor.runtime.executable import Executable

    with tx.Device(provider="cpu") as device:
        data = bounded_manifest()
        kernel = Executable(device, data, None, None, None)
        a = device.zeros((257,))
        monkeypatch.setattr(device, "empty", lambda *a, **kw: pytest.fail("allocated outside domain"))
        with pytest.raises(ValueError, match="outside shape domain"):
            kernel.prepare(a, a, scale=1.0)
        kernel._released = True
        device._executables.clear()


def test_public_bindings_are_independent_and_validate_borrowed_lifetimes():
    from tensor.runtime.executable import Executable

    with tx.Device(provider="cpu") as device:
        kernel = Executable(device, bounded_manifest(), None, None, None)
        storage = device.zeros((256,))
        a, b = storage.prefix((7,)), storage.prefix((129,))
        first = kernel.bind(a, a, a, scale=2.0)
        second = kernel.bind(b, b, b, scale=3.0)
        assert list(first.descriptor.grid) == [1, 1, 1]
        assert list(second.descriptor.grid) == [2, 1, 1]
        assert first.arguments[-1].scalar == 7 and second.arguments[-1].scalar == 129
        a.release()
        with pytest.raises(RuntimeError, match="released"):
            first.launch()
        kernel._released = True
        device._executables.clear()
