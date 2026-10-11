"""Composition must count a tied parameter once across child modules."""

from types import SimpleNamespace
from tensor_nn import Module, Parameter


def test_named_parameters_deduplicate_shared_child_weights():
    library = SimpleNamespace(device=None)
    parent = Module(library, name="parent")
    left = Module(library, name="left")
    right = Module(library, name="right")
    tied = Parameter(None, None, None, None, None, (8, 8))
    independent = Parameter(None, None, None, None, None, (8,))
    left.register_parameter("weight", tied)
    right.register_parameter("weight", tied)
    right.register_parameter("scale", independent)
    parent.register_module("left", left)
    parent.register_module("right", right)
    assert list(parent.named_parameters()) == [
        ("left.weight", tied),
        ("right.scale", independent),
    ]
    assert parent.parameters == {"left.weight": tied, "right.scale": independent}
