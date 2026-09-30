"""Reject ambiguous explicit launch overrides before lowering."""

import pytest

from tensor.compiler.build import BuildError, _launch


def test_launch_rejects_noninteger_and_oversized_blocks():
    with pytest.raises(BuildError, match="positive integers"):
        _launch({"grid": [2, True, 1], "block": [128, 1, 1], "shared_memory_bytes": 0})
    with pytest.raises(BuildError, match="1024 threads"):
        _launch({"grid": [1, 1, 1], "block": [1024, 2, 1], "shared_memory_bytes": 0})
