"""Guard the first artifact profile against ambiguous launches and signatures."""

import pytest

from tensor.build import BuildError, _entrypoint, _launch


def test_launch_rejects_noninteger_and_oversized_blocks():
    with pytest.raises(BuildError, match="positive integers"):
        _launch({"grid": [2, True, 1], "block": [128, 1, 1], "shared_memory_bytes": 0})
    with pytest.raises(BuildError, match="1024 threads"):
        _launch({"grid": [1, 1, 1], "block": [1024, 2, 1], "shared_memory_bytes": 0})


def test_generated_kernel_must_match_pointer_only_export():
    source = 'extern "C" __global__ void kernel(float* a, int n) { }'
    with pytest.raises(BuildError, match="pointer-only"):
        _entrypoint(source, 2)
