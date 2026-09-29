"""Tensor's CUDA kernel workbench. Compiler imports are deferred until build."""

from tensor.cuda import Buffer, Device, Executable, bench


def assert_close(actual, expected, *, rtol=1e-5, atol=1e-8) -> None:
    """Compare a device Buffer or host array with a NumPy reference."""
    import numpy as np

    if isinstance(actual, Buffer):
        actual = actual.to_numpy()
    if isinstance(expected, Buffer):
        expected = expected.to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def build(source, out, *, target=None, nvcc=None, cache_dir=None) -> dict:
    """Build a TileLang source file through the same path as `tensor build`."""
    from pathlib import Path
    from tensor.build import build_artifact

    return build_artifact(Path(source), Path(out), target=target, nvcc=nvcc, cache_dir=cache_dir)


def cache_info(cache_dir=None) -> dict:
    from pathlib import Path
    from tensor.build import cache_info as _cache_info

    return _cache_info(Path(cache_dir) if cache_dir else None)

__version__ = "0.1.0"

__all__ = ["Buffer", "Device", "Executable", "assert_close", "bench", "build", "cache_info"]
