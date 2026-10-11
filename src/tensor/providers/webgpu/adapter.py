"""WebGPU adapter enumeration and diagnostics."""

from __future__ import annotations

from tensor.runtime.errors import TensorRuntimeError
from tensor.artifacts.webgpu import TARGET


def _adapters():
    try:
        import wgpu
    except ImportError as exc:
        raise TensorRuntimeError(
            "WebGPU needs the optional tensor-workspace[webgpu] extra"
        ) from exc
    try:
        adapters = wgpu.gpu.enumerate_adapters_sync()
    except Exception as exc:
        raise TensorRuntimeError(f"WebGPU adapter enumeration failed: {exc}") from exc
    # Software adapters are explicit in info; they are useful for contract CI,
    # but never count as a second physical GPU for Phase 5 acceptance.
    return sorted(adapters, key=lambda a: a.info["adapter_type"] == "CPU")


def probe(ordinal=0):
    adapters = _adapters()
    if type(ordinal) is not int or not 0 <= ordinal < len(adapters):
        raise TensorRuntimeError(
            f"WebGPU adapter ordinal {ordinal} unavailable ({len(adapters)} adapters)"
        )
    adapter = adapters[ordinal]
    return {
        "status": "ok",
        "ordinal": ordinal,
        "provider": "webgpu",
        "arch": TARGET,
        "adapter": dict(adapter.info),
        "features": sorted(adapter.features),
        "limits": dict(adapter.limits),
    }
