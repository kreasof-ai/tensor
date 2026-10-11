"""Public exports; implementation lives in the named modules."""

from tensor.providers.webgpu.device import Device
from tensor.providers.webgpu.adapter import probe

__all__ = ["Device", "probe"]
