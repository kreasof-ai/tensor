"""Optional training templates; no framework or compiler imports at runtime."""
from tensor.runtime.manual import ManualFunction, BackwardContext
from .nanogpt import GPTConfig, NanoGPT

__all__ = ['ManualFunction', 'BackwardContext', 'GPTConfig', 'NanoGPT']
