"""Bounded, manually differentiated neural-network training workloads."""
from tensor.manual import ManualFunction, BackwardContext
from .nanogpt import GPTConfig, NanoGPT

__all__ = ['ManualFunction', 'BackwardContext', 'GPTConfig', 'NanoGPT']
