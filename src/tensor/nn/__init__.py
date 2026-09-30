"""Compatibility namespace for the optional tensor-nn distribution."""
from tensor.runtime.manual import ManualFunction, BackwardContext

__all__ = ['ManualFunction', 'BackwardContext', 'GPTConfig', 'NanoGPT']


def __getattr__(name):
    if name not in ('GPTConfig', 'NanoGPT'):
        raise AttributeError(name)
    try:
        import tensor_nn
    except ModuleNotFoundError as error:
        if error.name != 'tensor_nn':
            raise
        raise ImportError('Install tensor-nn alongside Tensor to use training templates') from error
    return getattr(tensor_nn, name)
