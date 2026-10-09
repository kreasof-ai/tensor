"""Optional model loading and inference without framework/compiler imports."""
from .gguf import GGUF, GGUFError

def __getattr__(name):
    if name in ('LFM2', 'LFM2Request'):
        from .model import LFM2, LFM2Request
        return {'LFM2': LFM2, 'LFM2Request': LFM2Request}[name]
    if name == 'Tokenizer':
        from .tokenizer import Tokenizer
        return Tokenizer
    raise AttributeError(name)

__all__ = ['GGUF', 'GGUFError', 'LFM2', 'LFM2Request', 'Tokenizer']
