"""Optional model loading and inference without framework/compiler imports."""
from .gguf import GGUF, GGUFError

def __getattr__(name):
    if name == 'LFM2':
        from .model import LFM2
        return LFM2
    if name == 'Tokenizer':
        from .tokenizer import Tokenizer
        return Tokenizer
    raise AttributeError(name)

__all__ = ['GGUF', 'GGUFError', 'LFM2', 'Tokenizer']
