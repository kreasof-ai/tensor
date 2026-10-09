"""Optional model loading and inference without framework/compiler imports."""
from .common.gguf import GGUF, GGUFError

def __getattr__(name):
    if name in ('LFM2', 'LFM2Request'):
        from .lfm2.model import LFM2, LFM2Request
        return {'LFM2': LFM2, 'LFM2Request': LFM2Request}[name]
    if name == 'Tokenizer':
        from .common.tokenizer import Tokenizer
        return Tokenizer
    if name == 'Qwen35Batch':
        from .qwen35.decode import Qwen35Batch
        return Qwen35Batch
    if name == 'Qwen35Prefill':
        from .qwen35.prefill import Qwen35Prefill
        return Qwen35Prefill
    if name == 'Qwen35Checkpoint':
        from .qwen35.checkpoint import Qwen35Checkpoint
        return Qwen35Checkpoint
    if name == 'Qwen35MTP':
        from .qwen35.mtp.decode import Qwen35MTP
        return Qwen35MTP
    if name == 'Qwen35Verifier':
        from .qwen35.speculative.verifier import Qwen35Verifier
        return Qwen35Verifier
    if name == 'Qwen35Speculative':
        from .qwen35.speculative.engine import Qwen35Speculative
        return Qwen35Speculative
    raise AttributeError(name)

__all__ = ['GGUF', 'GGUFError', 'LFM2', 'LFM2Request', 'Tokenizer',
           'Qwen35Batch', 'Qwen35Prefill', 'Qwen35Checkpoint', 'Qwen35MTP', 'Qwen35Verifier', 'Qwen35Speculative']
