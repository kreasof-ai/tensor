"""Optional PyTorch adapter. Importing core tensor never imports this package."""
from pathlib import Path
from .backend import Backend, backend, reports
from .bridge import Kernel, close


def load(reference, *, project='.', module_cache=None, target=None, compile=False, **options):
    """Load a .tbin or an installed module::export as a functional custom op.

    Packaged execution is compiler-free. compile=True explicitly enables module
    source fallback; backend FX cache misses compile with NVRTC automatically.
    """
    if '::' in str(reference):
        from tensor.modules import resolve_reference
        if target is None:
            import torch
            target = 'sm_' + ''.join(map(str, torch.cuda.get_device_capability()))
        reference = resolve_reference(str(reference), project=project, module_cache=module_cache,
                                      target=target, compile=compile, **options)['path']
    elif options:
        raise ValueError('compiler options apply to module references only')
    kernel = Kernel(Path(reference))
    kernel.register()
    return kernel


__version__ = '0.1.0'
__all__ = ['Backend', 'Kernel', 'backend', 'reports', 'load', 'close']
