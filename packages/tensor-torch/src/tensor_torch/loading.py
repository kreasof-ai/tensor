"""Resolve installed module references or load an existing Tensor artifact."""

from pathlib import Path
from tensor_torch.kernel import Kernel


def load(
    reference, *, project=".", module_cache=None, target=None, compile=False, **options
):
    """Load a .tbin or an installed module::export as a functional custom op.

    Packaged execution is compiler-free. compile=True explicitly enables module
    source fallback; backend FX cache misses compile with NVRTC automatically.
    """
    if "::" in str(reference):
        from tensor.modules.project import resolve_reference

        if target is None:
            import torch

            target = "sm_" + "".join(map(str, torch.cuda.get_device_capability()))
        reference = resolve_reference(
            str(reference),
            project=project,
            module_cache=module_cache,
            target=target,
            compile=compile,
            **options,
        )["path"]
    elif options:
        raise ValueError("compiler options apply to module references only")
    kernel = Kernel(Path(reference))
    kernel.register()
    return kernel
