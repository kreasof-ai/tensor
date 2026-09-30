"""Bind an experimental training bundle to both installed implementations."""
import hashlib
from importlib.util import find_spec
from pathlib import Path

IMPLEMENTATION_MODULES = (
    'tensor.runtime', 'tensor.runtime.abi', 'tensor.runtime.signature',
    'tensor.runtime.manual', 'tensor.providers.cuda',
    'tensor.artifacts.format', 'tensor.providers.webgpu_contract',
    'tensor.compiler.tuning', 'tensor_nn.nanogpt', 'tensor_nn.kernels',
    'tensor_nn.provenance',
)


def implementation_hashes():
    return {name: hashlib.sha256(Path(find_spec(name).origin).read_bytes()).hexdigest()
            for name in IMPLEMENTATION_MODULES}
