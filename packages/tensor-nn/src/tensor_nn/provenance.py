"""Bind an experimental training bundle to both installed implementations."""

import hashlib
from importlib.util import find_spec
from pathlib import Path

IMPLEMENTATION_MODULES = ('tensor.runtime', 'tensor.runtime.buffer', 'tensor.runtime.errors', 'tensor.runtime.event', 'tensor.runtime.executable', 'tensor.runtime.session', 'tensor.runtime.timing', 'tensor.runtime.abi', 'tensor.runtime.signature', 'tensor.runtime.manual', 'tensor.providers.cuda.device', 'tensor.providers.cuda.driver', 'tensor.artifacts.format', 'tensor.artifacts.webgpu', 'tensor.compiler.tuning', 'tensor_nn.kernels', 'tensor_nn.provenance', 'tensor_nn.module', 'tensor_nn.parameter', 'tensor_nn.library', 'tensor_nn.optim', 'tensor_nn.templates', 'tensor_nn.templates.linear', 'tensor_nn.templates.normalization', 'tensor_nn.templates.activation', 'tensor_nn.templates.attention', 'tensor_nn.templates.embedding', 'tensor_nn.templates.loss', 'tensor_nn.kernels.elementwise', 'tensor_nn.kernels.linear', 'tensor_nn.kernels.embedding', 'tensor_nn.kernels.normalization', 'tensor_nn.kernels.attention', 'tensor_nn.kernels.loss', 'tensor_nn.kernels.optimizer',)


def implementation_hashes(extra_modules=()):
    return {
        name: hashlib.sha256(Path(find_spec(name).origin).read_bytes()).hexdigest()
        for name in (*IMPLEMENTATION_MODULES, *extra_modules)
    }
