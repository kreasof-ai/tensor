"""Runtime provider selection, independent from compiler backend registration.

ABI 1 providers implement the shared Buffer/Executable workbench, with their
own memory, image loading, streams and events. Compiler providers are selected
only by the build command.
"""

from importlib import import_module

PROVIDERS = {"cuda": "tensor.providers.cuda", "cpu": "tensor.providers.cpu", "webgpu": "tensor.providers.webgpu"}


def Device(ordinal=0, *, provider="cuda", stream=None, require_capabilities=(), max_buffer_size=None):
    if provider not in PROVIDERS:
        raise ValueError(f"unknown runtime provider {provider!r}; choose {', '.join(PROVIDERS)}")
    implementation = import_module(PROVIDERS[provider]).Device
    if missing := set(require_capabilities) - implementation.capabilities:
        raise ValueError(f"provider {provider} lacks capabilities: {sorted(missing)}")
    if max_buffer_size is not None:
        if provider != "webgpu":
            raise ValueError("max_buffer_size is only supported by the WebGPU provider")
        return implementation(ordinal, stream=stream, max_buffer_size=max_buffer_size)
    return implementation(ordinal, stream=stream)
