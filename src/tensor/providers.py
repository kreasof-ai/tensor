"""Runtime provider selection, independent from compiler backend registration.

ABI 1 providers implement the shared Buffer/Executable workbench, with their
own memory, image loading, streams and events. Compiler providers are selected
only by the build command.
"""

from importlib import import_module

PROVIDERS = {"cuda": "tensor.cuda", "cpu": "tensor.cpu", "webgpu": "tensor.webgpu"}


def Device(ordinal=0, *, provider="cuda", stream=None, require_capabilities=()):
    if provider not in PROVIDERS:
        raise ValueError(f"unknown runtime provider {provider!r}; choose {', '.join(PROVIDERS)}")
    implementation = import_module(PROVIDERS[provider]).Device
    if missing := set(require_capabilities) - implementation.capabilities:
        raise ValueError(f"provider {provider} lacks capabilities: {sorted(missing)}")
    return implementation(ordinal, stream=stream)
