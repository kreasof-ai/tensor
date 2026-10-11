"""A small composition template around Tensor's explicit ManualFunction API."""

from tensor import ManualFunction


class Module:
    """A static-buffer module with an explicit forward/backward context.

    Register shared Parameter objects and child modules explicitly. This example
    supplies composition and lifetime management, not automatic differentiation.
    One forward may be pending per module; consume it or discard it before reuse.
    """

    def __init__(self, library, *, name):
        self.library = library
        self.device = library.device
        self.name = name
        self._parameters = {}
        self._modules = {}

    def bind(self, forward, backward):
        self.function = ManualFunction(forward, backward, name=self.name)

    def register_parameter(self, name, parameter):
        self._parameters[name] = parameter
        return parameter

    def register_module(self, name, module):
        self._modules[name] = module
        return module

    def named_parameters(self, prefix="", *, _seen=None):
        seen = set() if _seen is None else _seen
        for name, parameter in self._parameters.items():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                yield prefix + name, parameter
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix + name + ".", _seen=seen)

    @property
    def parameters(self):
        return dict(self.named_parameters())

    def empty(self, n, dtype="float16"):
        return self.device.empty((n,), dtype)

    def run(self, kind, parameters, *buffers):
        self.library.run(kind, parameters, *buffers)

    def forward(self, *inputs):
        return self.function.forward(*inputs)

    def backward(self, context, *gradients):
        return self.function.backward(context, *gradients)

    def __call__(self, *inputs):
        return self.forward(*inputs)

    def discard_contexts(self):
        if hasattr(self, "function") and self.function._pending is not None:
            self.function._pending.discard()
        for module in self._modules.values():
            module.discard_contexts()
