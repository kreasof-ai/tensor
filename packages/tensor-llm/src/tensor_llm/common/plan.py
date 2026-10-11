"""One binding and execution backbone for all model adapters and providers."""

from tensor.runtime.abi import BoundCall


def bind(device, kernel, arguments):
    values, symbols, launch = kernel._bind(arguments, {}, include_outputs=True)
    return BoundCall(device, kernel.manifest, values, symbols, launch, validated=True)


class ExecutionPlan(list):
    """Ordered borrowed kernels/buffers, with optional diagnostic labels.

    Adapters describe their layer order; this class binds, submits, and captures
    it. List-compatible entries keep graph rewriting and profiling inspectable.
    The model/request owns the kernels and buffers; a plan never releases them.
    """

    def __init__(self, device, calls=(), *, labeled=False):
        super().__init__(calls)
        self.device = device
        self.labeled = labeled

    def bind(self, kernel, arguments, *, label=None):
        call = bind(self.device, kernel, arguments)
        entry = (kernel, call, label) if self.labeled else (kernel, call)
        self.append(entry)
        return call

    def submit(self):
        for entry in self:
            self.device._launch(entry[0], entry[1])

    def capture(self, *, resources):
        """Use the provider's prepared submission mechanism with borrowed state."""
        if self.device.info.get("provider", "cuda") == "webgpu":
            return self.device.prepare_plan(self)
        from tensor.providers.cuda_graph import CudaGraph

        return CudaGraph(self.device, self.submit, resources=resources)


def submit(device, calls, *, after_launch=None):
    """Replay an adapter's derived/sliced plan through the same execution path."""
    if after_launch is not None:
        for entry in calls:
            device._launch(entry[0], entry[1])
            after_launch(entry)
    elif isinstance(calls, ExecutionPlan):
        calls.submit()
    else:
        for entry in calls:
            device._launch(entry[0], entry[1])


def capture(device, calls, *, resources):
    plan = calls if isinstance(calls, ExecutionPlan) else ExecutionPlan(device, calls)
    return plan.capture(resources=resources)
