"""A callable Tensor artifact with PyTorch-owned inputs and outputs."""

from collections import OrderedDict
from pathlib import Path
import threading
import torch
from tensor.artifacts.format import read_artifact
from tensor.runtime.signature import buffer_argument
from tensor_torch.signatures import metadata
from tensor_torch.registration import register
from tensor_torch.runtime.launch import launch
from tensor_torch.runtime.prepared import Prepared


class Kernel:
    """A functional artifact with PyTorch-owned outputs and explicit output names.

    Input buffers must be read-only in the exported kernel. Declared output buffers
    must be completely written and must not alias inputs. These are the functional
    custom-operator contract; artifacts do not encode arbitrary alias analysis.
    """

    def __init__(self, path, *, mutable_inputs=()):
        self.path = Path(path)
        self.manifest, _ = read_artifact(self.path)
        if self.manifest.get("provider", "cuda") != "cuda":
            raise ValueError("tensor-torch currently supports CUDA artifacts")
        names = self.manifest.get("outputs", [])
        if not names and not mutable_inputs:
            raise ValueError("functional kernels need declared outputs")
        self.inputs = [d for d in self.manifest["arguments"] if d["name"] not in names]
        self.outputs = [
            next(d for d in self.manifest["arguments"] if d["name"] == n) for n in names
        ]
        self.mutable_inputs = tuple(mutable_inputs)
        if len(set(self.mutable_inputs)) != len(self.mutable_inputs) or set(
            self.mutable_inputs
        ) - {d["name"] for d in self.inputs if buffer_argument(d)}:
            raise ValueError("mutable_inputs must name distinct input buffers")
        self._executables = {}
        self._prepared = OrderedDict()
        self._lock = threading.RLock()
        self._op = None
        self._into_op = None

    def metadata(self, args, *, symbolic=False):
        return metadata(self, args, symbolic=symbolic)

    def fake(self, args):
        outputs, device = self.metadata(args, symbolic=True)
        return [
            torch.empty(shape, dtype=dtype, device=device) for shape, dtype in outputs
        ]

    def _launch(self, args, outputs, **options):
        return launch(self, args, outputs, **options)

    def raw(self, *args):
        contract = self.metadata(args)
        outputs = [
            torch.empty(shape, dtype=dtype, device=contract[1])
            for shape, dtype in contract[0]
        ]
        self._launch(args, outputs, contract=contract, trusted_outputs=True)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def prepare(self, *args, outputs=None):
        """Bind fixed tensors/scalars once, retaining storage for repeated submission.

        The returned call launches on the current PyTorch stream and never allocates
        outputs. Preparation imports DLPack; call outputs are available as .outputs.
        Changing input data in place is supported; changing tensor metadata is not.
        """
        if outputs is None:
            outputs = self.fake(args)
        state = self._launch(args, list(outputs), enqueue=False)
        return Prepared(state, args, tuple(outputs))

    def register(self):
        return register(self)

    def __call__(self, *args):
        op = self._op or self.register()
        result = op(
            [v for d, v in zip(self.inputs, args) if buffer_argument(d)],
            [v for d, v in zip(self.inputs, args) if not buffer_argument(d)],
        )
        return result[0] if len(result) == 1 else tuple(result)

    def into(self, *args, outputs):
        """Explicitly mutate disjoint output buffers, with a registered mutation schema."""
        if self._into_op is None:
            self.register()
        self._into_op(
            [v for d, v in zip(self.inputs, args) if buffer_argument(d)],
            [v for d, v in zip(self.inputs, args) if not buffer_argument(d)],
            list(outputs),
        )
