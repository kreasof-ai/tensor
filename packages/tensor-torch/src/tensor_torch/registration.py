"""Functional custom operators, explicit output mutation, and fake execution."""

from itertools import count
import torch
from tensor.runtime.signature import buffer_argument

_operator_ids = count()


def register(kernel):
    if kernel.mutable_inputs:
        raise ValueError(
            "manual mutable kernels cannot register as functional custom operators"
        )
    if kernel._op is not None:
        return kernel._op
    import hashlib

    name = "kernel_" + hashlib.sha256(kernel.path.read_bytes()).hexdigest()[:16]
    # Multiple loads are independent custom ops with the same artifact semantics.
    name += "_" + str(next(_operator_ids))

    def unpack(tensors, scalars):
        ti, si = iter(tensors), iter(scalars)
        return tuple(
            next(ti) if buffer_argument(d) else next(si) for d in kernel.inputs
        )

    @torch.library.custom_op(
        "tensor_torch::" + name,
        mutates_args=(),
        schema="(Tensor[] tensors, Scalar[] scalars) -> Tensor[]",
    )
    def op(tensors, scalars):
        args = unpack(tensors, scalars)
        contract = kernel.metadata(args)
        outputs = [
            torch.empty(shape, dtype=dtype, device=contract[1])
            for shape, dtype in contract[0]
        ]
        kernel._launch(args, outputs, contract=contract, trusted_outputs=True)
        return outputs

    @op.register_fake
    def fake(tensors, scalars):
        return kernel.fake(unpack(tensors, scalars))

    @torch.library.custom_op(
        "tensor_torch::" + name + "_into",
        mutates_args={"outputs"},
        schema="(Tensor[] tensors, Scalar[] scalars, Tensor(a!)[] outputs) -> ()",
    )
    def into(tensors, scalars, outputs):
        kernel._launch(unpack(tensors, scalars), outputs)

    @into.register_fake
    def into_fake(tensors, scalars, outputs):
        expected = kernel.fake(unpack(tensors, scalars))
        if len(expected) != len(outputs):
            raise ValueError("wrong output count")
        for a, b in zip(expected, outputs):
            if (
                a.dtype != b.dtype
                or a.device != b.device
                or a.ndim != b.ndim
                or not b.is_contiguous()
            ):
                raise ValueError("output metadata mismatch")
            for lhs, rhs in zip(a.shape, b.shape):
                torch._check(lhs == rhs)

    kernel._op, kernel._into_op = op, into
    return op
