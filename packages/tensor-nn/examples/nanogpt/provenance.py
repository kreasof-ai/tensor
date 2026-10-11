"""Fingerprint the example alongside the Tensor NN templates it composes."""

from tensor_nn.provenance import implementation_hashes as template_hashes

EXAMPLE_MODULES = (
    "tensor_nn_examples.nanogpt",
    "tensor_nn_examples.nanogpt.config",
    "tensor_nn_examples.nanogpt.plan",
    "tensor_nn_examples.nanogpt.block",
    "tensor_nn_examples.nanogpt.model",
    "tensor_nn_examples.nanogpt.bundle",
    "tensor_nn_examples.nanogpt.provenance",
)


def implementation_hashes():
    return template_hashes(EXAMPLE_MODULES)
