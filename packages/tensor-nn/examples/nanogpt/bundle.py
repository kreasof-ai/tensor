"""Validate the nanoGPT example's configuration and complete kernel coverage."""

import json
from pathlib import Path
from tensor_nn.library import KernelLibrary
from tensor_nn_examples.nanogpt.config import GPTConfig
from tensor_nn_examples.nanogpt.plan import requirements
from tensor_nn_examples.nanogpt.provenance import implementation_hashes


class TrainingBundle(KernelLibrary):
    def __init__(self, directory, device):
        manifest = json.loads((Path(directory) / "training.json").read_text())
        if manifest["schema"] != "tensor.manual-nanogpt.v2":
            raise ValueError(
                "unsupported training bundle schema; rebuild for these templates"
            )
        self.config = GPTConfig(**manifest["config"])
        super().__init__(
            directory, device, manifest, implementation=implementation_hashes()
        )
        if set(requirements(self.config)) != set(manifest["kernels"]):
            raise ValueError("incomplete training kernel coverage")
