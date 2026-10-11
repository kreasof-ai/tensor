"""Load a verified training bundle and submit fixed-buffer Tensor calls."""

from pathlib import Path
import hashlib
from tensor.runtime.abi import BoundCall
from tensor_nn.kernels import identity
from tensor_nn.provenance import implementation_hashes


class KernelLibrary:
    """Verify a precompiled training bundle; cache validated fixed-buffer calls."""

    def __init__(self, directory, device, manifest, *, implementation=None):
        self.directory = Path(directory)
        self.device = device
        self.manifest = manifest
        if device.info["provider"] != "cuda":
            raise ValueError("NN templates currently require CUDA kernel bundles")
        expected = implementation_hashes() if implementation is None else implementation
        if expected != self.manifest.get("implementation_sha256"):
            raise ValueError(
                "install the matching Tensor and tensor-nn wheels for this training bundle"
            )
        if (
            hashlib.sha256((self.directory / "training.tpack").read_bytes()).hexdigest()
            != self.manifest["module_sha256"]
        ):
            raise ValueError("training module checksum mismatch")
        self.loaded = {}
        self.calls = {}
        for record in self.manifest["kernels"].values():
            path = self.directory / record["artifact"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("training artifact checksum mismatch")
        self.launch_count = 0

    def run(self, kind, p, *args):
        key = identity(kind, p)
        if key not in self.loaded:
            self.loaded[key] = self.device.load(
                self.directory / self.manifest["kernels"][key]["artifact"]
            )
        executable = self.loaded[key]
        call_key = (key, tuple(id(v) if hasattr(v, "pointer") else v for v in args))
        if call_key not in self.calls:
            values, symbols, launch = executable._bind(args, {}, include_outputs=True)
            self.calls[call_key] = BoundCall(
                self.device,
                executable.manifest,
                values,
                symbols,
                launch,
                validated=True,
            )
        call = self.calls[call_key]
        executable._check()
        for buffer in call.storage:
            if hasattr(buffer, "_released") and (
                buffer._released or buffer._generation != executable._generation
            ):
                raise RuntimeError("training plan references a released buffer")
        self.device._launch(executable, call)
        self.launch_count += 1
