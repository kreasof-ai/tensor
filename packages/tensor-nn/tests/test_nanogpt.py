"""Training bundle coverage, compiler-free imports and opt-in manual training."""

import ast
import os
from pathlib import Path
import subprocess
import sys
import pytest
from tensor_nn_examples.nanogpt import GPTConfig, requirements, parameter_shapes, initial_weights
from tensor_nn.kernels import source


def test_training_templates_and_tied_parameter_identity():
    for config in [GPTConfig.diagnostic(), GPTConfig()]:
        requirements_ = requirements(config)
        assert len(requirements_) == len(set(requirements_))
        for kind, p in requirements_.values():
            ast.parse(source(kind, p))
        weights = initial_weights(config)
        assert {
            name: value.shape for name, value in weights.items()
        } == parameter_shapes(config)
        assert "head" not in weights
        assert all(value.dtype.name == "float32" for value in weights.values())


@pytest.mark.parametrize(
    "parameters",
    [
        {"heads": 0},
        {"layers": True},
        {"width": 31},
        {"vocab": 257},
        {"loss_scale": float("nan")},
        {"learning_rate": 0},
    ],
)
def test_invalid_training_configuration(parameters):
    with pytest.raises(ValueError):
        GPTConfig(**parameters)


def test_training_runtime_imports_without_framework_or_compiler():
    script = """import importlib.abc,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','tilelang','tvm','tvm_ffi','triton'}:raise ImportError(fullname)
sys.meta_path.insert(0,Guard())
from tensor_nn import ManualFunction
assert not any(name.startswith("tensor_nn_examples") for name in sys.modules)
from tensor_nn_examples.nanogpt import NanoGPT,GPTConfig
assert GPTConfig.diagnostic().width==32
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_bundle_schema_and_two_distribution_fingerprints_are_enforced(tmp_path):
    import json
    from types import SimpleNamespace
    from tensor_nn_examples.nanogpt import TrainingBundle
    from tensor_nn_examples.nanogpt.provenance import implementation_hashes

    from tensor_nn.provenance import implementation_hashes as template_hashes

    assert not any(name.startswith("tensor_nn_examples") for name in template_hashes())
    hashes = implementation_hashes()
    assert "tensor.runtime.manual" in hashes and "tensor_nn_examples.nanogpt" in hashes
    device = SimpleNamespace(info={"provider": "cuda"})
    path = tmp_path / "training.json"
    path.write_text(json.dumps({"schema": "tensor.manual-nanogpt.v1"}))
    with pytest.raises(ValueError, match="rebuild"):
        TrainingBundle(tmp_path, device)
    path.write_text(
        json.dumps(
            {
                "schema": "tensor.manual-nanogpt.v2",
                "config": vars(GPTConfig.diagnostic()),
                "implementation_sha256": {},
            }
        )
    )
    with pytest.raises(ValueError, match="matching Tensor and tensor-nn"):
        TrainingBundle(tmp_path, device)


CUDA = pytest.mark.skipif(
    os.environ.get("TENSOR_PHASE6") != "1",
    reason="set TENSOR_PHASE6=1 for CUDA training validation",
)


@pytest.fixture(scope="module")
def diagnostic_bundle(tmp_path_factory):
    from benchmarks.training.nanogpt.producer import produce

    directory = Path(os.environ.get("TENSOR_PHASE6_DIAGNOSTIC_BUNDLE", ""))
    if not (directory / "training.json").is_file():
        directory = tmp_path_factory.mktemp("nanogpt") / "bundle"
        produce(directory, GPTConfig.diagnostic())
    return directory


@CUDA
def test_ten_manual_updates_match_pytorch(tmp_path, diagnostic_bundle):
    from benchmarks.training.nanogpt.validate import validate

    report = validate(diagnostic_bundle, steps=10, out=tmp_path / "result.json")
    assert report["status"] == "passed" and len(report["steps"]) == 10


@CUDA
def test_composed_modules_share_weights_and_consume_contexts(diagnostic_bundle):
    import numpy as np
    import tensor
    from tensor_nn_examples.nanogpt import NanoGPT

    with tensor.Device() as device:
        model = NanoGPT(diagnostic_bundle, device)
        assert model.head.weight is model.embedding.token is model.parameters["token"]
        registered = list(model.named_parameters())
        assert len({id(parameter) for _, parameter in registered}) == len(registered)
        assert len(registered) == len(parameter_shapes(model.config))
        batch = np.zeros((model.config.batch, model.config.sequence), dtype=np.int32)
        model.set_batch(batch, batch)
        _, context = model(model.tokens)
        with pytest.raises(RuntimeError, match="previous backward context"):
            model(model.tokens)
        model.discard_contexts()
        assert context.consumed
        # Discarding the composite must also release every child's pending tape.
        model.forward_backward()
        assert model.function._pending is None
        for block in model.blocks:
            assert block.function._pending is None
            assert all(
                child.function._pending is None for child in block._modules.values()
            )
        model.update()
        assert model.steps == 1 and np.isfinite(model.read_loss())
        model.reset()
        assert model.steps == 0
        model.forward_backward()
        assert np.isfinite(model.read_loss())
