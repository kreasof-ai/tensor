"""Opt-in real NVIDIA checks. Set TENSOR_P0_CUDA=1 on a CUDA build host."""

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("TENSOR_P0_CUDA") != "1", reason="requires opt-in NVIDIA host with nvcc")


@pytest.mark.parametrize("size", [1, 127, 128, 129, 1025])
def test_opaque_artifact_in_fresh_consumer_process(tmp_path, size):
    from experiments.p0.artifact_build import compile_bundle, prepare
    from experiments.p0.cuda_driver import Driver

    _, info = Driver().device_info()
    source, executable = tmp_path / "source.zip", tmp_path / "kernel.tbin"
    prepare(source, size=size, arch=info["arch"])
    compile_bundle(source, executable)
    consumer_python = os.environ.get("TENSOR_P0_RUNTIME_PYTHON", sys.executable)
    proc = subprocess.run([consumer_python, "-m", "experiments.p0.artifact_run", "validate", str(executable)],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["status"] == "passed"
    assert result["compiler_imports"] == []


def test_corrected_workload_numerics():
    from experiments.p0.numerics import run

    result = run()
    assert result["status"] == "passed"
    assert len(result["results"]) == 12
