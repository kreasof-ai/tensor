"""CUDA selection and diagnostics work without a GPU or compiler packages."""

import os
import subprocess
import sys

import pytest

from experiments.p0 import cuda_toolchain
from experiments.p0.artifact_format import ArtifactError


@pytest.fixture
def compiler(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)
    binary = tmp_path / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    binary.parent.mkdir()
    binary.touch()
    binary.chmod(0o755)
    return binary


@pytest.mark.parametrize("variable", ["CUDA_HOME", "CUDA_PATH"])
def test_configured_toolkit_wins_over_path(compiler, monkeypatch, variable):
    monkeypatch.setenv(variable, str(compiler.parent.parent))
    monkeypatch.setenv("PATH", "")
    assert cuda_toolchain.resolve_nvcc() == str(compiler)


def test_explicit_compiler_wins_over_configured_toolkit(compiler, monkeypatch):
    monkeypatch.setenv("CUDA_HOME", "/missing/toolkit")
    assert cuda_toolchain.resolve_nvcc(str(compiler)) == str(compiler)


def test_unconfigured_compiler_uses_path(compiler, monkeypatch):
    monkeypatch.setenv("PATH", str(compiler.parent))
    assert cuda_toolchain.resolve_nvcc() == str(compiler)


def test_invalid_toolkit_does_not_silently_use_another_compiler(compiler, monkeypatch):
    missing = compiler.parent.parent / "missing"
    monkeypatch.setenv("CUDA_HOME", str(missing))
    monkeypatch.setenv("PATH", str(compiler.parent))
    with pytest.raises(ArtifactError, match="CUDA compiler unavailable") as error:
        cuda_toolchain.resolve_nvcc()
    assert str(missing) in str(error.value)


def test_missing_headers_report_original_error_and_remedy(monkeypatch):
    monkeypatch.setattr(cuda_toolchain, "resolve_nvcc", lambda nvcc: sys.executable)

    def run(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "CUDA test compiler", "")
        return subprocess.CompletedProcess(command, 1, "", "fatal error: cuda_runtime.h: No such file")

    monkeypatch.setattr(cuda_toolchain.subprocess, "run", run)
    with pytest.raises(ArtifactError) as error:
        cuda_toolchain.doctor()
    assert "cuda_runtime.h" in str(error.value)
    assert "CCCL" in str(error.value)
    assert "CUDA_HOME" in str(error.value)


def test_successful_exit_without_cubin_is_not_available(monkeypatch):
    monkeypatch.setattr(cuda_toolchain, "resolve_nvcc", lambda nvcc: sys.executable)
    monkeypatch.setattr(cuda_toolchain.subprocess, "run",
                        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "CUDA test compiler", ""))
    with pytest.raises(ArtifactError, match="did not produce an ELF cubin"):
        cuda_toolchain.doctor()
