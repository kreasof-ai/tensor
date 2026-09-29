"""The producer audit must work with Windows-style subprocess imports/calls."""

import subprocess
import sys

import pytest

from tools.phase2_producer import compiler_audit


def test_compiler_audit_preserves_popen_class_and_deactivates():
    original = subprocess.Popen
    with compiler_audit() as calls:
        # asyncio.windows_utils uses this inheritance during module import.
        class WindowsProcess(subprocess.Popen):
            pass

        assert issubclass(WindowsProcess, original)
        assert subprocess.check_output([sys.executable, "-c", "print('allowed')"], text=True).strip() == "allowed"
        with pytest.raises(RuntimeError, match="external compilation prohibited"):
            subprocess.run(["nvcc", "--version"], check=True)
        assert len(calls) == 1
    assert subprocess.Popen is original
    sys.audit("subprocess.Popen", "nvcc", ["nvcc", "--version"], None, None)
    assert len(calls) == 1


def test_compiler_audit_handles_windows_quoted_command_and_pathext_case():
    with compiler_audit() as calls:
        with pytest.raises(RuntimeError, match="external compilation prohibited"):
            sys.audit("subprocess.Popen", None, '"C:\\Program Files\\CUDA\\nvcc.EXE" --version', None, None)
        with pytest.raises(RuntimeError, match="external compilation prohibited"):
            sys.audit("subprocess.Popen", None, ["clang.exe", "--version"], None, None)
        assert len(calls) == 2
