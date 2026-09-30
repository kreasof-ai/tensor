"""Compatibility entry point for benchmarks.nanogpt.benchmark."""
from pathlib import Path
import importlib
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    runpy.run_module("benchmarks.nanogpt.benchmark", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("benchmarks.nanogpt.benchmark")
