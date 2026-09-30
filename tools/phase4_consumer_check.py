"""Compatibility entry point for benchmarks.inference.fx_consumer."""
from pathlib import Path
import importlib
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    runpy.run_module("benchmarks.inference.fx_consumer", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("benchmarks.inference.fx_consumer")
