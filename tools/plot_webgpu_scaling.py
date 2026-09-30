"""Compatibility entry point for scripts.plots.plot_webgpu_scaling."""
from pathlib import Path
import importlib
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if __name__ == "__main__":
    runpy.run_module("scripts.plots.plot_webgpu_scaling", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("scripts.plots.plot_webgpu_scaling")
