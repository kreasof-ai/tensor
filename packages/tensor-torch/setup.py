"""Optional compact submission shim; CUDA headers/libraries are not build inputs."""
import os
from setuptools import Extension, setup

setup(ext_modules=[Extension('tensor_torch._launch', ['src/tensor_torch/launch.c'],
    define_macros=[('Py_LIMITED_API','0x030C0000')], py_limited_api=True,
    libraries=[] if os.name=='nt' else ['dl'], optional=True)],
    options={'bdist_wheel':{'py_limited_api':'cp312'}})
