"""Toolkit-free adapter builds, with an opt-in PyTorch-versioned C++ executor."""
import os
from pathlib import Path
from setuptools import Extension, setup

extensions = [Extension('tensor_torch._launch', ['src/tensor_torch/launch.c'],
    define_macros=[('Py_LIMITED_API','0x030C0000')], py_limited_api=True,
    libraries=[] if os.name=='nt' else ['dl'], optional=True)]
options = {'build': {'build_base': 'build/portable'},
           'bdist_wheel': {'py_limited_api': 'cp312'}}
if os.environ.get('TENSOR_TORCH_BUILD_NATIVE') == '1':
    # Use the builder's installed Torch. Do not download Torch in isolated builds
    # or use CUDAExtension: this is CPU C++ using Torch's generic device hooks.
    import torch
    version = tuple(map(int, torch.__version__.split('.')[:2]))
    if version != (2, 14):
        raise RuntimeError('the native executor currently targets PyTorch 2.14')
    module = '_executor_2_14'
    root = Path(torch.__file__).parent
    extensions.append(Extension('tensor_torch.' + module, ['src/tensor_torch/executor.cpp'],
        include_dirs=[str(root/'include'), str(root/'include/torch/csrc/api/include')],
        library_dirs=[str(root/'lib')], libraries=['torch_python', 'torch_cpu', 'c10'] + ([] if os.name=='nt' else ['dl']),
        define_macros=[('TENSOR_EXECUTOR_MODULE', module),
                       ('_GLIBCXX_USE_CXX11_ABI', str(int(torch.compiled_with_cxx11_abi())))],
        extra_compile_args=['/std:c++20', '/O2'] if os.name=='nt' else ['-std=c++20', '-O3', '-g0', '-fvisibility=hidden'],
        extra_link_args=[] if os.name=='nt' else ['-Wl,-rpath,$ORIGIN/../torch/lib'],
        language='c++'))
    # THPVariable access uses the CPython and Torch ABIs. Never label this wheel
    # abi3; the independent submission shim remains usable in portable wheels.
    options = {'build': {'build_base': 'build/native-2.14'}}

setup(ext_modules=extensions, options=options)
