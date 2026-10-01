"""Optional producer-built native WebGPU encoding; ordinary wheels stay Python."""
import os
from setuptools import Extension, setup

extensions=[]
if os.environ.get('TENSOR_BUILD_WEBGPU_NATIVE')=='1':
    extensions.append(Extension('tensor.providers._webgpu_native',
        sources=['src/tensor/native/webgpu_plan.c']))
setup(ext_modules=extensions)
