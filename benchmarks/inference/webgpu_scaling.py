"""Prepare exact scaling-benchmark workloads for the native WebGPU provider."""
from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import hashlib
import time
from pathlib import Path

import numpy as np

import tensor as tx
from tensor.artifacts.format import read_artifact
from scripts.validation.webgpu_validation import specialize

ROOT = Path(__file__).resolve().parents[2]


class WebGPUCase:
    def __init__(self, device, name, args, reference, directory, cache):
        self.device, self.inputs, self.kernel = device, [], None
        directory.mkdir(parents=True, exist_ok=True)
        if name.startswith('pointwise'):
            example, constants = 'elementwise.py', {'SIZE': args[0].numel()}
        elif name.startswith('gemm'):
            m,k = args[0].shape
            n = args[1].shape[0]
            example, constants = 'webgpu_gemm.py', {
                'M':m, 'N':n, 'K':k, 'DTYPE':'float16', 'OUTPUT_DTYPE':'float16',
                'TRANSPOSE_B':True, 'USE_BIAS':True, 'RELU':True,
            }
        else:
            b,h,s,d = args[0].shape
            example, constants = 'flash_attention.py', {
                'BATCH':b, 'HEADS':h, 'SEQ_LEN':s, 'HEAD_DIM':d,
                'IS_CAUSAL':name.endswith('True'), 'BLOCK_M':8, 'BLOCK_N':16,
            }
        source = directory/'source.py'
        source.write_text(specialize((ROOT/'examples'/example).read_text(), constants))
        artifact = directory/'kernel.tbin'
        artifact.unlink(missing_ok=True)
        start = time.perf_counter()
        result = tx.build(source, artifact, provider='webgpu', cache_dir=cache)
        build_seconds = time.perf_counter()-start
        manifest,files = read_artifact(artifact)
        try:
            start = time.perf_counter()
            self.kernel = device.load(artifact)
            pipeline_seconds = time.perf_counter()-start
            hashes = []
            for arg in args:
                value = arg.detach().cpu().numpy()
                hashes.append(hashlib.sha256(value.tobytes()).hexdigest())
                self.inputs.append(device.from_numpy(value))
            expected = reference.detach().cpu().numpy()
            output = self()
            try:
                actual = output.to_numpy()
                np.testing.assert_allclose(actual, expected, atol=.002, rtol=.02)
                maximum_error = float(np.max(np.abs(actual.astype('float32')-expected.astype('float32'))))
            finally:
                output.release()
            self.evidence = {
                'provider':'webgpu', 'status':'passed', 'same_inputs_as_cuda':True,
                'inputs_sha256':hashes, 'maximum_absolute_error':maximum_error,
                'atol':.002, 'rtol':.02, 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
                'wgsl_sha256':hashlib.sha256(files['kernel.wgsl']).hexdigest(),
                'launch':manifest['launch'], 'workgroup_storage_bytes':manifest['webgpu']['workgroup_storage_bytes'],
                'build_seconds':build_seconds, 'build_cache_hit':result['cache_hit'],
                'pipeline_create_seconds':pipeline_seconds,
            }
        except BaseException:
            self.close()
            raise

    def __call__(self):
        return self.kernel(*self.inputs)

    def close(self):
        if self.kernel is not None:
            self.kernel.release()
            self.kernel = None
        for buffer in self.inputs:
            buffer.release()
        self.inputs.clear()
