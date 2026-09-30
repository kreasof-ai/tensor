"""Opt-in NVRTC compiler identity, shared ABI and cross-session event checks."""

import os
from pathlib import Path

import numpy as np
import pytest

import tensor as tx
from tensor.compiler.build import build_artifact

pytestmark = pytest.mark.skipif(os.environ.get("TENSOR_P2_CUDA") != "1",reason="set TENSOR_P2_CUDA=1")
ROOT=Path(__file__).resolve().parents[2]


def test_nvrtc_and_nvcc_cache_entries_and_numerics_are_distinct(tmp_path):
    source=ROOT / "examples/dynamic_affine.py"
    nvrtc=tmp_path / "nvrtc.tbin"
    nvcc=tmp_path / "nvcc.tbin"
    rtc=build_artifact(source,nvrtc,compiler="nvrtc",cache_dir=tmp_path / "cache")
    offline=build_artifact(source,nvcc,compiler="nvcc",cache_dir=tmp_path / "cache")
    assert rtc["cache_key"] != offline["cache_key"]
    with tx.Device() as device:
        kernels=[device.load(path) for path in (nvrtc,nvcc)]
        for size in (1,127,128,129,1025):
            a=device.arange(size); b=device.ones((size,))
            results=[kernel(a,b,scale=2.5).to_numpy() for kernel in kernels]
            np.testing.assert_array_equal(results[0],results[1])


def test_provider_event_orders_work_between_cuda_sessions(tmp_path):
    path=tmp_path / "elementwise.tbin"
    build_artifact(ROOT / "examples/elementwise.py",path,compiler="nvrtc")
    with tx.Device() as producer, tx.Device() as consumer:
        kernel=producer.load(path)
        a=producer.arange(129); b=producer.ones((129,))
        out=kernel(a,b)
        event=producer.record_event()
        assert event.descriptor.flags == 0
        consumer.wait(producer.get_event(event.descriptor))
        consumer.synchronize()
        tx.assert_close(out,2*a.to_numpy()+1)
        event.release()
        with tx.Device(provider="cpu") as cpu:
            event=producer.record_event()
            with pytest.raises(RuntimeError,match="different provider"):
                cpu.wait(event)
            event.release()


def test_cuda_executable_release_completes_queued_work(tmp_path):
    path=tmp_path / "elementwise.tbin"
    build_artifact(ROOT / "examples/elementwise.py",path,compiler="nvrtc")
    with tx.Device() as device:
        kernel=device.load(path)
        descriptor=kernel.descriptor
        assert descriptor.flags == 1 and descriptor.workspace.byte_size == 0
        out=kernel(device.arange(129),device.ones((129,)))
        kernel.release()
        kernel.release()
        tx.assert_close(out,2*np.arange(129,dtype='float32')+1)
        with pytest.raises(RuntimeError,match="released"):
            device.get_executable(descriptor)
        with pytest.raises(RuntimeError,match="released"):
            kernel(device.arange(129),device.ones((129,)))
        with device.load(path) as replacement:
            assert replacement.descriptor.handle != descriptor.handle
            assert device.get_executable(replacement.descriptor) is replacement
