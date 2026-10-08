#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TENSOR_NVRTC_HOME="${TENSOR_NVRTC_HOME:-$PWD/build/nvrtc-12.9}"
export TENSOR_LLT_CUDA=1
export TENSOR_P4_CUDA=1
python="${LLT_PYTHON:-$PWD/.venv/bin/python}"
mkdir -p build/llt-qualification
"$python" -m pytest -o addopts= -q tests/runtime/test_bfloat16.py packages/tensor-torch/tests/test_llt.py packages/tensor-torch/tests/test_adapter.py
"$python" -m benchmarks.llt.qualify cold
"$python" -m benchmarks.llt.qualify gradients
"$python" -m benchmarks.llt.qualify training --steps 1000 --lr 0.001
"$python" -m benchmarks.llt.leak
for phase in generation attention decode loss-memory systems; do
    "$python" -m benchmarks.llt.scaling "$phase"
done
