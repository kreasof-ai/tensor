"""Run Tensor's dynamic_affine example artifact on PyTorch-owned buffers.

First build it from the Tensor checkout:
    tensor build examples/dynamic_affine.py --out build/dynamic_affine.tbin --compiler nvrtc
Then run:
    python packages/tensor-torch/examples/call_kernel.py build/dynamic_affine.tbin
"""

import argparse
import torch
import tensor_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    args = parser.parse_args()
    kernel = tensor_torch.load(args.artifact)
    x = torch.randn(129, device="cuda")
    y = torch.randn_like(x)
    result = kernel(x, y, 2.0)
    torch.testing.assert_close(result, torch.relu(x * 2.0 + y))
    print("Tensor artifact executed successfully")


if __name__ == "__main__":
    main()
