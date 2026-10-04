"""Validate the migrated addition artifact with only Tensor and NumPy."""

import argparse
from pathlib import Path

import numpy as np
import tensor as tx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--provider", choices=("cuda", "webgpu"), default="cuda")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    a_host = np.linspace(-1, 1, 1025, dtype=np.float32)
    b_host = np.full((1025,), 2, dtype=np.float32)
    expected = a_host + b_host
    with tx.Device(args.device, provider=args.provider) as device:
        kernel = device.load(args.artifact)
        a = device.from_numpy(a_host)
        b = device.from_numpy(b_host)
        result = kernel(a, b)
        tx.assert_close(result, expected)

        # Bind every argument, including the output, to reuse an allocation.
        result = device.empty((1025,), "float32")
        kernel.launch(a, b, result)
        tx.assert_close(result, expected)
        print("PASS: 1025 addition results match NumPy (allocated and reused outputs)")


if __name__ == "__main__":
    main()
