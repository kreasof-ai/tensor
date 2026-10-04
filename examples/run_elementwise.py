"""Execute the quickstart artifact using only Tensor and NumPy."""

import argparse
from pathlib import Path

import numpy as np
import tensor as tx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path, help="elementwise .tbin from the quickstart")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    with tx.Device(args.device) as device:
        kernel = device.load(args.artifact)
        a = device.arange(129)
        b = device.ones((129,))
        result = kernel(a, b).to_numpy()
        expected = np.maximum(2 * np.arange(129, dtype=np.float32) + 1, 0)
        tx.assert_close(result, expected)
        print("PASS: 129 elementwise results match NumPy")
        print(result[:8])


if __name__ == "__main__":
    main()
