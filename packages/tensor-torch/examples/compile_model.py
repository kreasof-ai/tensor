"""Compile a supported pointwise region and inspect its compilation report."""

import json
import torch
import tensor_torch


def main():
    backend = tensor_torch.Backend()
    function = torch.compile(lambda x, y: torch.relu(x * 2 + y), backend=backend)
    with torch.inference_mode():
        x = torch.randn(129, device="cuda")
        y = torch.randn_like(x)
        torch.testing.assert_close(function(x, y), torch.relu(x * 2 + y))
    print(json.dumps(backend.report, indent=2))


if __name__ == "__main__":
    main()
