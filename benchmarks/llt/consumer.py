"""Run from an installed-wheel environment outside the source tree.

Pass a warmed artifact directory and output JSON. Copy model.py beside this file
as model_fixture.py. No compiler package may be installed or imported.
"""

import builtins
import gc
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys

original_import = builtins.__import__


def guarded(name, *args, **kwargs):
    if name.split(".")[0] in {"tilelang", "tvm", "tvm_ffi"}:
        raise ImportError("compiler import blocked: " + name)
    return original_import(name, *args, **kwargs)


builtins.__import__ = guarded
import numpy as np
import torch
import tensor
import tensor_torch
from tensor_torch.llt import Operators, AdamW
from model_fixture import Config, Model

assert "site-packages" in tensor.__file__, tensor.__file__
assert "site-packages" in tensor_torch.__file__, tensor_torch.__file__
installed = {d.metadata["Name"].lower() for d in metadata.distributions()}
assert not installed & {"tilelang", "apache-tvm-ffi", "apache-tvm"}, installed
ops = Operators(sys.argv[1])
torch.set_num_threads(4)
with tensor.Device() as device:
    buffer = device.from_numpy(
        np.linspace(-1, 1, 129, dtype="float32"), dtype="bfloat16"
    )
    borrowed = torch.from_dlpack(buffer)
    torch.testing.assert_close(
        borrowed, torch.linspace(-1, 1, 129, device="cuda").bfloat16()
    )
    del borrowed
    gc.collect()
    buffer.release()

losses = {}
for architecture in ("llt", "naive"):
    model = Model(Config(architecture=architecture, checkpoint=True), ops).cuda()
    optimizer = AdamW(model.parameters(), ops, lr=0.001, max_norm=1.0)
    x = torch.arange(33, device="cuda").reshape(1, -1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(x, x + 1)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss).item()
    losses[architecture] = loss.item()
    del model, optimizer
# Exercise the streaming classifier's manual accumulation in the consumer too.
x = torch.randn(7, 48, device="cuda", requires_grad=True)
w = torch.randn(259, 48, device="cuda", requires_grad=True)
labels = torch.arange(7, device="cuda")
streamed = ops.linear_cross_entropy(x, w, labels, chunk_size=2)
streamed.backward()
assert torch.isfinite(streamed).item() and torch.isfinite(w.grad).all().item()
# Prepared plans must use the matching native executor in this wheel.
assert ops.plans and all(p._native is not None for p in ops.plans.values())
assert all(a["cache_hit"] for a in ops.report["artifacts"])
assert not {"tilelang", "tvm", "tvm_ffi"} & sys.modules.keys()
assert not ops.report["fallbacks"]
result = {
    "status": "passed",
    "torch": torch.__version__,
    "tensor": tensor.__file__,
    "tensor_torch": tensor_torch.__file__,
    "compiler_packages": False,
    "native_plans": len(ops.plans),
    "losses": losses,
    "coverage": ops.report,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
}
Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + "\n")
print("Compiler-free installed consumer passed:", losses, flush=True)
