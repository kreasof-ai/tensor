"""Reproduce Tensor's current missing BF16 DLPack/ABI support."""
import torch
import tensor as tx
from tensor.runtime.abi import DTYPES

print('bfloat16 in Tensor ABI:', 'bfloat16' in DTYPES)
with tx.Device() as device:
    value=torch.ones(8,device='cuda',dtype=torch.bfloat16)
    device.from_dlpack(value)
