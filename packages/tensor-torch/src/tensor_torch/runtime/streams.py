"""Current PyTorch streams cached by CUDA device and stream handle."""

import torch

_streams = {}


def current_stream(ordinal):
    # PyTorch exposes this low-overhead handle API to its own compiler backends.
    # Keep a public-API fallback for supported versions that omit the helper.
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if getter is None:
        return torch.cuda.current_stream(ordinal)
    handle = getter(ordinal)
    key = (ordinal, handle)
    stream = _streams.get(key)
    if stream is None:
        stream = torch.cuda.current_stream(ordinal)
        _streams[key] = stream
    return stream


def clear():
    _streams.clear()
