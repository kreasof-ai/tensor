"""Own Tensor device sessions without owning PyTorch allocations."""

import atexit
import threading
import torch
from tensor.providers.cuda.device import Device
from tensor_torch.runtime.streams import clear

_sessions = {}
_lock = threading.RLock()


def session(ordinal):
    # Retain the primary context once; launch streams are selected for every call.
    with _lock:
        if ordinal not in _sessions:
            _sessions[ordinal] = Device(ordinal).__enter__()
        return _sessions[ordinal]


def close():
    """Synchronize and release adapter-owned modules, never PyTorch allocations."""
    with _lock:
        for device in _sessions.values():
            # Launches may have used any PyTorch stream on this device.
            with torch.cuda.device(device.ordinal):
                torch.cuda.synchronize()
            device.__exit__(None, None, None)
        _sessions.clear()
        clear()


atexit.register(close)
