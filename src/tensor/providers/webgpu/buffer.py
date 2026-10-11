"""Opaque WebGPU buffer handles and readback."""

from __future__ import annotations

from itertools import count
from tensor.runtime.buffer import Buffer


_handles = count(1)


class OpaqueBuffer(Buffer):
    def to_numpy(self):
        # Copy submission is ordered after all writes/dispatches on this queue;
        # mapping its destination waits for completion. A separate queue-wide
        # fence before the copy adds another host round trip without ordering it.
        self._check()
        return self.device._download(self)

    @property
    def pointer(self):
        raise BufferError("WebGPU buffers have opaque handles, not device pointers")
