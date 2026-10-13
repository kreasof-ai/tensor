"""Opaque WebGPU buffer handles and readback."""

from __future__ import annotations

from itertools import count

from tensor.runtime.buffer import Buffer, BufferView

_handles = count(1)


class OpaqueBuffer(Buffer):
    def to_numpy(self):
        # Copy submission is ordered after all writes/dispatches on this queue;
        # mapping its destination waits for completion. A separate queue-wide
        # fence before the copy adds another host round trip without ordering it.
        self._check()
        return self.device._download(self)

    def view(self, shape):
        return OpaqueView(self, shape)

    @property
    def pointer(self):
        raise BufferError("WebGPU buffers have opaque handles, not device pointers")


class OpaqueView(BufferView, OpaqueBuffer):
    """A new opaque binding descriptor sharing a base buffer's GPU storage."""

    def __init__(self, base, shape):
        super().__init__(base, shape)
        self._handle = next(_handles)
        self._allocated = base._allocated
        self._storage = base._storage
        self._readback = None
        self.device._buffer_handles[self._handle] = self

    def _dispose(self):
        if self._released:
            return
        if self._readback is not None:
            self._readback.destroy()
            self._readback = None
        self.device._buffer_handles.pop(self._handle, None)
        super()._dispose()
