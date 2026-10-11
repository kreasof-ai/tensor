"""Provider-owned completion events."""

from __future__ import annotations

import ctypes as c
from tensor.runtime.abi import ABI_MAJOR, EventDescriptor
from tensor.runtime.errors import TensorRuntimeError
from tensor.runtime.session import _identities


class Event:
    """Provider-owned completion point. Buffers and external streams stay borrowed."""

    def __init__(self, device, handle):
        self.device, self.handle = device, handle
        self._generation = device._generation
        self._released = False
        self._descriptor = EventDescriptor(
            ABI_MAJOR,
            c.sizeof(EventDescriptor),
            device.device_type,
            device.ordinal,
            device._session_id,
            next(_identities),
            int("async_launch" not in device.capabilities),
            0,
        )

    @property
    def descriptor(self):
        self._check(self.device)
        return EventDescriptor.from_buffer_copy(self._descriptor)

    def _check(self, device):
        if (
            self._released
            or self._generation != self.device._generation
            or device.device_type != self.device.device_type
            or device.ordinal != self.device.ordinal
        ):
            raise TensorRuntimeError(
                "event is released or belongs to a different provider/device"
            )
        self.device._check()

    def release(self):
        if not self._released:
            self._check(self.device)
            self.device._dispose_event(self)
            self.device._events.remove(self)
            self._released = True
