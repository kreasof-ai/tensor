"""Executable and event ownership within a device session."""

from __future__ import annotations

from itertools import count
import ctypes as c
from tensor.runtime.abi import ABI_MAJOR, EventDescriptor, ExecutableDescriptor


_identities = count(1)


class Session:
    """Common executable ownership; providers still own vendor resources."""

    def _start_session(self):
        self._session_id = next(_identities)
        self._executables = {}

    def get_executable(self, descriptor):
        if (
            not isinstance(descriptor, ExecutableDescriptor)
            or descriptor.abi_version != ABI_MAJOR
            or descriptor.struct_size < c.sizeof(ExecutableDescriptor)
        ):
            raise self.error("unsupported executable descriptor ABI")
        self._check()
        if (
            descriptor.device_type != self.device_type
            or descriptor.device_ordinal != self.ordinal
            or descriptor.session != self._session_id
        ):
            raise self.error(
                "executable descriptor belongs to a different device session"
            )
        executable = self._executables.get(descriptor.handle)
        if executable is None:
            raise self.error("executable handle is released or unknown")
        expected = executable._descriptor
        if (
            descriptor.argument_count != expected.argument_count
            or descriptor.flags != expected.flags
            or bytes(descriptor.workspace) != bytes(expected.workspace)
        ):
            raise self.error("executable descriptor metadata mismatch")
        return executable

    def get_event(self, descriptor):
        if (
            not isinstance(descriptor, EventDescriptor)
            or descriptor.abi_version != ABI_MAJOR
            or descriptor.struct_size < c.sizeof(EventDescriptor)
        ):
            raise self.error("unsupported event descriptor ABI")
        self._check()
        if (
            descriptor.device_type != self.device_type
            or descriptor.device_ordinal != self.ordinal
            or descriptor.session != self._session_id
        ):
            raise self.error("event descriptor belongs to a different device session")
        event = next(
            (
                item
                for item in self._events
                if item._descriptor.handle == descriptor.handle
            ),
            None,
        )
        if event is None:
            raise self.error("event handle is released or unknown")
        if descriptor.flags != event._descriptor.flags or descriptor.reserved:
            raise self.error("event descriptor metadata mismatch")
        return event
