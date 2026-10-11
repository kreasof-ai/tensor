"""Validated executable calls and output allocation."""

from __future__ import annotations

import ctypes as c
from tensor.runtime.abi import (
    ABI_MAJOR,
    BoundCall,
    ExecutableDescriptor,
    WorkspaceRequirements,
)
from tensor.runtime.signature import (
    bind_shapes,
    buffer_argument,
    resolve_launch,
    resolve_shape,
)
from tensor.runtime.buffer import Buffer
from tensor.runtime.session import _identities


class Executable:
    def __init__(self, device, manifest: dict, module, function, image):
        self.device, self.manifest, self.module, self.function, self._image = (
            device,
            manifest,
            module,
            function,
            image,
        )
        self._shared_limit = 0
        self._generation = device._generation
        self._released = False
        self._handle = next(_identities)
        workspace = WorkspaceRequirements(0, 1, device.device_type, 0, 0)
        self._descriptor = ExecutableDescriptor(
            ABI_MAJOR,
            c.sizeof(ExecutableDescriptor),
            device.device_type,
            device.ordinal,
            device._session_id,
            self._handle,
            len(manifest.get("abi", manifest["arguments"])),
            int("async_launch" in device.capabilities),
            workspace,
        )
        device._executables[self._handle] = self

    def _check(self):
        if not self.device._open:
            raise self.device.error("device session is closed")
        if self._generation != self.device._generation:
            raise self.device.error("executable belongs to a previous device session")
        if self._released:
            raise self.device.error("executable has been released")

    @property
    def descriptor(self):
        self._check()
        return ExecutableDescriptor.from_buffer_copy(self._descriptor)

    def workspace_requirements(self):
        self._check()
        return WorkspaceRequirements.from_buffer_copy(self._descriptor.workspace)

    def release(self):
        if not self._released:
            self._check()
            self.device.synchronize()
            self._dispose()

    def _dispose(self):
        if not self._released:
            self.device._unload_executable(self)
            self._released = True
            self.device._executables.pop(self._handle, None)
            self.module = self.function = self._image = None

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()

    def _bind(self, args: tuple, kwargs: dict, *, include_outputs: bool):
        descriptors = [
            item
            for item in self.manifest["arguments"]
            if include_outputs or item["name"] not in self.manifest.get("outputs", [])
        ]
        if len(args) > len(descriptors):
            raise ValueError(
                f"kernel expects at most {len(descriptors)} arguments, received {len(args)}"
            )
        values = dict(zip((item["name"] for item in descriptors), args))
        names = {item["name"] for item in descriptors}
        dimensions = {}
        for name, value in kwargs.items():
            if name in values:
                raise ValueError(f"argument {name} supplied twice")
            if name in names:
                values[name] = value
            elif name in self.manifest.get("symbols", {}):
                dimensions[name] = value
            else:
                raise ValueError(f"unknown kernel argument {name}")
        missing = names - values.keys()
        if missing:
            raise ValueError(f"missing kernel arguments: {sorted(missing)}")
        for descriptor in descriptors:
            if buffer_argument(descriptor):
                value = values[descriptor["name"]]
                if not isinstance(value, Buffer):
                    raise TypeError(f"{descriptor['name']} must be a Tensor Buffer")
                if value.device is not self.device:
                    raise ValueError(
                        f"{descriptor['name']} belongs to a different device session"
                    )
                if value._released or value._generation != self._generation:
                    raise self.device.error("buffer has been released")
                # V1 omitted alignment metadata; retain CUDA allocation alignment there.
                alignment = descriptor.get("alignment", 256)
                if (
                    "opaque_buffer_handles" not in self.device.capabilities
                    and value.pointer % alignment
                ):
                    raise ValueError(
                        f"{descriptor['name']} needs {alignment}-byte pointer alignment"
                    )
        symbols, values = bind_shapes(self.manifest, values, dimensions)
        launch = resolve_launch(self.manifest["launch"], symbols)
        for field in ("grid", "block"):
            if any(
                value > limit
                for value, limit in zip(launch[field], self.device.limits[field])
            ):
                raise ValueError(f"launch {field} exceeds this device's limits")
        return values, symbols, launch

    def launch(self, *arguments, **bindings) -> None:
        """Launch with all frontend arguments; symbolic dimensions are inferred."""
        self._check()
        values, symbols, launch = self._bind(arguments, bindings, include_outputs=True)
        call = BoundCall(
            self.device, self.manifest, values, symbols, launch, validated=True
        )
        self.device._launch(self, call)

    def prepare(self, *inputs, **bindings):
        """Allocate declared outputs without launching, for repeated benchmarks."""
        self._check()
        outputs = self.manifest.get("outputs", [])
        if not outputs:
            raise ValueError(
                "artifact has no declared outputs; call launch with every buffer"
            )
        supplied, symbols, _ = self._bind(inputs, bindings, include_outputs=False)
        generated = {}
        try:
            for item in self.manifest["arguments"]:
                if item["name"] in outputs:
                    generated[item["name"]] = self.device.empty(
                        resolve_shape(item["shape"], symbols), item["dtype"]
                    )
        except BaseException:
            for buffer in generated.values():
                buffer.release()
            raise
        supplied.update(generated)
        ordered = tuple(supplied[item["name"]] for item in self.manifest["arguments"])
        # Explicit scalars are already present in ordered; only inferred symbols go in kwargs.
        implicit = {
            name: value for name, value in symbols.items() if name not in supplied
        }
        return ordered, implicit, generated

    def __call__(self, *inputs, **bindings):
        """Allocate declared outputs, launch, and return one Buffer or a tuple."""
        ordered, dimensions, generated = self.prepare(*inputs, **bindings)
        try:
            self.launch(*ordered, **dimensions)
        except BaseException:
            for buffer in generated.values():
                buffer.release()
            raise
        outputs = self.manifest["outputs"]
        result = tuple(generated[name] for name in outputs)
        return result[0] if len(result) == 1 else result
