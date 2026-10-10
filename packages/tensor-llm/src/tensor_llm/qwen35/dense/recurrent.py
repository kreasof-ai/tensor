"""Own exact speculative update journals and materialize state on demand."""

import ctypes as ct
import hashlib
import json

import numpy as np

from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall
from tensor_llm.common.artifacts import identity
from .artifacts import validate_implementation


class RecurrentJournal:
    def __init__(self, target, window):
        self.target = target
        self.device = target.device
        self.window = window
        self.dirty = np.zeros(target.capacity, dtype=bool)
        self.counts = self.device.empty((target.capacity,), "int32")
        self.banks = self.device.empty((target.capacity,), "int32")
        for buffer in (self.counts, self.banks):
            self.device.driver.call("cuMemsetD8_v2", buffer.pointer, 0, buffer.nbytes)
        self.saved = {}
        self.profiles = {}
        c = target.config
        for layer, kind in enumerate(c.layers):
            if kind == "linear_attention":
                shape = (2, target.capacity, window, c.value_heads)
                self.saved[layer] = [
                    self.device.empty(shape + (c.key_dim,), "float32") for _ in range(2)
                ]
                self.saved[layer].extend(
                    self.device.empty(shape, "float32") for _ in range(2)
                )

    def profile(self, batch):
        slots = batch.slots
        if slots not in self.profiles:
            directory = self.target.bundle.resolve()
            manifest = json.loads(
                (directory / f"recurrent-s{slots}-w{self.window}.json").read_text()
            )
            validate_implementation(manifest, self.target.implementation)
            if (
                manifest["schema"] != "tensor.qwen35-recurrent-journal.v1"
                or manifest["target"] != self.device.info["arch"]
                or (manifest["slots"], manifest["window"], manifest["pool"])
                != (slots, self.window, self.target.capacity)
            ):
                raise ValueError("recurrent journal profile mismatch")
            kernels = {}
            for key, row in manifest["kernels"].items():
                path = (directory / row["path"]).resolve()
                if (
                    not path.is_relative_to(directory)
                    or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
                ):
                    raise ValueError("recurrent artifact integrity mismatch")
                kernels[key] = self.device.load(path)
            mapping = self.device.empty((slots,), "int32")
            common = dict(slots=slots, window=self.window, pool=self.target.capacity)

            def bind(kind, args, **extra):
                kernel = kernels[identity(kind, dict(common, **extra))]
                values, symbols, launch = kernel._bind(args, {}, include_outputs=True)
                return kernel, BoundCall(
                    self.device,
                    kernel.manifest,
                    values,
                    symbols,
                    launch,
                    validated=True,
                )

            calls = []
            c = self.target.config
            for layer, saved in self.saved.items():
                calls.append(
                    bind(
                        "gdn_materialize",
                        (
                            mapping,
                            self.target.states[layer][0],
                            *saved,
                            self.counts,
                            self.banks,
                        ),
                        heads=c.value_heads,
                        d=c.key_dim,
                        tile=32,
                    )
                )
            calls.append(bind("defer_clear", (mapping, self.counts)))

            def submit():
                for kernel, call in calls:
                    self.device._launch(kernel, call)

            resources = (
                mapping,
                self.counts,
                self.banks,
                *kernels.values(),
                *(buffer for values in self.saved.values() for buffer in values),
                *(state[0] for state in self.target.states.values()),
            )
            graph = CudaGraph(self.device, submit, resources=resources)
            self.profiles[slots] = dict(
                kernels=kernels, mapping=mapping, graph=graph, bind=bind
            )
        return self.profiles[slots]

    def plans(self, verifier):
        profile = self.profile(verifier)
        bind = profile["bind"]
        c = self.target.config
        b = verifier.buffers
        scans, commits = [], []
        for layer, saved in self.saved.items():
            scans.append(
                bind(
                    "gdn_deferred",
                    (
                        *(
                            b[f"{layer}_{name}"]
                            for name in ("q", "k", "v", "g", "beta")
                        ),
                        b["mapping"],
                        b["lengths"],
                        self.target.states[layer][0],
                        b["gdn_out"],
                        *saved,
                        self.counts,
                        self.banks,
                    ),
                    heads=c.value_heads,
                    d=c.key_dim,
                    tile=32,
                )
            )
            commits.append(
                bind(
                    "conv_commit",
                    (
                        b[f"{layer}_gdn_input"],
                        b["mapping"],
                        b["lengths"],
                        self.target.states[layer][1],
                    ),
                    channels=c.qkv_width,
                    projection=c.qkv_width + c.value_width + 2 * c.value_heads,
                )
            )
        commits.append(
            bind("defer_accept", (b["mapping"], b["lengths"], self.counts, self.banks))
        )
        return scans, commits

    def materialize(self, requests):
        if not any(self.dirty[r] for r in requests):
            return
        slots = 1 << (len(requests) - 1).bit_length()
        # A profile is already created by the round that left this journal.
        if slots not in self.profiles:
            self.profile(self.target.batch(len(requests), self.window, verify=True))
        profile = self.profiles[slots]
        mapping = np.full(slots, -1, dtype="int32")
        mapping[: len(requests)] = requests
        self.device.driver.call(
            "cuMemcpyHtoD_v2",
            profile["mapping"].pointer,
            ct.c_void_p(mapping.ctypes.data),
            mapping.nbytes,
        )
        self.device.driver.call("cuStreamSynchronize", None)
        profile["graph"].launch()
        self.device.synchronize()
        self.dirty[requests] = False

    def forget(self, request):
        self.device.driver.call(
            "cuMemsetD8_v2", self.counts.pointer + request * 4, 0, 4
        )
        self.dirty[request] = False

    def close(self):
        for profile in self.profiles.values():
            profile["graph"].close()
            profile["mapping"].release()
            for kernel in profile["kernels"].values():
                kernel.release()
        for values in self.saved.values():
            for buffer in values:
                buffer.release()
        self.counts.release()
        self.banks.release()
