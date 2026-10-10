"""One CUDA graph for drafting, verification, prefix commit, and draft repair."""

import ctypes as ct
import hashlib
import json

import numpy as np

from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall
from tensor_llm.common.artifacts import identity
from .artifacts import validate_implementation


class SpeculativeRound:
    def __init__(self, engine, count):
        self.engine = engine
        self.device = engine.target.device
        self.window = engine.window
        self.slots = 1 << (count - 1).bit_length()
        self.verifier = engine.target.batch(count, self.window, verify=True)
        self.repair = engine.drafter.batch(count, self.window)
        self.draft = engine.drafter.batch(count) if self.window > 2 else None
        directory = engine.target.bundle.resolve()
        manifest = json.loads(
            (directory / f"speculation-s{self.slots}-w{self.window}.json").read_text()
        )
        validate_implementation(manifest, engine.target.implementation)
        if (
            manifest["schema"] != "tensor.qwen35-dense-controls.v1"
            or manifest["target"] != self.device.info["arch"]
            or (manifest["slots"], manifest["window"]) != (self.slots, self.window)
        ):
            raise ValueError("speculative control profile mismatch")
        self.kernels = {}
        for key, row in manifest["kernels"].items():
            path = (directory / row["path"]).resolve()
            if (
                not path.is_relative_to(directory)
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise ValueError("speculative artifact integrity mismatch")
            self.kernels[key] = self.device.load(path)
        self.control = self.device.empty((self.slots, 5), "int32")
        self.result = self.device.empty((self.slots, self.window + 2), "int32")
        self.calls = []
        self._plan()
        resources = [self.control, self.result, *self.kernels.values()]
        for batch in (self.verifier, self.repair, self.draft):
            if batch is not None:
                resources.extend(batch.graph.resources)
        if hasattr(engine.target, "deferred"):
            journal = engine.target.deferred
            resources.extend((journal.counts, journal.banks))
            resources.extend(
                buffer for values in journal.saved.values() for buffer in values
            )
            resources.extend(journal.profile(self.verifier)["kernels"].values())
        self.graph = CudaGraph(
            self.device, self._submit, resources=tuple(dict.fromkeys(resources))
        )

    def _add(self, kind, arguments, **extra):
        parameters = dict(slots=self.slots, window=self.window, **extra)
        kernel = self.kernels[identity(kind, parameters)]
        values, symbols, launch = kernel._bind(arguments, {}, include_outputs=True)
        self.calls.append(
            (
                kernel,
                BoundCall(
                    self.device,
                    kernel.manifest,
                    values,
                    symbols,
                    launch,
                    validated=True,
                ),
            )
        )

    def _plan(self):
        v = self.verifier.buffers
        self._add(
            "spec_setup",
            (
                self.control,
                *(
                    v[n]
                    for n in (
                        "mapping",
                        "position",
                        "lengths",
                        "tokens",
                        "head_position",
                        "head_active",
                    )
                ),
            ),
        )
        for depth in range(1, self.window - 1):
            d = self.draft.buffers
            self._add(
                "spec_draft",
                (
                    self.control,
                    v["tokens"],
                    *(
                        d[n]
                        for n in (
                            "mapping",
                            "position",
                            "lengths",
                            "tokens",
                            "head_position",
                            "head_active",
                            "mode",
                        )
                    ),
                ),
                depth=depth,
            )
            self.calls.extend(self.draft.calls)
            self._add("spec_proposal", (d["predicted"], v["tokens"]), depth=depth)
        commits = self.verifier.commit_calls
        if hasattr(self.engine.target, "deferred"):
            scans, commits = self.engine.target.deferred.plans(self.verifier)
            scan_key = identity(
                "gdn_scan",
                dict(
                    slots=self.slots,
                    chunk=self.window,
                    pool=self.engine.capacity,
                    heads=self.engine.target.config.value_heads,
                    d=self.engine.target.config.key_dim,
                    tile=32,
                    commit=False,
                ),
            )
            scan_kernel = self.verifier.kernels[scan_key]
            replacements = iter(scans)
            self.calls.extend(
                next(replacements) if kernel is scan_kernel else (kernel, call)
                for kernel, call in self.verifier.calls
            )
        else:
            self.calls.extend(self.verifier.calls)
        self._add(
            "spec_accept",
            (self.control, v["tokens"], v["predicted"], v["lengths"], self.result),
        )
        self.calls.extend(commits)
        r = self.repair.buffers
        self._add(
            "spec_repair",
            (
                self.control,
                self.result,
                v["normal"],
                r["hidden"],
                *(
                    r[n]
                    for n in (
                        "mapping",
                        "position",
                        "lengths",
                        "tokens",
                        "head_position",
                        "head_active",
                        "mode",
                    )
                ),
            ),
            width=self.engine.target.config.width,
        )
        self.calls.extend(self.repair.calls)
        self._add("spec_result", (r["predicted"], self.result))

    def _submit(self):
        for kernel, call in self.calls:
            self.device._launch(kernel, call)

    def run(self, requests):
        target = self.engine.target
        target._check()
        self.engine.drafter._check()
        if not 1 <= len(requests) <= self.slots:
            raise ValueError("invalid speculative batch size")
        slots = [r.slot for r in requests]
        if len(set(slots)) != len(slots):
            raise ValueError("duplicate request state")
        control = np.zeros((self.slots, 5), dtype="int32")
        control[:, 0] = -1
        control[:, 3:] = -1
        for i, request in enumerate(requests):
            slot = request.slot
            if (
                slot is None
                or not 0 <= slot < target.capacity
                or not target.occupied[slot]
                or slot in target.pending
            ):
                raise ValueError("unavailable request state")
            remaining = min(
                self.window,
                request.limit - len(request.output),
                target.context - int(target.positions[slot]),
            )
            if remaining <= 0 or not request.output:
                raise ValueError("request cannot decode")
            control[i] = (
                slot,
                target.positions[slot],
                remaining,
                request.output[-1],
                self.engine.cached[slot],
            )
        self.device.driver.call(
            "cuMemcpyHtoD_v2",
            self.control.pointer,
            ct.c_void_p(control.ctypes.data),
            control.nbytes,
        )
        self.device.driver.call("cuStreamSynchronize", None)
        self.graph.launch()
        result = self.result.to_numpy()[: len(requests)]
        emitted = []
        for i, (request, row) in enumerate(zip(requests, result)):
            accepted = int(row[0])
            if not 1 <= accepted <= control[i, 2]:
                raise RuntimeError("invalid device acceptance count")
            target.positions[request.slot] += accepted
            self.engine.drafter.positions[request.slot] = target.positions[request.slot]
            self.engine.cached[request.slot] = int(row[-1])
            if hasattr(target, "deferred"):
                target.deferred.dirty[request.slot] = accepted > 1
            emitted.append(row[1 : accepted + 1].tolist())
        stats = self.engine.stats
        stats["rounds"] += 1
        stats["request_rounds"] += len(requests)
        stats["proposed"] += int(np.sum(control[: len(requests), 2] - 1))
        stats["accepted"] += sum(len(tokens) - 1 for tokens in emitted)
        stats["emitted"] += sum(map(len, emitted))
        return emitted

    def close(self):
        self.graph.close()
        self.control.release()
        self.result.release()
        for kernel in self.kernels.values():
            kernel.release()
