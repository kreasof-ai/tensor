"""Shared weights and request state; compact, capacity-specific CUDA graphs."""

from __future__ import annotations

import ctypes as ct
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from threading import get_ident

import numpy as np

from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime import Buffer
from tensor.runtime.abi import BoundCall
from tensor_llm.common.artifacts import identity
from .checkpoint import DenseCheckpoint
from .projections import schedule
from .artifacts import implementation, validate_implementation


class ExecutionBatch:
    def __init__(self, owner, manifest):
        self.owner = owner
        self.device = owner.device
        value = json.loads(Path(manifest).read_text())
        validate_implementation(value, owner.implementation)
        if (
            value["schema"] != "tensor.qwen35-dense.v1"
            or value["config"] != json.loads(json.dumps(asdict(owner.config)))
            or value["target"] != self.device.info["arch"]
            or (value["pool"], value["context"]) != (owner.capacity, owner.context)
        ):
            raise ValueError("dense execution profile mismatch")
        self.slots, self.chunk = value["slots"], value["chunk"]
        self.rows = self.slots * self.chunk
        self.verify = value.get("verify", False)
        self.draft = value.get("draft", False)
        if self.draft != getattr(owner, "draft", False):
            raise ValueError("wrong checkpoint branch")
        self.head_rows = self.rows if self.verify else self.slots
        self.kernels = {}
        self.buffers = {}
        self.views = []
        self.graph = None
        self.commit_graph = None
        self.commit_calls = []
        directory = Path(manifest).resolve().parent
        for key, row in value["kernels"].items():
            path = (directory / row["path"]).resolve()
            if (
                not path.is_relative_to(directory)
                or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]
            ):
                raise ValueError("dense artifact integrity mismatch")
            self.kernels[key] = self.device.load(path)
        self._allocate()
        self.calls = self._plan()
        self.graph = CudaGraph(
            self.device,
            self._submit,
            resources=(
                *self.buffers.values(),
                *self.kernels.values(),
                *owner.weights.values(),
                *(b for state in owner.states.values() for b in state),
                *((owner.hidden,) if self.draft else ()),
            ),
        )
        if self.verify:
            self.commit_graph = CudaGraph(
                self.device, self._submit_commit, resources=self.graph.resources
            )

    def _allocate(self):
        c = self.owner.config
        r = self.rows
        s = self.slots

        def alloc(name, shape, dtype="float32"):
            self.buffers[name] = self.device.empty(shape, dtype)

        for name, shape in (
            ("tokens", (r,)),
            ("mapping", (s,)),
            ("position", (s,)),
            ("lengths", (s,)),
            ("flat_position", (r,)),
            ("active", (r,)),
            ("predicted", (self.head_rows,)),
            ("head_position", (self.head_rows,)),
            ("head_active", (self.head_rows,)),
        ):
            alloc(name, shape, "int32")
        for name in ("residual", "normal"):
            alloc(name, (r, c.width), "bfloat16")
        for name in ("mixed", "ffn"):
            alloc(name, (r, c.width))
        alloc("gdn_input", (r, c.qkv_width + c.value_width + 2 * c.value_heads))
        alloc("conv", (r, c.qkv_width))
        for name in ("q", "k", "v", "gdn_out"):
            alloc(name, (r, c.value_heads, c.key_dim))
        for name in ("g", "beta"):
            alloc(name, (r, c.value_heads))
        alloc("gdn_normal", (r, c.value_width), "bfloat16")
        alloc("att_input", (r, 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim))
        alloc("att_q", (r, c.heads, c.head_dim), "bfloat16")
        alloc("att_out", (r, c.heads * c.head_dim), "bfloat16")
        alloc("mlp_input", (r, 2 * c.intermediate))
        alloc("activation", (r, c.intermediate), "bfloat16")
        alloc("last", (s, c.width), "bfloat16")
        if self.owner.fused_head:
            tiles = (c.vocab + 63) // 64
            alloc("head_maximum", (self.head_rows, tiles))
            alloc("head_index", (self.head_rows, tiles), "int32")
        else:
            alloc("logits", (self.head_rows, c.vocab))
        for kind, rows, k, o in (
            ("gdn", r, c.width, c.qkv_width + c.value_width + 2 * c.value_heads),
            ("att", r, c.width, 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim),
            ("mlp", r, c.width, 2 * c.intermediate),
            ("out", r, c.value_width, c.width),
            ("down", r, c.intermediate, c.width),
            ("head", self.head_rows, c.width, c.vocab),
        ):
            parts = schedule(rows, k, o, c.width)["parts"]
            if parts > 1:
                alloc("projection_" + kind, (rows, parts, o))
        if self.verify:
            for layer, kind in enumerate(c.layers):
                if kind != "linear_attention":
                    continue
                alloc(
                    f"{layer}_gdn_input",
                    (r, c.qkv_width + c.value_width + 2 * c.value_heads),
                )
                for name in ("q", "k", "v"):
                    alloc(f"{layer}_{name}", (r, c.value_heads, c.key_dim))
                for name in ("g", "beta"):
                    alloc(f"{layer}_{name}", (r, c.value_heads))
        if self.draft:
            alloc("hidden", (r, c.width), "bfloat16")
            alloc("join", (r, 2 * c.width), "bfloat16")
            alloc("mode", (1,), "int32")

    def _plan(self):
        c = self.owner.config
        w = self.owner.weights
        b = self.buffers
        r = self.rows
        s = self.slots
        common = dict(slots=s, chunk=self.chunk, pool=self.owner.capacity)
        calls = []

        def add(kind, p, *args, to_commit=False):
            kernel = self.kernels[identity(kind, p)]
            values, symbols, launch = kernel._bind(args, {}, include_outputs=True)
            (self.commit_calls if to_commit else calls).append(
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

        def linear(name, x, out, rows=r):
            weight = w[name + ".weight"]
            o, k = weight.shape
            dest = (
                "head"
                if o == c.vocab
                else (
                    "gdn"
                    if o == c.qkv_width + c.value_width + 2 * c.value_heads
                    else (
                        "att"
                        if o == 2 * c.heads * c.head_dim + 2 * c.kv_heads * c.head_dim
                        else (
                            "mlp"
                            if o == 2 * c.intermediate
                            else "down" if k == c.intermediate else "out"
                        )
                    )
                )
            )
            p = schedule(rows, k, o, c.width)
            if p["parts"] == 1:
                # A contiguous singleton-axis view writes directly to the
                # output allocation; no separate copy/merge kernel is needed.
                scratch = Buffer(
                    self.device, out.pointer, (rows, 1, o), out.dtype, owner=out
                )
                self.views.append(scratch)
            else:
                scratch = b["projection_" + dest]
            add("split_linear", p, x, weight, scratch)
            if p["parts"] > 1:
                add("split_merge", p, scratch, out)

        def norm(name, x=None):
            args = (b["residual"], w[name + ".weight"], b["normal"])
            add(
                "rms" if x is None else "add_rms",
                dict(r=r, c=c.width, eps=c.epsilon),
                *((x, *args) if x is not None else args),
            )

        add(
            "controls",
            common,
            b["position"],
            b["lengths"],
            b["flat_position"],
            b["active"],
        )
        add(
            "embedding",
            dict(r=r, c=c.width, vocab=c.vocab),
            b["tokens"],
            w["model.language_model.embed_tokens.weight"],
            b["residual"],
        )
        if self.draft:
            add(
                "gather_hidden",
                dict(common, c=c.width),
                self.owner.hidden,
                b["mapping"],
                b["mode"],
                b["hidden"],
            )
            add(
                "mtp_join",
                dict(r=r, c=c.width, eps=c.epsilon),
                b["residual"],
                b["hidden"],
                w["mtp.pre_fc_norm_embedding.weight"],
                w["mtp.pre_fc_norm_hidden.weight"],
                b["join"],
            )
            linear("mtp.fc", b["join"], b["mixed"])
            add("mtp_cast", dict(r=r, c=c.width), b["mixed"], b["residual"])
        for layer, kind in enumerate(c.layers):
            root = f"model.language_model.layers.{layer}."
            norm(root + "input_layernorm", None if layer == 0 else b["ffn"])
            if kind == "linear_attention":
                prefix = root + "linear_attn."
                proj = c.qkv_width + c.value_width + 2 * c.value_heads
                saved = {
                    name: b[f"{layer}_{name}"] if self.verify else b[name]
                    for name in ("gdn_input", "q", "k", "v", "g", "beta")
                }
                linear(prefix + "input", b["normal"], saved["gdn_input"])
                state, history = self.owner.states[layer]
                control = dict(commit=False) if self.verify else {}
                conv_args = (
                    saved["gdn_input"],
                    w[prefix + "conv1d.weight"],
                    b["mapping"],
                    b["lengths"],
                    history,
                    b["conv"],
                )
                add(
                    "gdn_conv",
                    dict(common, **control, channels=c.qkv_width, projection=proj),
                    *conv_args,
                )
                add(
                    "gdn_prepare",
                    dict(
                        common,
                        heads=c.value_heads,
                        key_heads=c.key_heads,
                        d=c.key_dim,
                        channels=c.qkv_width,
                        projection=proj,
                    ),
                    b["conv"],
                    saved["gdn_input"],
                    w[prefix + "dt_bias"],
                    w[prefix + "A_log"],
                    saved["q"],
                    saved["k"],
                    saved["v"],
                    saved["g"],
                    saved["beta"],
                )
                scan_args = (
                    saved["q"],
                    saved["k"],
                    saved["v"],
                    saved["g"],
                    saved["beta"],
                    b["mapping"],
                    b["lengths"],
                    state,
                    b["gdn_out"],
                )
                add(
                    "gdn_scan",
                    dict(common, **control, heads=c.value_heads, d=c.key_dim, tile=32),
                    *scan_args,
                )
                if self.verify:
                    add(
                        "gdn_conv",
                        dict(common, channels=c.qkv_width, projection=proj),
                        *conv_args,
                        to_commit=True,
                    )
                    add(
                        "gdn_scan",
                        dict(common, heads=c.value_heads, d=c.key_dim, tile=32),
                        *scan_args,
                        to_commit=True,
                    )
                add(
                    "gdn_norm",
                    dict(
                        common,
                        heads=c.value_heads,
                        d=c.value_dim,
                        channels=c.qkv_width,
                        projection=proj,
                        eps=c.epsilon,
                    ),
                    b["gdn_out"],
                    saved["gdn_input"],
                    w[prefix + "norm.weight"],
                    b["gdn_normal"],
                )
                linear(prefix + "out_proj", b["gdn_normal"], b["mixed"])
            else:
                prefix = root + "self_attn."
                linear(prefix + "input", b["normal"], b["att_input"])
                kc, vc = self.owner.states[layer]
                att = dict(
                    common,
                    heads=c.heads,
                    kv_heads=c.kv_heads,
                    d=c.head_dim,
                    capacity=self.owner.context,
                    eps=c.epsilon,
                )
                add(
                    "attention_qkv",
                    dict(att, theta=c.theta),
                    b["att_input"],
                    w[prefix + "q_norm.weight"],
                    w[prefix + "k_norm.weight"],
                    b["mapping"],
                    b["flat_position"],
                    b["active"],
                    kc,
                    vc,
                    b["att_q"],
                )
                add(
                    "attention",
                    att,
                    b["att_q"],
                    kc,
                    vc,
                    b["att_input"],
                    b["mapping"],
                    b["position"],
                    b["lengths"],
                    b["att_out"],
                )
                linear(prefix + "o_proj", b["att_out"], b["mixed"])
            norm(root + "post_attention_layernorm", b["mixed"])
            linear(root + "mlp.input", b["normal"], b["mlp_input"])
            add(
                "swiglu",
                dict(common, c=c.intermediate),
                b["mlp_input"],
                b["activation"],
            )
            linear(root + "mlp.down_proj", b["activation"], b["ffn"])
        norm("model.language_model.norm", b["ffn"])
        if not self.verify:
            add(
                "last_rows",
                dict(common, c=c.width),
                b["normal"],
                b["lengths"],
                b["last"],
            )
        if self.draft:
            add(
                "store_hidden",
                dict(common, c=c.width),
                b["last"],
                b["mapping"],
                b["lengths"],
                self.owner.hidden,
            )
        if self.owner.fused_head:
            add(
                "head_linear",
                schedule(self.head_rows, c.width, c.vocab, c.width),
                b["normal"] if self.verify else b["last"],
                w["model.language_model.embed_tokens.weight"],
                b["head_maximum"],
                b["head_index"],
            )
            add(
                "head_argmax",
                dict(r=self.head_rows, tiles=(c.vocab + 63) // 64, vocab=c.vocab),
                b["head_maximum"],
                b["head_index"],
                b["head_active"],
                b["predicted"],
            )
        else:
            linear(
                "model.language_model.embed_tokens",
                b["normal"] if self.verify else b["last"],
                b["logits"],
                rows=self.head_rows,
            )
            add(
                "argmax",
                dict(r=self.head_rows, vocab=c.vocab),
                b["logits"],
                b["predicted"],
                b["head_position"],
                b["head_active"],
            )
        return calls

    def _submit(self):
        for kernel, call in self.calls:
            self.device._launch(kernel, call)

    def _submit_commit(self):
        for kernel, call in self.commit_calls:
            self.device._launch(kernel, call)

    def write(self, name, value):
        value = np.ascontiguousarray(value, dtype="int32")
        buffer = self.buffers[name]
        if value.shape != buffer.shape:
            raise ValueError("control shape mismatch")
        self.device.driver.call(
            "cuMemcpyHtoD_v2",
            buffer.pointer,
            ct.c_void_p(value.ctypes.data),
            value.nbytes,
        )

    def run(self, requests, sequences, *, hidden=None):
        self.owner._check()
        if self.verify and hasattr(self, "_verified"):
            raise ValueError(
                "commit or discard pending verification before reusing its workspace"
            )
        if any(request in self.owner.pending for request in requests):
            raise ValueError("request has an uncommitted verification")
        if hasattr(self.owner, "deferred"):
            self.owner.deferred.materialize(requests)
        if len(requests) > self.slots or len(requests) != len(sequences):
            raise ValueError("invalid execution batch")
        mapping = np.full(self.slots, -1, dtype="int32")
        positions = np.zeros(self.slots, dtype="int32")
        lengths = np.zeros(self.slots, dtype="int32")
        tokens = np.full((self.slots, self.chunk), -1, dtype="int32")
        for i, (request, sequence) in enumerate(zip(requests, sequences)):
            if (
                not 0 <= request < self.owner.capacity
                or not self.owner.occupied[request]
            ):
                raise ValueError("unowned request state")
            if not 1 <= len(sequence) <= self.chunk:
                raise ValueError("invalid chunk length")
            if self.owner.positions[request] + len(sequence) > self.owner.context:
                raise ValueError("context exhausted")
            if any(
                type(t) is not int or not 0 <= t < self.owner.config.vocab
                for t in sequence
            ):
                raise ValueError("invalid token ID")
            mapping[i] = request
            positions[i] = self.owner.positions[request]
            lengths[i] = len(sequence)
            tokens[i, : len(sequence)] = sequence
        if len(set(requests)) != len(requests):
            raise ValueError("duplicate request state in one batch")
        for name, value in (
            ("mapping", mapping),
            ("position", positions),
            ("lengths", lengths),
            ("tokens", tokens.reshape(-1)),
            ("head_position", np.zeros(self.head_rows, dtype="int32")),
            (
                "head_active",
                (
                    (np.arange(self.chunk)[None, :] < lengths[:, None]).reshape(-1)
                    if self.verify
                    else lengths > 0
                ).astype("int32"),
            ),
        ):
            self.write(name, value)
        if self.draft:
            if hidden is not None:
                hidden._check()
                if (
                    hidden.shape != self.buffers["hidden"].shape
                    or str(hidden.dtype) != "bfloat16"
                    or hidden.device is not self.device
                ):
                    raise ValueError("MTP requires aligned target BF16 hidden rows")
                self.device.driver.call(
                    "cuMemcpyDtoD_v2",
                    self.buffers["hidden"].pointer,
                    hidden.pointer,
                    hidden.nbytes,
                )
            elif self.chunk != 1:
                raise ValueError("recursive MTP drafting requires one token")
            self.write("mode", np.array([int(hidden is None)], dtype="int32"))
        self.device.driver.call("cuStreamSynchronize", None)
        self.graph.launch()
        predicted = self.buffers["predicted"].to_numpy()
        if self.verify:
            self._verified = (list(requests), lengths.copy())
            for request in requests:
                self.owner.pending[request] = self
            return predicted.reshape(self.slots, self.chunk)[: len(requests)].tolist()
        for request, sequence in zip(requests, sequences):
            self.owner.positions[request] += len(sequence)
        return predicted[: len(requests)].tolist()

    def commit(self, accepted):
        self.owner._check()
        if not self.verify or not hasattr(self, "_verified"):
            raise ValueError("no pending verification")
        requests, proposed = self._verified
        if len(accepted) != len(requests) or any(
            type(n) is not int or not 1 <= n <= proposed[i]
            for i, n in enumerate(accepted)
        ):
            raise ValueError("invalid accepted prefix lengths")
        lengths = np.zeros(self.slots, dtype="int32")
        lengths[: len(accepted)] = accepted
        self.write("lengths", lengths)
        self.device.driver.call("cuStreamSynchronize", None)
        self.commit_graph.launch()
        self.device.synchronize()
        for request, count in zip(requests, accepted):
            self.owner.positions[request] += count
        for request in requests:
            del self.owner.pending[request]
        del self._verified

    def discard(self):
        if not self.verify or not hasattr(self, "_verified"):
            raise ValueError("no pending verification")
        for request in self._verified[0]:
            del self.owner.pending[request]
        del self._verified

    def close(self):
        if self.commit_graph:
            self.commit_graph.close()
            self.commit_graph = None
        if self.graph:
            self.graph.close()
            self.graph = None
        for b in self.buffers.values():
            b.release()
        for k in self.kernels.values():
            k.release()


class DenseEngine:
    def __init__(
        self,
        checkpoint,
        bundle,
        device,
        *,
        capacity=256,
        context=4096,
        fused_head=False,
    ):
        self.device = device
        self.thread = get_ident()
        self.capacity = capacity
        self.context = context
        self.fused_head = fused_head
        self.implementation = implementation()
        self.pending = {}
        self.closed = False
        self.generation = device._generation
        self.checkpoint = DenseCheckpoint(checkpoint)
        self.config = self.checkpoint.config
        self.positions = np.zeros(capacity, dtype="int32")
        self.occupied = np.zeros(capacity, dtype=bool)
        self.weights = self.checkpoint.upload(device)
        self.states = {}
        self.batches = {}
        self.bundle = Path(bundle)
        self.device.driver.lib.cuMemsetD8_v2.argtypes = [
            ct.c_uint64,
            ct.c_ubyte,
            ct.c_size_t,
        ]
        self.device.driver.lib.cuMemsetD8_v2.restype = ct.c_int
        c = self.config
        for layer, kind in enumerate(c.layers):
            shapes = (
                (
                    (capacity, c.value_heads, c.value_dim, c.key_dim),
                    (capacity, c.qkv_width, 3),
                )
                if kind == "linear_attention"
                else ((capacity, c.kv_heads, context, c.head_dim),) * 2
            )
            dtype = "float32" if kind == "linear_attention" else "bfloat16"
            self.states[layer] = [device.empty(shape, dtype) for shape in shapes]

    def _check(self):
        if self.closed or self.generation != self.device._generation:
            raise RuntimeError("dense engine is closed or its device session expired")
        if get_ident() != self.thread:
            raise RuntimeError("dense engine must run on its owner thread")
        self.device._check()

    def admit(self):
        self._check()
        free = np.flatnonzero(~self.occupied)
        if not free.size:
            raise ValueError("request capacity exhausted")
        request = int(free[0])
        self.positions[request] = 0
        # Only recurrent state must be zeroed. KV beyond position is never read.
        for layer, kind in enumerate(self.config.layers):
            if kind == "linear_attention":
                for buffer in self.states[layer]:
                    size = buffer.nbytes // self.capacity
                    self.device.driver.call(
                        "cuMemsetD8_v2", buffer.pointer + request * size, 0, size
                    )
        self.occupied[request] = True
        return request

    def release(self, request):
        self._check()
        if not 0 <= request < self.capacity or not self.occupied[request]:
            raise ValueError("request is not owned")
        if request in self.pending:
            raise ValueError(
                "discard pending verification before releasing its request"
            )
        if hasattr(self, "deferred"):
            self.deferred.forget(request)
        self.occupied[request] = False
        self.positions[request] = 0

    def batch(self, count, chunk=1, *, verify=False):
        self._check()
        if not 1 <= count <= self.capacity:
            raise ValueError("invalid active count")
        slots = 1 << (count - 1).bit_length()
        key = (slots, chunk, verify)
        if key not in self.batches:
            path = (
                self.bundle
                / f'{"mtp-" if getattr(self,"draft",False) else ""}inference-s{slots}-t{chunk}{"-verify" if verify else ""}.json'
            )
            self.batches[key] = ExecutionBatch(self, path)
        return self.batches[key]

    def decode(self, requests, tokens):
        return self.batch(len(requests)).run(requests, [[token] for token in tokens])

    def prefill(self, requests, prompts, *, chunk=32):
        predicted = [None] * len(requests)
        for begin in range(0, max(map(len, prompts)), chunk):
            active = [i for i, prompt in enumerate(prompts) if begin < len(prompt)]
            for group in range(0, len(active), 32):
                indices = active[group : group + 32]
                batch = self.batch(len(indices), chunk)
                result = batch.run(
                    [requests[i] for i in indices],
                    [prompts[i][begin : begin + chunk] for i in indices],
                )
                for i, token in zip(indices, result):
                    predicted[i] = token
        return predicted

    def close(self):
        if self.closed:
            return
        self._check()
        for batch in self.batches.values():
            batch.close()
        if hasattr(self, "deferred"):
            self.deferred.close()
        for state in self.states.values():
            for b in state:
                b.release()
        for w in self.weights.values():
            w.release()
        self.closed = True
