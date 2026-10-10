"""Official dense MTP branch sharing the target's embedding and head."""

from dataclasses import replace
from threading import get_ident

import numpy as np

from .engine import DenseEngine


class DenseMTP(DenseEngine):
    def __init__(self, target):
        target._check()
        self.target = target
        self.draft = True
        self.device = target.device
        self.thread = get_ident()
        self.fused_head = target.fused_head
        self.implementation = target.implementation
        self.capacity = target.capacity
        self.context = target.context
        self.checkpoint = target.checkpoint
        self.config = replace(target.config, layers=("full_attention",))
        self.bundle = target.bundle
        self.batches = {}
        self.pending = {}
        self.closed = False
        self.generation = self.device._generation
        self.positions = np.zeros(self.capacity, dtype="int32")
        self.occupied = np.zeros(self.capacity, dtype=bool)
        self.owned_weights = self.checkpoint.upload(self.device, branch="mtp")
        self.weights = {}
        for name, buffer in self.owned_weights.items():
            alias = name.replace("mtp.layers.0.", "model.language_model.layers.0.")
            if name == "mtp.norm.weight":
                alias = "model.language_model.norm.weight"
            self.weights[alias] = buffer
        self.weights["model.language_model.embed_tokens.weight"] = target.weights[
            "model.language_model.embed_tokens.weight"
        ]
        c = self.config
        self.states = {
            0: [
                self.device.empty(
                    (self.capacity, c.kv_heads, self.context, c.head_dim), "bfloat16"
                )
                for _ in range(2)
            ]
        }
        self.hidden = self.device.empty((self.capacity, c.width), "bfloat16")

    def _check(self):
        self.target._check()
        super()._check()

    def admit_at(self, slot):
        self._check()
        if self.occupied[slot]:
            raise ValueError("MTP cache slot already owned")
        self.occupied[slot] = True
        self.positions[slot] = 0

    def close(self):
        if self.closed:
            return
        self._check()
        for batch in self.batches.values():
            batch.close()
        for state in self.states.values():
            for buffer in state:
                buffer.release()
        self.hidden.release()
        for weight in self.owned_weights.values():
            weight.release()
        self.closed = True
