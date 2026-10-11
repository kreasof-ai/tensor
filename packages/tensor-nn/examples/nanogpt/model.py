"""nanoGPT assembled from native Tensor modules, with an explicit training step."""

import ctypes as ct
import time
import numpy as np
from tensor_nn_examples.nanogpt.bundle import TrainingBundle
from tensor_nn.module import Module
from tensor_nn.parameter import Parameter
from tensor_nn.optim import AdamW
from tensor_nn.templates import TokenPositionEmbedding, LayerNorm, Linear, CrossEntropy
from tensor_nn_examples.nanogpt.config import initial_weights
from tensor_nn_examples.nanogpt.block import TransformerBlock


class NanoGPT(Module):
    """Static, bias-free GPT with FP16 compute and tied token/classifier weights."""

    def __init__(self, directory, device, *, seed=20260930):
        started = time.perf_counter()
        previous_buffers = set(device._buffers)
        super().__init__(TrainingBundle(directory, device), name="nanogpt")
        self.config = self.library.config
        self.initial = initial_weights(self.config, seed)
        for name, array in self.initial.items():
            self.register_parameter(name, Parameter.from_numpy(device, array))
        cfg = self.config
        self.embedding = self.register_module(
            "embedding",
            TokenPositionEmbedding(
                self.library,
                self.parameters["token"],
                self.parameters["position"],
                batch=cfg.batch,
                sequence=cfg.sequence,
                width=cfg.width,
                vocab=cfg.vocab,
            ),
        )
        self.blocks = []
        for layer in range(cfg.layers):
            self.blocks.append(
                self.register_module(
                    f"block_{layer}",
                    TransformerBlock(self.library, cfg, self.parameters, layer=layer),
                )
            )
        self.final_norm = self.register_module(
            "final_norm",
            LayerNorm(self.library, cfg.rows, cfg.width, self.parameters["ln_f"]),
        )
        self.head = self.register_module(
            "head", Linear(self.library, cfg.rows, self.parameters["token"])
        )
        self.criterion = self.register_module(
            "criterion", CrossEntropy(self.library, rows=cfg.rows, vocab=cfg.vocab)
        )
        self.optimizer = AdamW(
            self.library,
            self.parameters,
            loss_scale=cfg.loss_scale,
            learning_rate=cfg.learning_rate,
            clip=cfg.clip,
        )
        self.tokens = device.empty((cfg.rows,), "int32")
        self.targets = device.empty((cfg.rows,), "int32")
        self.loss_grad = device.full((cfg.rows,), cfg.loss_scale / cfg.rows)
        self.loss = self.criterion.output
        self.clip = self.optimizer.clip
        self.context = None
        self.bind(self._forward, self._backward)
        self.allocated_bytes = sum(
            b.nbytes for b in device._buffers if b not in previous_buffers
        )
        self.prepare_seconds = time.perf_counter() - started

    @property
    def steps(self):
        return self.optimizer.steps

    def _forward(self, ctx, tokens):
        tape = []

        def call(module, *inputs):
            output, context = module(*inputs)
            tape.append((module, context))
            return output

        x = call(self.embedding, tokens)
        for block in self.blocks:
            x = call(block, x)
        x = call(self.final_norm, x)
        logits = call(self.head, x)
        ctx.metadata["tape"] = tape
        return logits

    def _backward(self, ctx, gradient):
        for module, context in reversed(ctx.metadata["tape"]):
            gradient = module.backward(context, gradient)[0]
        return None

    def forward_backward(self):
        logits, self.context = self(self.tokens)
        _, loss_context = self.criterion(logits, self.targets)
        gradient, _ = self.criterion.backward(loss_context, self.loss_grad)
        self.backward(self.context, gradient)
        self.context = None

    def update(self):
        self.optimizer.step()

    def step(self):
        self.forward_backward()
        self.update()

    def upload(self, buffer, value):
        value = np.ascontiguousarray(value, dtype=buffer.dtype).reshape(buffer.shape)
        buffer._check()
        self.device.synchronize()
        self.device.driver.call(
            "cuMemcpyHtoD_v2",
            buffer.pointer,
            ct.c_void_p(value.ctypes.data),
            value.nbytes,
        )
        self.device.driver.call("cuStreamSynchronize", None)

    def set_batch(self, tokens, targets):
        cfg = self.config
        for data in (tokens, targets):
            array = np.asarray(data)
            if array.shape != (cfg.batch, cfg.sequence) or not np.issubdtype(
                array.dtype, np.integer
            ):
                raise ValueError("token batches must match the training configuration")
            if np.any(array < 0) or np.any(array >= cfg.vocab):
                raise ValueError("token ids must be within the vocabulary")
        self.upload(self.tokens, tokens)
        self.upload(self.targets, targets)

    def reset(self):
        self.discard_contexts()
        for name, param in self.parameters.items():
            initial = self.initial[name].reshape(-1)
            self.upload(param.weight, initial)
            self.upload(param.half, initial)
            self.upload(param.moment, np.zeros_like(initial))
            self.upload(param.variance, np.zeros_like(initial))
        self.context = None
        self.optimizer.steps = 0

    def read_loss(self):
        return self.criterion.read()

    def read_norm(self):
        return self.optimizer.read_norm()
