"""Experimental native MTP drafter sharing the target's embedding and head.

This executes the official draft layer, not a speculative acceptance engine.
Callers must initialize its shifted-token/target-hidden cache over the prefix.
Target verification and recurrent rollback remain separate required work.
"""
from dataclasses import asdict, replace as config_replace
import ctypes as ct
import hashlib
import json
from pathlib import Path
from threading import get_ident

import numpy as np

from tensor.providers.cuda_graph import CudaGraph
from tensor.runtime.abi import BoundCall
from ..checkpoint import Qwen35Checkpoint
from ..decode import Qwen35Batch, implementation_hashes
from ..artifacts import identity


def mtp_implementation_hashes():
    from ..kernels import mtp
    result = implementation_hashes()
    for name, path in ((__name__, __file__), (mtp.__name__, mtp.__file__)):
        result[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


class Qwen35MTP(Qwen35Batch):
    """One native draft layer with private KV; target weights remain borrowed."""

    def __init__(self, target, bundle, *, graphs=True):
        target._check()
        self.target = target
        self.device, self.thread = target.device, get_ident()
        self._generation = self.device._generation
        self.closed = False
        self.weights, self.kernels, self.buffers, self.states = {}, {}, {}, {}
        self.owned_weights = {}
        self.graph = None
        self.prefills = set()
        self.checkpoint = Qwen35Checkpoint(target.checkpoint.directory, branch='mtp')
        self.config = config_replace(target.config, layers=('full_attention',))
        self.slots, self.context, self.splits = target.slots, target.context, target.splits
        self.kv_dtype = target.kv_dtype
        self.position = np.zeros(self.slots, dtype='int32')
        self.active = np.zeros(self.slots, dtype='int32')
        self.bundle = Path(bundle).resolve()
        value = json.loads((self.bundle/'inference.json').read_text())
        if (value.get('schema') != 'tensor.qwen35-batch.v1' or value.get('draft') != 'qwen35-mtp'
                or value.get('implementation') != mtp_implementation_hashes()
                or value.get('config') != json.loads(json.dumps(asdict(self.config)))
                or value.get('target_config') != json.loads(json.dumps(asdict(target.config)))
                or value.get('target') != self.device.info['arch']
                or (value.get('slots'), value.get('context'), value.get('splits'), value.get('kv_dtype'))
                    != (self.slots, self.context, self.splits, self.kv_dtype)):
            raise ValueError('MTP bundle source, configuration or target mismatch; rebuild it')
        try:
            for key, row in value['kernels'].items():
                descriptor = row.get('logical', row)
                path = (self.bundle/descriptor['path']).resolve()
                if (not path.is_relative_to(self.bundle)
                        or hashlib.sha256(path.read_bytes()).hexdigest() != descriptor['sha256']):
                    raise ValueError('MTP artifact checksum mismatch')
                self.kernels[key] = self.device.load(path)
            self.owned_weights = self.checkpoint.upload(self.device, pack_experts=True)
            for name, buffer in self.owned_weights.items():
                alias = name.replace('mtp.layers.0.', 'model.language_model.layers.0.')
                if name == 'mtp.norm.weight': alias = 'model.language_model.norm.weight'
                self.weights[alias] = buffer
            for name in ('model.language_model.embed_tokens.weight', 'lm_head.weight'):
                self.weights[name] = target.weights[name]
            self._allocate()
            self.buffers['mtp_hidden'] = self.device.empty((self.slots, 2048), 'bfloat16')
            self.buffers['mtp_join'] = self.device.empty((self.slots, 4096), 'bfloat16')
            self.buffers['mtp_fc'] = self.device.empty((self.slots, 2048), 'float32')
            base = Qwen35Batch._plan(self)
            prefix = []
            def bind(kind, p, *args):
                kernel = self.kernels[identity(kind, p)]
                values, symbols, launch = kernel._bind(args, {}, include_outputs=True)
                prefix.append((kernel, BoundCall(self.device, kernel.manifest,
                    values, symbols, launch, validated=True), None))
            b = self.buffers
            bind('mtp_join', dict(r=self.slots, c=2048, eps=self.config.epsilon),
                 b['residual'], b['mtp_hidden'], self.weights['mtp.pre_fc_norm_embedding.weight'],
                 self.weights['mtp.pre_fc_norm_hidden.weight'], b['mtp_join'])
            bind('bf16_linear', dict(r=self.slots, k=4096, o=2048),
                 b['mtp_join'], self.weights['mtp.fc.weight'], b['mtp_fc'])
            bind('mtp_cast', dict(r=self.slots, c=2048), b['mtp_fc'], b['residual'])
            self.plan = [base[0], *prefix, *base[1:]]
            from ..pipeline import replace
            replace(self, self.bundle)
            if graphs:
                self.graph = CudaGraph(self.device, self._submit,
                    resources=(*self.weights.values(), *self.buffers.values(),
                               *(b for state in self.states.values() for b in state), *self.kernels.values()))
            target.prefills.add(self)
        except BaseException:
            self.close()
            raise

    def _check(self):
        super()._check()
        self.target._check()

    def draft(self, tokens, hidden, *, read_logits=False):
        """Advance consecutive MTP positions using next-token IDs and target states.

        ``hidden`` is the target's final normalized BF16 hidden state for the
        preceding token. Pass a device buffer to avoid a GPU/CPU round trip.
        Prefix initialization begins at position zero; skipping directly to a
        long context would leave unreadable entries in the draft KV cache.
        """
        self._check()
        dest = self.buffers['mtp_hidden']
        if (getattr(hidden, 'shape', None) != dest.shape
                or str(getattr(hidden, 'dtype', None)) != 'bfloat16'
                or getattr(hidden, 'device', None) is not self.device):
            raise ValueError('MTP hidden state must be a BF16 buffer on the target device')
        hidden._check()
        self.device.driver.call('cuMemcpyDtoD_v2', dest.pointer, hidden.pointer, dest.nbytes)
        self.device.driver.call('cuStreamSynchronize', None)
        return super().forward(tokens, read_logits=read_logits)

    @property
    def allocated_bytes(self):
        return sum(b.nbytes for b in (*self.owned_weights.values(), *self.buffers.values(),
                   *(b for state in self.states.values() for b in state)))

    def close(self):
        if self.closed: return
        self.target.prefills.discard(self)
        # Base teardown only owns the MTP weights. Embedding and head belong to
        # the target and must remain usable after the drafter is closed.
        self.weights = self.owned_weights
        super().close()
        self.owned_weights = {}
