"""Batched greedy verification with exact accepted-prefix recurrent snapshots.

KV rows beyond the committed prefix are overwritten on the next pass. GDN
matrices and convolution histories instead require explicit prefix restoration.
"""
import ctypes as ct
import hashlib
import json
from pathlib import Path
import numpy as np
from tensor.runtime.abi import BoundCall
from tensor.providers.cuda_graph import CudaGraph
from ..artifacts import identity
from ..prefill import Qwen35Prefill


def implementation_hashes():
    from .. import prefill
    from ..kernels import speculative
    return {m.__name__: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
            for m in (prefill, speculative)} | {
                __name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class Qwen35Verifier(Qwen35Prefill):
    def __init__(self, model, bundle):
        self.commit_graph = None; self.pending_verification = False
        manifest = json.loads((Path(bundle)/'prefill.json').read_text())
        if (manifest.get('verification') != 'greedy-prefix-snapshots'
                or manifest.get('implementation') != implementation_hashes()):
            raise ValueError('verification source mismatch; rebuild the bundle')
        super().__init__(model, bundle)

    def _allocate(self):
        super()._allocate()
        self.buffers['verify_logits'] = self.device.empty((self.rows, self.model.config.vocab))
        self.buffers['predictions'] = self.device.empty((self.rows,), 'int32')
        self.buffers['accepted'] = self.device.empty((self.model.slots,), 'int32')
        for layer, kind in enumerate(self.model.config.layers):
            if kind != 'linear_attention': continue
            for i, state in enumerate(self.model.states[layer]):
                self.buffers[f'saved_{layer}_{i}'] = self.device.empty(
                    (self.model.slots, self.chunk-1, *state.shape[1:]), 'float32')

    def _bind(self, kind, p, **values):
        kernel = self.kernels[identity(kind, p)]
        args = tuple(values[a['name']] for a in kernel.manifest['arguments'])
        values, symbols, launch = kernel._bind(args, {}, include_outputs=True)
        return kernel, BoundCall(self.device, kernel.manifest, values, symbols, launch, validated=True)

    def _plan(self):
        super()._plan()
        reverse = {id(kernel): key for key, kernel in self.kernels.items()}
        gdn = {identity(kind, dict(slots=self.model.slots, chunk=self.chunk)): kind
               for kind in ('gdn_scan', 'gdn_conv')}
        layer_ids = iter(i for i,k in enumerate(self.model.config.layers) if k == 'linear_attention')
        current_layer = None
        for index, (kernel, bound) in enumerate(self.plan):
            kind = gdn.get(reverse.get(id(kernel)))
            if kind is None: continue
            if kind == 'gdn_conv': current_layer = next(layer_ids)
            values = dict(zip((a['name'] for a in kernel.manifest.get('abi', kernel.manifest['arguments'])), bound.storage))
            values['checkpoints'] = self.buffers[f'saved_{current_layer}_{0 if kind == "gdn_scan" else 1}']
            self.plan[index] = self._bind('spec_'+kind,
                dict(slots=self.model.slots, chunk=self.chunk), **values)
        # The ordinary prefill only evaluates the final row. Verification needs
        # greedy predictions for every input row and leaves positions uncommitted.
        self.plan = self.plan[:-4]
        b = self.buffers; s = self.model.slots
        self.plan.append(self._bind('verify_head', dict(r=self.rows, k=2048, o=self.model.config.vocab),
            x=b['normal'], w=self.model.weights['lm_head.weight'], out=b['verify_logits']))
        self.plan.append(self._bind('argmax', dict(r=self.rows, vocab=self.model.config.vocab),
            logits=b['verify_logits'], tokens=b['predictions'], positions=b['flat_positions'], active=b['flat_active']))
        self.commit_plan = []
        for layer,kind in enumerate(self.model.config.layers):
            if kind != 'linear_attention': continue
            for i,state in enumerate(self.model.states[layer]):
                self.commit_plan.append(self._bind('restore', dict(slots=s, chunk=self.chunk, shape=list(state.shape[1:])),
                    checkpoints=b[f'saved_{layer}_{i}'], state=state, accepted=b['accepted'], lengths=b['lengths']))
        self.commit_plan.append(self._bind('last_rows', dict(slots=s, chunk=self.chunk, width=2048),
            x=b['normal'], lengths=b['accepted'], out=self.model.buffers['normal']))
        self.commit_graph = CudaGraph(self.device, self._submit_commit,
            resources=(*self.buffers.values(), *self.kernels.values(),
                       *(v for state in self.model.states.values() for v in state), self.model.buffers['normal']))

    def _submit_commit(self):
        for kernel, bound in self.commit_plan: self.device._launch(kernel, bound)

    def forward(self, tokens, lengths=None, *, read_logits=False):
        if self.pending_verification: raise RuntimeError('commit the previous verification first')
        # Reuse all ordinary input validation and uploads. Its host position
        # update is undone: this graph does not change persistent positions.
        before = self.model.position.copy()
        super().forward(tokens, lengths)
        self.model.position = before
        self.pending_verification = True
        result = self.buffers['predictions'].to_numpy().reshape(self.model.slots, self.chunk)
        return (result, self.buffers['verify_logits'].to_numpy()) if read_logits else result

    def commit(self, counts):
        if self.closed or not self.pending_verification: raise RuntimeError('no verification to commit')
        counts = np.asarray(counts)
        lengths = self.buffers['lengths'].to_numpy()
        if (counts.shape != lengths.shape or counts.dtype.kind not in 'iu'
                or np.any(counts < (lengths > 0)) or np.any(counts > lengths)):
            raise ValueError('accepted input counts outside the verified prefix')
        value = np.ascontiguousarray(counts, dtype='int32')
        dest = self.buffers['accepted']
        self.device.driver.call('cuMemcpyHtoD_v2', dest.pointer, ct.c_void_p(value.ctypes.data), value.nbytes)
        self.device.driver.call('cuStreamSynchronize', None)
        self.commit_graph.launch()
        self.model.position += value
        self.model._write('positions', self.model.position)
        self.pending_verification = False

    def close(self):
        if self.commit_graph: self.commit_graph.close(); self.commit_graph = None
        super().close()
