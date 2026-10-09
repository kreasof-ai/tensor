"""Native chunked initialization of the MTP shifted-token/target-hidden cache."""
import hashlib
import json
from pathlib import Path

from tensor.runtime.abi import BoundCall
from ..artifacts import identity
from .decode import mtp_implementation_hashes
from ..prefill import Qwen35Prefill


def implementation_hashes():
    from .. import prefill
    result = mtp_implementation_hashes()
    for name, path in ((__name__, __file__), (prefill.__name__, prefill.__file__)):
        result[name] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return result


class Qwen35MTPPrefill(Qwen35Prefill):
    def __init__(self, mtp, bundle):
        value = json.loads((Path(bundle)/'prefill.json').read_text())
        if (value.get('draft') != 'qwen35-mtp' or value.get('implementation') != implementation_hashes()):
            raise ValueError('MTP prefill source mismatch; rebuild its bundle')
        super().__init__(mtp, bundle)

    def _plan(self):
        super()._plan()
        extra = []
        def call(kind, p, **bindings):
            kernel = self.kernels[identity(kind, p)]
            args = tuple(bindings[a['name']] for a in kernel.manifest['arguments'])
            values, symbols, launch = kernel._bind(args, {}, include_outputs=True)
            extra.append((kernel, BoundCall(self.device, kernel.manifest, values, symbols, launch, validated=True)))
        b, w = self.buffers, self.model.weights
        call('mtp_join', dict(r=self.rows, c=2048, eps=self.model.config.epsilon),
             embedding=b['residual'], hidden=b['mtp_hidden'],
             embedding_weight=w['mtp.pre_fc_norm_embedding.weight'],
             hidden_weight=w['mtp.pre_fc_norm_hidden.weight'], out=b['mtp_join'])
        call('bf16_linear', dict(r=self.rows, k=4096, o=2048),
             x=b['mtp_join'], w=w['mtp.fc.weight'], out=b['mtp_fc'])
        call('mtp_cast', dict(r=self.rows, c=2048), x=b['mtp_fc'], out=b['residual'])
        # Controls and token embedding precede the MTP join; the ordinary
        # one-layer attention/MoE path then consumes the fused BF16 residual.
        self.plan[2:2] = extra

    def forward(self, tokens, hidden, lengths=None, *, read_logits=False):
        self.model._check()
        if self.closed: raise RuntimeError('MTP prefill is closed')
        dest = self.buffers['mtp_hidden']
        if (getattr(hidden, 'shape', None) != dest.shape
                or str(getattr(hidden, 'dtype', None)) != 'bfloat16'
                or getattr(hidden, 'device', None) is not self.device):
            raise ValueError('MTP prefill needs matching flat BF16 target hidden states')
        hidden._check()
        self.device.driver.call('cuMemcpyDtoD_v2', dest.pointer, hidden.pointer, dest.nbytes)
        self.device.driver.call('cuStreamSynchronize', None)
        return super().forward(tokens, lengths, read_logits=read_logits)
