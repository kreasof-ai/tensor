"""Choose a captured verification width while borrowing one resident model."""
import hashlib
from pathlib import Path
import numpy as np


def implementation_hashes():
    return {__name__:hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class VerifierPool:
    def __init__(self, contexts):
        self.contexts = dict(sorted(contexts.items()))
        if not self.contexts or any(key != value.chunk for key,value in self.contexts.items()):
            raise ValueError('verification widths must match their contexts')
        self.chunk = max(self.contexts)
        self.active = self.contexts[self.chunk]
        self.model = self.active.model
        self.device = self.model.device
        if any(context.model is not self.model or context.closed for context in self.contexts.values()):
            raise ValueError('verification contexts must borrow the same live model')
        self.closed = False
        self.window_calls = {key:0 for key in self.contexts}

    @property
    def buffers(self):
        return self.active.buffers

    @property
    def pending_verification(self):
        return any(context.pending_verification for context in self.contexts.values())

    @property
    def recompute_cache_bytes(self):
        return sum(getattr(context,'recompute_cache_bytes',0) for context in self.contexts.values())

    @property
    def snapshot_cache_bytes(self):
        return sum(buffer.nbytes for context in self.contexts.values()
                   for name,buffer in context.buffers.items() if name.startswith('saved_'))

    def forward(self, tokens, lengths=None, *, read_logits=False):
        self.model._check()
        if self.closed or self.pending_verification:
            raise RuntimeError('verification pool must be live and committed')
        tokens = np.asarray(tokens)
        if tokens.shape != (self.model.slots,self.chunk) or tokens.dtype.kind not in 'iu':
            raise ValueError('verification pool requires its maximum token width')
        if lengths is None: lengths = np.full(self.model.slots,self.chunk,'int32')
        lengths = np.asarray(lengths,dtype='int32')
        if lengths.shape != (self.model.slots,) or np.any(lengths<0) or np.any(lengths>self.chunk):
            raise ValueError('invalid pooled verification lengths')
        width = next(key for key in self.contexts if key >= int(lengths.max(initial=0)))
        self.active = self.contexts[width]
        result = self.active.forward(tokens[:,:width],lengths,read_logits=read_logits)
        self.window_calls[width] += 1
        if width == self.chunk: return result
        predictions, logits = result if read_logits else (result,None)
        padded = np.zeros(tokens.shape,'int32'); padded[:,:width] = predictions
        if not read_logits: return padded
        # Padding exists only for the diagnostic API. The serving path reads
        # predictions and borrows the selected context's actual hidden buffer.
        values = np.zeros((self.model.slots,self.chunk,self.model.config.vocab),logits.dtype)
        values[:,:width] = logits.reshape(self.model.slots,width,-1)
        return padded,values.reshape(-1,self.model.config.vocab)

    def commit(self, counts):
        if self.closed: raise RuntimeError('verification pool is closed')
        self.active.commit(counts)

    def close(self):
        if self.closed: return
        for context in self.contexts.values(): context.close()
        self.closed = True


class RepairPool:
    def __init__(self, verifier, contexts):
        self.verifier = verifier
        self.contexts = dict(contexts)
        if set(contexts) != set(verifier.contexts): raise ValueError('verification and repair widths differ')
        self.chunk = verifier.chunk
        self.model = contexts[self.chunk].model
        if any(context.model is not self.model or context.chunk != key or context.closed
               for key,context in contexts.items()):
            raise ValueError('repair contexts must borrow the same live MTP model')
        self.closed = False

    def forward(self, tokens, hidden, lengths):
        if self.closed or self.verifier.closed or self.verifier.pending_verification:
            raise RuntimeError('repair needs a live committed verifier')
        if np.shape(tokens) != (self.model.slots,self.chunk):
            raise ValueError('repair pool requires its maximum token width')
        width = self.verifier.active.chunk
        if np.any(np.asarray(lengths)>width): raise ValueError('repair lengths exceed the selected verifier')
        return self.contexts[width].forward(tokens[:,:width],hidden,lengths)

    def close(self):
        if self.closed: return
        for context in self.contexts.values(): context.close()
        self.closed = True
