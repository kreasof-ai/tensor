"""Verifier that stores recurrent inputs and replays only rejected prefixes."""
import hashlib
import json
from pathlib import Path

from tensor.providers.cuda_graph import CudaGraph
from ..artifacts import identity
from ..prefill import Qwen35Prefill
from .verifier import Qwen35Verifier


def implementation_hashes():
    from ..kernels import recompute
    return {recompute.__name__: hashlib.sha256(Path(recompute.__file__).read_bytes()).hexdigest(),
            __name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


class Qwen35RecomputeVerifier(Qwen35Verifier):
    """Keep the frozen scan arithmetic without storing each speculative state."""

    def __init__(self, model, bundle):
        manifest = json.loads((Path(bundle)/'prefill.json').read_text())
        if (manifest.get('state_recompute') != 'accepted-prefix-recompute'
                or manifest.get('state_recompute_implementation') != implementation_hashes()
                or manifest.get('chunk') not in (64,128)):
            raise ValueError('recompute source or window mismatch; rebuild the bundle')
        super().__init__(model, bundle)

    def _allocate(self):
        Qwen35Prefill._allocate(self)
        d, b, s, r = self.device, self.buffers, self.model.slots, self.rows
        b['verify_logits'] = d.empty((r, self.model.config.vocab))
        b['predictions'] = d.empty((r,), 'int32')
        b['accepted'] = d.empty((s,), 'int32')
        self.recompute_cache = {}
        for layer, kind in enumerate(self.model.config.layers):
            if kind != 'linear_attention': continue
            state, history = self.model.states[layer]
            shapes = dict(initial=state.shape, initial_history=history.shape,
                          q=(r,32,128), k=(r,32,128), v=(r,32,128),
                          g=(r,32), beta=(r,32), x=(r,8192), replay_lengths=(s,))
            cache = {}
            for name, shape in shapes.items():
                dtype = 'bfloat16' if name == 'x' else 'int32' if name == 'replay_lengths' else 'float32'
                cache[name] = b[f'_recompute_{layer}_{name}'] = d.empty(shape, dtype)
            self.recompute_cache[layer] = cache
        self.recompute_cache_bytes = sum(x.nbytes for cache in self.recompute_cache.values() for x in cache.values())

    def _plan(self):
        # The ordinary scan and convolution already match the frozen verifier.
        Qwen35Prefill._plan(self)
        s, b = self.model.slots, self.buffers
        p = dict(slots=s, chunk=self.chunk)
        conv = self.kernels[identity('gdn_conv', p)]
        scan = self.kernels[identity('gdn_scan', p)]
        layers = {id(state): layer for layer, states in self.model.states.items()
                  if self.model.config.layers[layer] == 'linear_attention' for state in states}
        plan, seen = [], {'gdn_conv': set(), 'gdn_scan': set()}
        for kernel, bound in self.plan[:-4]:
            if kernel is conv or kernel is scan:
                values = dict(zip((a['name'] for a in kernel.manifest.get('abi', kernel.manifest['arguments'])), bound.storage))
                layer = layers[id(values['state'])]
                cache = self.recompute_cache[layer]
                if kernel is conv:
                    plan.append(self._bind('recompute_save_conv', p, x=values['x'], state=values['state'],
                                           saved_x=cache['x'], initial=cache['initial_history']))
                    seen['gdn_conv'].add(layer)
                else:
                    plan.append(self._bind('recompute_save_scan', p, state=values['state'],
                        query=values['q'], key=values['k'], value=values['v'], g=values['g'], beta=values['beta'],
                        initial=cache['initial'], saved_q=cache['q'], saved_k=cache['k'], saved_v=cache['v'],
                        saved_g=cache['g'], saved_beta=cache['beta']))
                    seen['gdn_scan'].add(layer)
            plan.append((kernel, bound))
        expected = set(self.recompute_cache)
        if any(value != expected for value in seen.values()):
            raise ValueError('recompute plan does not cover each recurrent layer')
        plan.append(self._bind('verify_head', dict(r=self.rows,k=2048,o=self.model.config.vocab),
                              x=b['normal'], w=self.model.weights['lm_head.weight'], out=b['verify_logits']))
        plan.append(self._bind('argmax', dict(r=self.rows,vocab=self.model.config.vocab),
            logits=b['verify_logits'], tokens=b['predictions'], positions=b['flat_positions'], active=b['flat_active']))
        self.plan = plan
        self.commit_plan = []
        for layer, cache in self.recompute_cache.items():
            state, history = self.model.states[layer]
            self.commit_plan.append(self._bind('recompute_restore_scan', p, initial=cache['initial'],
                accepted=b['accepted'], lengths=b['lengths'], state=state, replay_lengths=cache['replay_lengths']))
            self.commit_plan.append(self._bind('gdn_scan', p, q=cache['q'], k=cache['k'], v=cache['v'],
                g=cache['g'], beta=cache['beta'], lengths=cache['replay_lengths'], state=state, out=b['gdn_out']))
            self.commit_plan.append(self._bind('recompute_restore_conv', p, initial=cache['initial_history'],
                saved_x=cache['x'], accepted=b['accepted'], lengths=b['lengths'], state=history))
        self.commit_plan.append(self._bind('last_rows', dict(slots=s,chunk=self.chunk,width=2048),
                                          x=b['normal'], lengths=b['accepted'], out=self.model.buffers['normal']))
        self.commit_graph = CudaGraph(self.device, self._submit_commit,
            resources=(*b.values(), *self.kernels.values(),
                       *(v for states in self.model.states.values() for v in states), self.model.buffers['normal']))
