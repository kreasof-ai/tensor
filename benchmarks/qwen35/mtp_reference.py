"""Independent CPU evaluation of the official one-layer MTP branch."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import torch

from tensor_llm.qwen35.checkpoint import Qwen35Checkpoint
from .reference import Reference


class QuantizedKV(list):
    def append(self, x):
        blocks = x.float().reshape(*x.shape[:-1], 2, 128)
        scales = blocks.abs().amax(-1, keepdim=True).clamp_min(1e-12)/448
        value = ((blocks/scales).to(torch.float8_e4m3fn).float()*scales).bfloat16().reshape(x.shape)
        super().append(value)


class MTPReference(Reference):
    def __init__(self, path, slots, *, kv_dtype='fp8'):
        torch.set_num_threads(2)
        draft = Qwen35Checkpoint(path, branch='mtp')
        target = Qwen35Checkpoint(path)
        mapping = {name: (draft, name) for name in draft.tensors}
        for name in draft.tensors:
            alias = name.replace('mtp.layers.0.', 'model.language_model.layers.0.')
            if name == 'mtp.norm.weight': alias = 'model.language_model.norm.weight'
            mapping[alias] = (draft, name)
        for name in ('model.language_model.embed_tokens.weight', 'lm_head.weight'):
            mapping[name] = (target, name)
        self.checkpoint = SimpleNamespace(
            tensors={n: owner.tensors[original] for n, (owner, original) in mapping.items()},
            read=lambda name: mapping[name][0].read(mapping[name][1]))
        self.config = replace(target.config, layers=('full_attention',))
        self.slots = slots
        self.position = np.zeros(slots, dtype='int32')
        self.states, self.histories = {}, {}
        if kv_dtype not in ('fp8', 'bfloat16'): raise ValueError('unsupported KV precision')
        make = QuantizedKV if kv_dtype == 'fp8' else list
        self.cache = {0: [(make(), make()) for _ in range(slots)]}

    def forward(self, tokens, hidden):
        tokens = np.asarray(tokens, dtype='int32')
        active = torch.from_numpy(tokens >= 0)
        info = self.checkpoint.tensors['model.language_model.embed_tokens.weight']
        rows = []
        with info.shard.open('rb') as stream:
            for token in tokens:
                stream.seek(info.offset+max(int(token), 0)*2048*2)
                rows.append(np.frombuffer(stream.read(2048*2), dtype='uint16').copy())
        embedding = torch.from_numpy(np.stack(rows)).view(torch.bfloat16)
        embedding[~active] = 0
        hidden = torch.from_numpy(np.asarray(hidden, dtype='float32')).bfloat16()
        joined = torch.cat((self.norm('mtp.pre_fc_norm_embedding', embedding),
                            self.norm('mtp.pre_fc_norm_hidden', hidden)), dim=-1)
        residual = self.linear('mtp.fc', joined).bfloat16()
        root = 'model.language_model.layers.0.'
        mixed = self.attention(0, self.norm(root+'input_layernorm', residual), active)
        residual = (residual+mixed.bfloat16()).bfloat16()
        ffn = self.mlp(0, self.norm(root+'post_attention_layernorm', residual))
        residual = (residual+ffn.bfloat16()).bfloat16()
        normal = self.norm('model.language_model.norm', residual)
        logits = self.linear('lm_head', normal)
        self.position += active.numpy().astype('int32')
        return logits.numpy()


if __name__ == '__main__':
    import argparse, json
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--samples', type=Path, required=True)
    a = p.parse_args()
    samples = np.load(a.samples/'native.npz')
    ref = MTPReference(a.checkpoint, samples['tokens'].shape[1])
    records = []
    for step, (tokens, hidden, actual) in enumerate(zip(samples['tokens'], samples['hidden'], samples['logits'])):
        expected = ref.forward(tokens, hidden)
        active = tokens >= 0
        error = float(np.linalg.norm(actual[active]-expected[active])/np.linalg.norm(expected[active]))
        records.append(dict(step=step, relative_logit_rms=error, finite=bool(np.isfinite(actual).all()),
            greedy_matches=int((actual.argmax(-1)[active] == expected.argmax(-1)[active]).sum()),
            active_slots=int(active.sum()), passes_3_percent_gate=error <= .03))
        print('MTP CPU reference', records[-1], flush=True)
    (a.samples/'reference-report.json').write_text(json.dumps(dict(records=records,
        status='passed' if all(r['passes_3_percent_gate'] and r['finite'] for r in records) else 'failed',
        scope='one MTP layer on identical supplied hidden states; target model quality is separate',
        speculative_verification=False, full_stress_target_reached=False), indent=2)+'\n')
