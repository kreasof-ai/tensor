"""Reproduce full-prefix MTP agreement diagnostics; not an accelerated stress run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import tensor

from tensor_llm import Qwen35Batch, Qwen35MTP, Qwen35Prefill
from tensor_llm.qwen35.mtp.prefill import Qwen35MTPPrefill
from .mtp_calibrate import calibrate_long
from .mtp_chain import measure


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'decoder', 'draft', 'prefill', 'draft-prefill', 'workload', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=256)
    p.add_argument('--chain-rounds', type=int, default=64)
    a = p.parse_args()
    workload = json.loads(a.workload.read_text())
    tokens = np.array([r['prompt_token_ids'] for r in workload['requests']], dtype='int32').T
    a.out.mkdir(parents=True, exist_ok=True)
    provenance = dict(workload_sha256=hashlib.sha256(a.workload.read_bytes()).hexdigest(),
        bundles={name: dict(path=str(bundle.resolve()), manifest_sha256=hashlib.sha256((bundle/file).read_bytes()).hexdigest())
                 for name, bundle, file in (('decoder', a.decoder, 'inference.json'),
                    ('draft', a.draft, 'inference.json'), ('prefill', a.prefill, 'prefill.json'),
                    ('draft_prefill', a.draft_prefill, 'prefill.json'))},
        full_stress_target_reached=False, speculative_verification=False)
    (a.out/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    with tensor.Device() as d, Qwen35Batch(a.checkpoint, a.decoder, d,
            progress=lambda s: print(s, flush=True)) as target:
        with Qwen35MTP(target, a.draft) as draft:
            prefill = Qwen35Prefill(target, a.prefill)
            try:
                draft_prefill = Qwen35MTPPrefill(draft, a.draft_prefill)
                try:
                    calibrate_long(target, draft, prefill, draft_prefill, tokens, a.out/'long-prefix', decode_steps=a.steps)
                finally:
                    draft_prefill.close()
            finally:
                prefill.close()
            measure(target, draft, a.out/'chain', proposals=3, rounds=a.chain_rounds)


if __name__ == '__main__': main()
