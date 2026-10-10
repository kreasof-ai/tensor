"""Full-model verification and rollback against a frozen snapshot verifier."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .spec_run import install_selected, make_verifier
from .whole_tune import Snapshot
from .prefill_comparison import cache_prefix_hashes


def run(model, control, selected, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    prefix = Snapshot(model)
    chunk=json.loads((Path(selected)/'prefill.json').read_text())['chunk']
    lengths = np.array([chunk,chunk-1,chunk//2+5,0,1,3,chunk,3*chunk//4], 'int32')
    if model.slots != len(lengths): raise ValueError('recompute qualification requires C8')
    tokens = np.repeat(prefix.tokens[:,None], chunk, axis=1)
    valid = (np.arange(chunk)[None,:] < lengths[:,None]).reshape(-1)
    depths = (1,2,3,17,chunk//2+5,chunk-1,chunk)
    # Hash large outputs so the two executors never occupy GPU memory together.
    def digest(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    def observe(bundle):
        prefix.restore(model)
        verifier = make_verifier(model,bundle)
        try:
            install_selected(verifier,bundle)
            predictions,logits = verifier.forward(tokens,lengths,read_logits=True)
            # Inactive rows have no defined expert output. Compare every actual
            # verified input, including shortened slots, without hashing tails.
            finite = bool(np.isfinite(logits[valid]).all())
            forward = dict(predictions=digest(predictions.reshape(-1)[valid]),logits=digest(logits[valid]),
                           states=[digest(x.to_numpy()) for layer,states in model.states.items()
                                   if model.config.layers[layer]=='linear_attention' for x in states])
            del logits
            # Each commit must begin with a new verification from the real prefix.
            commits = []
            for depth in depths:
                if not verifier.pending_verification:
                    prefix.restore(model); verifier.forward(tokens,lengths)
                counts = np.minimum(lengths,depth).astype('int32')
                verifier.commit(counts)
                commits.append(dict(depth=depth,positions=model.position.tolist(),
                    states=[digest(x.to_numpy()) for layer,states in model.states.items()
                            if model.config.layers[layer]=='linear_attention' for x in states],
                    normal=digest(model.buffers['normal'].to_numpy()),
                    kv_prefix=cache_prefix_hashes(model)))
            return dict(finite=finite,forward=forward,commits=commits,
                        cache_bytes=getattr(verifier,'recompute_cache_bytes',0))
        finally:
            verifier.close(); prefix.restore(model)
    expected=observe(control); actual=observe(selected)
    report=dict(scope='frozen verifier vs accepted-prefix replay at the real 32K prefix',
        prefix_positions=prefix.position.tolist(),lengths=lengths.tolist(),rejection_depths=list(depths),
        forward_bitwise_equal=expected['forward']==actual['forward'],
        forward_component_equal={key:expected['forward'][key]==actual['forward'][key]
                                 for key in expected['forward']},
        commits_bitwise_equal=expected['commits']==actual['commits'],
        finite=expected['finite'] and actual['finite'],recompute_cache_bytes=actual['cache_bytes'],
        control_bundle=str(control),selected_bundle=str(selected),canonical_model_qualified=False)
    report['passed']=report['finite'] and report['forward_bitwise_equal'] and report['commits_bitwise_equal']
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Recompute frozen-verifier quality',report,flush=True)
    if not report['passed']: raise RuntimeError('recompute verifier differs from frozen verification or rollback')
    return report
