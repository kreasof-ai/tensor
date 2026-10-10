"""Compare batched verification with native serial decoding at a real prefix."""
import json
from pathlib import Path
import numpy as np


def run(model, draft, verify_bundle, out, *, steps=2):
    from tensor_llm import Qwen35Verifier
    from .spec_benchmark import PrefixSnapshot
    from .spec_run import install_selected,make_verifier
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = PrefixSnapshot(model, draft)
    inputs, reference, greedy = [], [], []
    pending = snapshot.target.tokens.copy()
    verifier = None
    try:
        for _ in range(steps):
            inputs.append(pending.copy())
            pending, logits = model.forward(pending, read_logits=True)
            reference.append(logits)
            greedy.append(pending.copy())
        snapshot.restore(model, draft)
        verifier = make_verifier(model, verify_bundle)
        install_selected(verifier, verify_bundle)
        if not 1 <= steps <= verifier.chunk:
            raise ValueError('quality steps must fit the verification chunk')
        tokens = np.zeros((model.slots, verifier.chunk), 'int32')
        tokens[:, :steps] = np.stack(inputs, axis=1)
        predictions, logits = verifier.forward(
            tokens, np.full(model.slots, steps, 'int32'), read_logits=True)
        actual = logits.reshape(model.slots, verifier.chunk, -1)[:, :steps]
        expected = np.stack(reference, axis=1)
        expected_greedy = np.stack(greedy, axis=1)
        report = dict(
            scope='teacher-forced verification vs same native serial target at initialized prefix',
            device=model.device.info, prefix_positions=snapshot.target.position.tolist(),
            steps=steps, kv_dtype=model.kv_dtype, threshold=.03,
            relative_logit_rms=float(np.linalg.norm(actual-expected)/np.linalg.norm(expected)),
            greedy_matches=int((predictions[:, :steps] == expected_greedy).sum()),
            greedy_total=model.slots*steps, finite=bool(np.isfinite(actual).all()),
            canonical_model_qualified=False, model_throughput_qualified=False)
        report['passed'] = (report['finite'] and report['relative_logit_rms'] <= .03
                            and report['greedy_matches'] == report['greedy_total'])
        (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        np.savez(out/'sample.npz', inputs=tokens, predictions=predictions[:, :steps],
                 reference_greedy=expected_greedy, actual_logits=actual, reference_logits=expected)
        verifier.commit(np.full(model.slots, steps, 'int32'))
        print('Verification serial quality', report, flush=True)
        return report
    finally:
        if verifier is not None:
            verifier.close()
        snapshot.restore(model, draft)
