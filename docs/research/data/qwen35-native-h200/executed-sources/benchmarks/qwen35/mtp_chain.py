"""Measure native multi-token MTP prefix agreement using serial target checks.

The target advances through its own true greedy continuation each round.
Correct draft-cache entries are rebuilt from those target states. This is
acceptance calibration, not batched verification or a speculative speed claim.
"""
import json
from pathlib import Path
import time
import numpy as np


def measure(model, mtp, out, *, proposals=3, rounds=64):
    model._check(); mtp._check()
    if (type(proposals) is not int or not 1 <= proposals <= 4 or type(rounds) is not int or rounds < 1
            or not np.array_equal(model.position, mtp.position)
            or not np.array_equal(model.active, mtp.active) or not model.active.all()
            or np.any(model.position+rounds*(proposals+1) > model.context)):
        raise ValueError('chain calibration requires synchronized, active, initialized slots and context space')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    hidden = [model.device.empty((model.slots, 2048), 'bfloat16') for _ in range(proposals+1)]
    histogram = np.zeros(proposals+1, dtype='int64')
    records = []; start_context = model.position.copy()
    try:
        for index in range(rounds):
            base = mtp.position.copy()
            drafts = [mtp.buffers['tokens'].to_numpy()]
            pending = model.buffers['tokens'].to_numpy()
            started = time.perf_counter()
            for _ in range(proposals-1):
                drafts.append(mtp.draft(drafts[-1], mtp.buffers['normal']))
            extra_draft_seconds = time.perf_counter()-started
            truth = []; target_seconds = 0.
            for step in range(proposals+1):
                started = time.perf_counter()
                pending = model.forward(pending)
                target_seconds += time.perf_counter()-started
                truth.append(pending.copy())
                model.device.driver.call('cuMemcpyDtoD_v2', hidden[step].pointer,
                                         model.buffers['normal'].pointer, hidden[step].nbytes)
                model.device.driver.call('cuStreamSynchronize', None)
            equality = np.stack(drafts) == np.stack(truth[:proposals])
            accepted = np.cumprod(equality.astype('int32'), axis=0).sum(axis=0)
            histogram += np.bincount(accepted, minlength=proposals+1)
            # Rewind tentative full-attention draft entries. The old valid
            # prefix is immutable; correction overwrites only appended entries.
            mtp.position = base
            mtp._write('positions', base)
            for ids, state in zip(truth, hidden): mtp.draft(ids, state)
            records.append(dict(round=index, accepted_prefix_lengths=accepted.tolist(),
                extra_draft_seconds=extra_draft_seconds, serial_target_seconds=target_seconds,
                target_context=int(model.position.max())))
    finally:
        for buffer in hidden: buffer.release()
    report = dict(scope='native MTP chain agreement; serial target verification and cache correction',
        slots=model.slots, proposals=proposals, rounds=rounds, context_start=start_context.tolist(),
        context_end=model.position.tolist(), accepted_prefix_histogram=histogram.tolist(),
        mean_accepted_prefix=float(sum(i*n for i, n in enumerate(histogram))/histogram.sum()),
        mean_tokens_per_verification=float(1+sum(i*n for i, n in enumerate(histogram))/histogram.sum()),
        all_proposals_agree=float(histogram[-1]/histogram.sum()),
        mean_additional_draft_seconds=float(np.mean([r['extra_draft_seconds'] for r in records])),
        speculative_verification=False, full_stress_target_reached=False, model_throughput_qualified=False)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'rounds.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print('MTP chain calibration', report, flush=True)
    return report
