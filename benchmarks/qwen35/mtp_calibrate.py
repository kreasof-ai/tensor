"""Measure a native one-token MTP draft against teacher-forced and AR targets.

Execute ``calibrate(model, mtp, tokens, out)`` on the resident owner thread.
This resets both executors, and performs serial target evaluation to measure
draft agreement. It is not accelerated speculative decoding or a stress run.
"""
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def calibrate(model, mtp, tokens, out, *, decode_steps=128):
    model._check(); mtp._check()
    tokens = np.asarray(tokens)
    if (tokens.ndim != 2 or tokens.shape[1] != model.slots or tokens.shape[0] < 2
            or tokens.dtype.kind not in 'iu' or np.any(tokens < 0)
            or np.any(tokens >= model.config.vocab) or type(decode_steps) is not int or decode_steps < 1
            or tokens.shape[0]+decode_steps > model.context):
        raise ValueError('calibration requires valid [steps, slots] tokens and room for decode')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model.reset(); mtp.reset()
    truth = model.forward(tokens[0])
    records = []
    for phase, steps in (('teacher_forced', len(tokens)-1), ('autoregressive', decode_steps)):
        for index in range(steps):
            inputs = tokens[index+1] if phase == 'teacher_forced' else truth.copy()
            started = time.perf_counter()
            proposal = mtp.draft(inputs, model.buffers['normal'])
            draft_seconds = time.perf_counter()-started
            started = time.perf_counter()
            truth = model.forward(inputs)
            target_seconds = time.perf_counter()-started
            records.append(dict(phase=phase, index=index, target_context=int(model.position.max()),
                matches=int((proposal == truth).sum()), slots=model.slots,
                draft_seconds=draft_seconds, target_seconds=target_seconds))
            if (index+1) % 128 == 0:
                print('MTP calibration', phase, index+1, 'matches',
                      sum(r['matches'] for r in records if r['phase']==phase), flush=True)
    summaries = {}
    for phase in ('teacher_forced', 'autoregressive'):
        subset = [r for r in records if r['phase'] == phase]
        summaries[phase] = dict(steps=len(subset), proposed_tokens=len(subset)*model.slots,
            matches=sum(r['matches'] for r in subset),
            agreement=sum(r['matches'] for r in subset)/(len(subset)*model.slots),
            mean_draft_seconds=float(np.mean([r['draft_seconds'] for r in subset])),
            mean_target_seconds=float(np.mean([r['target_seconds'] for r in subset])))
    report = dict(scope='serial one-token draft agreement/cost diagnostic; no batched verification',
        slots=model.slots, prompt_steps=len(tokens), kv_dtype=model.kv_dtype,
        token_ids_sha256=hashlib.sha256(np.ascontiguousarray(tokens, dtype='int32').tobytes()).hexdigest(),
        draft_weight_bytes=mtp.checkpoint.weight_bytes, draft_allocated_bytes=mtp.allocated_bytes,
        summaries=summaries, speculative_verification=False, full_stress_target_reached=False,
        model_throughput_qualified=False)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print('MTP calibration result', report, flush=True)
    return report


def calibrate_long(model, mtp, target_prefill, draft_prefill, tokens, out, *, decode_steps=256):
    """Initialize the real long prefix and measure serial one-token agreement.

    This resets both models and closes the supplied prefix graphs before decode.
    It leaves the resident models initialized at the diagnostic's final context.
    """
    model._check(); mtp._check()
    tokens = np.asarray(tokens)
    if (tokens.ndim != 2 or tokens.shape[1] != model.slots or tokens.dtype.kind not in 'iu'
            or len(tokens) < 1 or np.any(tokens < 0) or np.any(tokens >= model.config.vocab)
            or target_prefill.model is not model or draft_prefill.model is not mtp
            or target_prefill.chunk != draft_prefill.chunk
            or type(decode_steps) is not int or decode_steps < 1
            or len(tokens)+decode_steps > model.context):
        raise ValueError('invalid long MTP diagnostic inputs or prefill executors')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    model.reset(); mtp.reset()
    chunk = target_prefill.chunk
    target_prefill_seconds = draft_prefill_seconds = 0.
    for start in range(0, len(tokens), chunk):
        count = min(chunk, len(tokens)-start)
        inputs = np.zeros((model.slots, chunk), dtype='int32')
        inputs[:, :count] = tokens[start:start+count].T
        lengths = np.full(model.slots, count, dtype='int32')
        started = time.perf_counter()
        pending = target_prefill.forward(inputs, lengths)
        target_prefill_seconds += time.perf_counter()-started
        shifted = np.zeros_like(inputs)
        if count > 1: shifted[:, :count-1] = inputs[:, 1:count]
        shifted[:, count-1] = tokens[start+count] if start+count < len(tokens) else pending
        started = time.perf_counter()
        proposal = draft_prefill.forward(shifted, target_prefill.buffers['normal'], lengths)
        draft_prefill_seconds += time.perf_counter()-started
        if (start//chunk+1) % 16 == 0:
            print('Native target/MTP initialized prefix', start+count, flush=True)
    target_prefill.close(); draft_prefill.close()
    records = []
    for index in range(decode_steps):
        started = time.perf_counter()
        truth = model.forward(pending)
        target_seconds = time.perf_counter()-started
        matches = int((proposal == truth).sum())
        started = time.perf_counter()
        proposal = mtp.draft(truth, model.buffers['normal'])
        draft_seconds = time.perf_counter()-started
        pending = truth
        records.append(dict(index=index, target_context=int(model.position.max()), matches=matches,
                            slots=model.slots, draft_seconds=draft_seconds, target_seconds=target_seconds))
    report = dict(scope='full-length prefix; short serial one-token MTP agreement diagnostic',
        prompt_steps=len(tokens), slots=model.slots, decode_steps=decode_steps,
        verified_proposals=decode_steps*model.slots, matches=sum(r['matches'] for r in records),
        agreement=sum(r['matches'] for r in records)/(decode_steps*model.slots),
        mean_draft_seconds=float(np.mean([r['draft_seconds'] for r in records])),
        mean_target_seconds=float(np.mean([r['target_seconds'] for r in records])),
        target_prefill_seconds=target_prefill_seconds, draft_prefill_seconds=draft_prefill_seconds,
        token_ids_sha256=hashlib.sha256(np.ascontiguousarray(tokens, dtype='int32').tobytes()).hexdigest(),
        target_end_positions=model.position.tolist(), draft_end_positions=mtp.position.tolist(),
        kv_dtype=model.kv_dtype, draft_weight_bytes=mtp.checkpoint.weight_bytes,
        draft_allocated_bytes=mtp.allocated_bytes, speculative_verification=False,
        full_stress_target_reached=False, model_throughput_qualified=False)
    (out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (out/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    print('Long-context MTP diagnostic', report, flush=True)
    return report
