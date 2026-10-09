"""Greedy speculative acceptance independent of model and device."""
import numpy as np


def accepted_prefix(inputs, predictions, lengths):
    """Return committed input counts and newly emitted greedy output sequences."""
    inputs, predictions, lengths = map(np.asarray, (inputs, predictions, lengths))
    if (inputs.ndim != 2 or predictions.shape != inputs.shape
            or lengths.shape != (inputs.shape[0],)
            or any(a.dtype.kind not in 'iu' for a in (inputs, predictions, lengths))
            or np.any(lengths < 0) or np.any(lengths > inputs.shape[1])):
        raise ValueError('invalid speculative verification result')
    counts = np.zeros(len(lengths), dtype='int32'); outputs = []
    for slot, length in enumerate(lengths):
        if length == 0:
            outputs.append([]); continue
        count = 1
        while count < length and predictions[slot, count-1] == inputs[slot, count]:
            count += 1
        counts[slot] = count
        outputs.append([*inputs[slot, 1:count].tolist(), int(predictions[slot, count-1])])
    return counts, outputs
