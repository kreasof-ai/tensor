"""Device-independent schedule discovery and producer profile selection.

Backends supply spaces, legality predicates and coupled moves. Callers own
compilation, correctness checks, timing and budgets; no device is opened here.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import math


def key(config):
    return tuple(sorted(config.items()))


def neighbors(config, spaces, *, coupled=None):
    space = spaces[config['family']]
    for axis, values in space.items():
        index = values.index(config.get(axis, values[0]))
        for offset in (-1, 1):
            if 0 <= index + offset < len(values):
                yield {**config, axis: values[index + offset]}
    if coupled is not None:
        yield from coupled(config, space)


class ScheduleSearch:
    """Beam exploration with family retention and deterministic restarts.

    Only correctness-checked positive finite timings should be recorded. A
    finite space ends with StopIteration; rejected candidates are still seen.
    """
    def __init__(self, seeds, *, spaces, width=8, coupled=None, legal=None):
        if type(width) is not int or width <= 0:
            raise ValueError('beam width must be a positive integer')
        if not spaces or any(not axes or any(not values for values in axes.values()) for axes in spaces.values()):
            raise ValueError('schedule spaces must have nonempty families and axes')
        self.width = width
        self.spaces = {family: {axis: tuple(values) for axis, values in axes.items()} for family, axes in spaces.items()}
        self.coupled, self.legal = coupled, legal
        self.pending = [dict(seed) for seed in seeds]
        self.seen, self.results = set(), []
        for seed in self.pending:
            if seed.get('family') not in self.spaces:
                raise ValueError('seed family is absent from the schedule space')
            for axis, values in self.spaces[seed['family']].items():
                if axis in seed and seed[axis] not in values:
                    raise ValueError('seed axis is absent from the schedule space: ' + axis)
        self.restarts = self._restarts()

    def _restarts(self):
        for family, space in self.spaces.items():
            axes = list(space)
            for values in itertools.product(*(space[axis] for axis in axes)):
                yield {'family': family, **dict(zip(axes, values))}

    def next(self):
        while True:
            if not self.pending:
                ranked = sorted(self.results, key=lambda row: row[0])
                selected = ranked[:self.width]
                for family in self.spaces:
                    selected += [row for row in ranked if row[1]['family'] == family][:2]
                self.pending = [candidate for _, config in selected
                                for candidate in neighbors(config, self.spaces, coupled=self.coupled)
                                if key(candidate) not in self.seen]
                if not self.pending:
                    self.pending = [next(self.restarts)]
            config = self.pending.pop(0)
            identity = key(config)
            if identity in self.seen:
                continue
            self.seen.add(identity)
            if self.legal is None or self.legal(config):
                return config

    def record(self, config, seconds):
        if math.isfinite(seconds) and seconds > 0:
            self.results.append((seconds, dict(config)))


class ScheduleProfile:
    """Target-bound producer settings keyed by operation and semantic parameters.

    Selectors may be partial; the most specific match wins. Conflicting matches
    at equal specificity are rejected instead of depending on record order.
    Runtime consumers use compiled artifacts, never this producer utility.
    """
    def __init__(self, data):
        if data.get('schema') != 'tensor.schedule-profile.v1':
            raise ValueError('unsupported schedule profile schema')
        if not isinstance(data.get('provider'), str) or not isinstance(data.get('target'), str):
            raise ValueError('schedule profile needs provider and target')
        if not isinstance(data.get('entries'), list):
            raise ValueError('schedule profile needs entries')
        seen = set()
        for entry in data['entries']:
            if (not isinstance(entry.get('operation'), str) or not isinstance(entry.get('parameters'), dict)
                    or not isinstance(entry.get('schedule'), dict)):
                raise ValueError('invalid schedule profile entry')
            identity = json.dumps([entry['operation'], entry['parameters']], sort_keys=True)
            if identity in seen:
                raise ValueError('duplicate schedule profile selector')
            seen.add(identity)
        self.data = json.loads(json.dumps(data, allow_nan=False))

    @property
    def sha256(self):
        return hashlib.sha256(json.dumps(self.data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def select(self, operation, parameters, *, provider, target):
        if (provider, target) != (self.data['provider'], self.data['target']):
            raise ValueError('schedule profile provider/target mismatch')
        matches = [entry for entry in self.data['entries'] if entry['operation'] == operation
                   and all(name in parameters and parameters[name] == value for name, value in entry['parameters'].items())]
        if not matches:
            return {}
        specificity = max(len(entry['parameters']) for entry in matches)
        matches = [entry for entry in matches if len(entry['parameters']) == specificity]
        if any(entry['schedule'] != matches[0]['schedule'] for entry in matches[1:]):
            raise ValueError('ambiguous schedule profile selector')
        return dict(matches[0]['schedule'])
