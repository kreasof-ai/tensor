"""JSON-compatible compilation reports returned as independent snapshots."""

import copy

_reports = []


def reports():
    return copy.deepcopy(_reports)
