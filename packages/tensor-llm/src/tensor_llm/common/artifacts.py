"""Stable kernel identities shared by model producers and runtime executors."""
import hashlib
import json


def identity(kind, parameters):
    return hashlib.sha256(json.dumps([kind, parameters], sort_keys=True).encode()).hexdigest()[:24]
