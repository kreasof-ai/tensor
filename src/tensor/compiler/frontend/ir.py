"""Portable frontend operator inventory."""

from __future__ import annotations

import json


def _op_set(serialized: str) -> list[str]:
    from tensor.compiler.errors import BuildError

    nodes = json.loads(serialized)["nodes"]

    def label(index: int) -> str:
        operation = nodes[index].get("data")
        if isinstance(operation, dict):
            for field in ("name", "global_name", "key", "op_name"):
                if isinstance(operation.get(field), str):
                    return operation[field]
            hint = operation.get("name_hint")
            if type(hint) is int and 0 <= hint < len(nodes):
                value = nodes[hint].get("data")
                if isinstance(value, str):
                    return value
        if isinstance(operation, str):
            return operation
        raise BuildError("cannot identify a frontend IR operator")

    ops = set()
    for node in nodes:
        if node.get("type") != "tirx.Call":
            continue
        index = node.get("data", {}).get("op")
        if type(index) is not int or not 0 <= index < len(nodes):
            raise BuildError("invalid operator reference in frontend IR")
        ops.add(label(index))
    return sorted(ops)
