"""CUDA code-generation targets and their exact device compatibility.

The architecture-specific Hopper target is deliberately explicit. Its cubins
run only on compute capability 9.0; they do not inherit forward compatibility.
"""
import re

TARGET = re.compile(r"(?:sm_[0-9]{2,3}|sm_90a)\Z")


def matches_device(target: str, arch: str) -> bool:
    return target == arch or (target == "sm_90a" and arch == "sm_90")
