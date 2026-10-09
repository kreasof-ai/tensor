"""Shared NVRTC configuration for native Qwen artifact producers."""
import os


def needs_build(source, destination, text, target):
    """Reuse a specialization only when its source and CUDA target both match."""
    from tensor.artifacts.format import read_artifact
    if not source.is_file() or not destination.is_file() or source.read_text() != text:
        return True
    manifest, _ = read_artifact(destination)
    return manifest['target'] != target


def build_artifact(source, destination, *, target):
    from tensor.compiler.build import build_artifact as compile_artifact
    return compile_artifact(
        source, destination, target=target, compiler='nvrtc',
        nvrtc_home=os.environ.get('TENSOR_NVRTC_HOME', 'build/nvrtc-12.9'))
