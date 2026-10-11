"""Dispatch explicit artifact builds to the selected compiler backend."""

from pathlib import Path
from tensor.compiler.errors import BuildError


def build_artifact(
    source_path: Path,
    output_path: Path,
    *,
    target=None,
    nvcc=None,
    cache_dir=None,
    compiler=None,
    nvrtc_home=None,
    provider="cuda",
):
    if provider == "cuda":
        from tensor.compiler.cuda.build import build_cuda as build
    elif provider == "cpu":
        from tensor.compiler.cpu import build_cpu as build
    elif provider == "webgpu":
        from tensor.compiler.webgpu.build import build_webgpu as build
    else:
        raise BuildError(f"unsupported compiler provider {provider!r}")
    return build(
        source_path,
        output_path,
        target=target,
        nvcc=nvcc,
        cache_dir=cache_dir,
        compiler=compiler,
        nvrtc_home=nvrtc_home,
    )
