"""tensor.artifacts.portable: migrated implementation with explicit ownership."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from tensor.artifacts.format import read_artifact


def preflight(path):
    from tensor.compiler.errors import BuildError
    from tensor.compiler.frontend.ir import _op_set

    manifest, files = read_artifact(path)
    for field, package in (
        ("tilelang_version", "tilelang"),
        ("tvm_ffi_version", "apache-tvm-ffi"),
    ):
        try:
            installed = version(package)
        except PackageNotFoundError as exc:
            raise BuildError(
                "portable compilation needs Tensor's pinned compiler environment"
            ) from exc
        if manifest[field] != installed:
            raise BuildError(
                f"portable frontend version mismatch: {package} {manifest[field]} requires that exact version; installed {installed}"
            )
    try:
        operations = _op_set(files["kernel.tirx.json"].decode())
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise BuildError(f"invalid portable operator graph: {exc}") from exc
    if operations != manifest["op_set"]:
        raise BuildError("portable operator set does not match the bundled frontend IR")
    return manifest, files
