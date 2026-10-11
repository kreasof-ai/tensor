"""Loading and compiling an installed module export."""

from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import tempfile
from tensor.runtime.abi import ABI_MAJOR, ABI_MINOR
from tensor.artifacts.format import ArtifactError, read_artifact
from tensor.modules.cache import _atomic
from tensor.modules.cache import _cached
from tensor.modules.content import _canonical
from tensor.modules.content import _json
from tensor.modules.content import _logical
from tensor.modules.errors import ModuleError


class Module:
    def __init__(self, cache, digest):
        self.cache, self.digest = Path(cache), digest

    @property
    def manifest(self):
        return _cached(self.cache, self.digest).manifest

    def resolve(
        self,
        export,
        *,
        provider="cuda",
        target=None,
        compile=False,
        compiler=None,
        nvcc=None,
        nvrtc_home=None,
        cache_dir=None,
    ):
        data = _cached(self.cache, self.digest)
        if export not in data.manifest["exports"]:
            raise ModuleError(f"unknown export: {export}")
        from tensor.providers import PROVIDERS

        if provider not in PROVIDERS:
            raise ModuleError(f"unknown runtime provider: {provider}")
        implementation = importlib.import_module(PROVIDERS[provider]).Device
        missing = set(data.manifest["capabilities"]) - implementation.capabilities
        if missing:
            raise ModuleError(
                f"provider {provider} lacks module capabilities: {sorted(missing)}"
            )
        if target is None:
            if provider == "cpu":
                target = "cpu-linux-x86_64"
            elif provider == "webgpu":
                from tensor.artifacts.webgpu import TARGET as WEBGPU_TARGET

                target = WEBGPU_TARGET
            else:
                from tensor.diagnostics import check_device

                device = check_device(0)
                if device["status"] != "ok":
                    raise ModuleError("no CUDA device; supply an exact target")
                target = device["arch"]
        from tensor.artifacts.format import TARGET

        if (
            (
                provider == "cuda"
                and (not isinstance(target, str) or not TARGET.fullmatch(target))
            )
            or (provider == "cpu" and target != "cpu-linux-x86_64")
            or (provider == "webgpu" and target != "webgpu-portable-v1")
        ):
            raise ModuleError("target does not match the requested provider")
        root = self.cache / "packages" / self.digest
        spec = data.manifest["exports"][export]
        binaries, candidates = {}, []
        signature = None
        for name in dict.fromkeys(
            [
                *spec.get("artifacts", []),
                *([spec["portable"]] if "portable" in spec else []),
            ]
        ):
            manifest, files = read_artifact(root / name)
            logical = _logical(manifest)
            if signature is not None and signature != logical:
                raise ModuleError("export artifact signatures disagree")
            signature = logical
            if (
                "source" in spec
                and manifest["source_sha256"]
                != hashlib.sha256(data.files[spec["source"]]).hexdigest()
            ):
                raise ModuleError("export artifact does not match its packaged source")
            if name in spec.get("artifacts", []):
                key = (manifest.get("provider", "cuda"), manifest["target"])
                if key in binaries:
                    raise ModuleError("ambiguous exact-target export artifacts")
                binaries[key] = name
            candidates.append(
                {
                    "path": name,
                    "provider": manifest.get("provider", "cuda"),
                    "target": manifest["target"],
                }
            )
        if (provider, target) in binaries:
            return {
                "path": str(root / binaries[provider, target]),
                "selection": "packaged",
                "module_sha256": self.digest,
                "provider": provider,
                "target": target,
                "export": export,
            }
        selected_compiler = compiler or (
            "wgsl"
            if provider == "webgpu"
            else "native"
            if provider == "cpu"
            else "nvcc"
            if nvcc
            else "nvrtc"
        )
        key = hashlib.sha256(
            _canonical(
                {
                    "module": self.digest,
                    "export": export,
                    "provider": provider,
                    "target": target,
                    "compiler": selected_compiler,
                    "abi": [ABI_MAJOR, ABI_MINOR],
                }
            )
        ).hexdigest()
        output = self.cache / "artifacts" / f"{key}.tbin"
        metadata = output.with_suffix(".json")
        try:
            record = _json(metadata.read_bytes())
            if record != {"sha256": hashlib.sha256(output.read_bytes()).hexdigest()}:
                raise ModuleError("cached export hash mismatch")
            cached, _ = read_artifact(output)
            if cached.get("provider", "cuda") != provider or cached["target"] != target:
                raise ModuleError("cached export target mismatch")
            if (
                "source" in spec
                and cached["source_sha256"]
                != hashlib.sha256(data.files[spec["source"]]).hexdigest()
            ):
                raise ModuleError("cached export source mismatch")
            logical = _logical(cached)
            if signature is not None and signature != logical:
                raise ModuleError("cached export signature mismatch")
            return {
                "path": str(output),
                "selection": "cached",
                "module_sha256": self.digest,
                "provider": provider,
                "target": target,
                "export": export,
            }
        except (OSError, ValueError, ArtifactError):
            pass
        if not compile:
            raise ModuleError(
                f"no exact-target artifact for {export} on {provider}/{target}; use --compile with the pinned compiler environment"
            )
        source = spec.get("portable", spec.get("source"))
        if source is None:
            raise ModuleError(
                "export has no portable representation or source to compile"
            )
        from tensor.compiler.build import build_artifact

        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            dir=output.parent, prefix=".compile-"
        ) as directory:
            temporary = Path(directory) / "kernel.tbin"
            result = build_artifact(
                root / source,
                temporary,
                provider=provider,
                target=target,
                compiler=compiler,
                nvcc=nvcc,
                nvrtc_home=nvrtc_home,
                cache_dir=cache_dir,
            )
            built, _ = read_artifact(temporary)
            logical = _logical(built)
            if signature is not None and signature != logical:
                raise ModuleError(
                    "compiled export signature differs from its portable contract"
                )
            _atomic(output, temporary.read_bytes())
        _atomic(
            metadata,
            _canonical({"sha256": hashlib.sha256(output.read_bytes()).hexdigest()}),
        )
        return {
            "path": str(output),
            "selection": "compiled",
            "module_sha256": self.digest,
            "provider": provider,
            "target": target,
            "export": export,
            "build": result,
        }

    def load(self, export, device, **options):
        resolved = self.resolve(
            export,
            provider=device.info["provider"],
            target=device.info["arch"],
            **options,
        )
        return device.load(resolved["path"])
