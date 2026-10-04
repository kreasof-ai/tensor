"""Build and verify local release files without uploading anything.

Run with Python 3.12 and uv available on PATH. No compiler or GPU is required.
"""

from __future__ import annotations

import argparse
from email.parser import BytesParser
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PROJECTS = {
    "tensor-workspace": ROOT,
    "tensor-nn": ROOT / "packages/tensor-nn",
    "tensor-llm": ROOT / "packages/tensor-llm",
    "tensor-torch": ROOT / "packages/tensor-torch",
}


def run(*args, cwd=ROOT, env=None):
    result = subprocess.run([str(arg) for arg in args], cwd=cwd, env=env,
                            capture_output=True, text=True)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="", file=sys.stderr)
        result.check_returncode()


def build_environment():
    env = os.environ.copy()
    for flag in ("TENSOR_BUILD_WEBGPU_NATIVE", "TENSOR_TORCH_BUILD_NATIVE"):
        env.pop(flag, None)
    return env


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_metadata(raw, project):
    metadata = BytesParser().parsebytes(raw)
    expected = tomllib.loads((project / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    require(metadata["Name"] == expected["name"], "distribution name mismatch")
    require(metadata["Version"] == expected["version"], "distribution version mismatch")
    require(metadata["License-Expression"] == "MIT", "missing MIT license expression")
    require(metadata.get_all("License-File") == ["LICENSE"], "missing license metadata")
    require("Development Status :: 3 - Alpha" in metadata.get_all("Classifier", []),
            "missing pre-1.0 classifier")
    require(metadata["Description-Content-Type"] == "text/markdown", "missing Markdown README")
    require(metadata.get_payload().strip(), "empty package README")
    require(metadata["Requires-Python"] == "<3.13,>=3.12", "unexpected Python support range")
    require(metadata.get_all("Project-URL"), "missing project URLs")
    dependencies = metadata.get_all("Requires-Dist", [])
    if expected["name"] == "tensor-workspace":
        require([dep for dep in dependencies if ";" not in dep] == ["numpy==2.5.3"],
                "core runtime dependencies must remain NumPy only")
    else:
        version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
        require(f"tensor-workspace=={version}" in dependencies, "optional package must pin the core version")
    return metadata


def check_archives(out):
    wheels = {}
    sources = {}
    license_text = (ROOT / "LICENSE").read_bytes()
    for project in PROJECTS.values():
        require((project / "LICENSE").read_bytes() == license_text, f"license differs: {project}")
    for path in sorted(out.glob("*.whl")):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            entries = [name for name in names if name.endswith(".dist-info/METADATA")]
            require(len(entries) == 1, f"invalid wheel metadata: {path}")
            name = BytesParser().parsebytes(archive.read(entries[0]))["Name"]
            require(name in PROJECTS and name not in wheels, f"unexpected or duplicate wheel: {path}")
            check_metadata(archive.read(entries[0]), PROJECTS[name])
            license_path = entries[0].removesuffix("METADATA") + "licenses/LICENSE"
            require(archive.read(license_path) == license_text, f"wheel license differs: {path}")
            if name == "tensor-workspace":
                require(all(item.startswith(("tensor/", "tensor_workspace-")) for item in names),
                        "core wheel contains files outside the runtime package")
                for item in ("tensor/include/tensor/abi.h", "tensor/native/host.cpp",
                             "tensor/native/host.rs", "tensor/native/webgpu_plan.c"):
                    require(item in names, f"missing installed native source/header: {item}")
            if name == "tensor-torch":
                require("tensor_torch/templates/attention.py" in names, "missing Torch kernel template")
            wheels[name] = path
    for path in sorted(out.glob("*.tar.gz")):
        with tarfile.open(path) as archive:
            members = archive.getnames()
            roots = {name.split("/")[0] for name in members}
            require(len(roots) == 1, f"invalid sdist root: {path}")
            prefix = roots.pop() + "/"
            raw = archive.extractfile(prefix + "PKG-INFO").read()
            name = BytesParser().parsebytes(raw)["Name"]
            require(name in PROJECTS and name not in sources, f"unexpected or duplicate sdist: {path}")
            check_metadata(raw, PROJECTS[name])
            require(archive.extractfile(prefix + "LICENSE").read() == license_text,
                    f"sdist license differs: {path}")
            require(prefix + "README.md" in members, f"sdist missing README: {path}")
            if name == "tensor-torch":
                for item in ("src/tensor_torch/launch.c", "src/tensor_torch/executor.cpp"):
                    require(prefix + item in members, f"sdist missing native source: {item}")
            sources[name] = path
    require(wheels.keys() == PROJECTS.keys(), "build one wheel for each of the four distributions")
    require(sources.keys() == PROJECTS.keys(), "build one sdist for each of the four distributions")
    return wheels, sources


SMOKE = '''
import importlib.abc
from importlib import metadata
import sys

class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tilelang', 'tvm', 'tvm_ffi', 'torch', 'triton',
                                     'wgpu', 'tensor_nn', 'tensor_llm', 'tensor_torch'}:
            raise ImportError('optional import prohibited: ' + fullname)

sys.meta_path.insert(0, Guard())
import numpy as np
import tensor
from tensor.cli import main
assert {d.metadata['Name'].lower() for d in metadata.distributions()} == {'tensor-workspace', 'numpy'}
assert tensor.__version__ == metadata.version('tensor-workspace')
tensor.assert_close(np.array([1, 2], dtype=np.float32), [1, 2])
for args in (['--version'], ['--help'], ['run', '--help'], ['build', '--help']):
    try:
        main(args)
    except SystemExit as exc:
        assert exc.code == 0
assert not any(name == 'tensor.compiler' or name.startswith('tensor.compiler.') for name in sys.modules)
print('PASS: installed core and CLI work without compiler/framework imports')
'''


def check_consumer(wheel, directory):
    print("Checking an isolated compiler-free consumer...", flush=True)
    environment = directory / "consumer"
    run("uv", "venv", "--python", "3.12", environment, cwd=directory)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run("uv", "pip", "install", "--python", python, wheel, cwd=directory)
    run(python, "-I", "-c", SMOKE, cwd=directory)
    run(python, "-I", "-m", "tensor", "--version", cwd=directory)
    cli = environment / ("Scripts/tensor.exe" if os.name == "nt" else "bin/tensor")
    run(cli, "--version", cwd=directory)
    run(python, "-I", ROOT / "examples/run_elementwise.py", "--help", cwd=directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/distributions",
                        help="new output directory, or existing files with --check-only")
    parser.add_argument("--check-only", action="store_true", help="verify already built distributions")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not args.check_only:
        require(not any(out.glob("*.whl")) and not any(out.glob("*.tar.gz")),
                "output already contains release files; choose a fresh --out directory")
        # Ordinary release builds should remain portable, even when a developer
        # has native-build switches set in their interactive environment.
        for project in PROJECTS.values():
            print(f"Building {project.name}...", flush=True)
            run("uv", "build", project, "--out-dir", out, env=build_environment())
    wheels, sources = check_archives(out)
    print("Checking archive contents and Twine metadata...", flush=True)
    run("uvx", "--from", "twine>=6.2,<8", "twine", "check", *wheels.values(), *sources.values())
    with tempfile.TemporaryDirectory(prefix="tensor-distribution-") as temporary:
        directory = Path(temporary)
        # Ensure source archives are self-contained, not just buildable in a checkout.
        rebuilt = directory / "rebuilt"
        for source in sources.values():
            print(f"Rebuilding {source.name} outside the checkout...", flush=True)
            run("uv", "build", source, "--wheel", "--out-dir", rebuilt, cwd=directory,
                env=build_environment())
        check_consumer(wheels["tensor-workspace"], directory)
    print(f"PASS: four wheels, four source archives, source rebuilds and compiler-free consumer: {out}")


if __name__ == "__main__":
    main()
