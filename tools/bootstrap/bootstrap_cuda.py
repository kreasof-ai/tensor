"""Install the pinned CUDA 12.9 build components in a local Linux directory.

Uses NVIDIA's redistribution archives and checks their published SHA-256s.
No GPU, root access, system installation, or Python dependencies are needed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = {
    "cuda_nvcc": ("12.9.86", "7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35"),
    "cuda_cudart": ("12.9.79", "1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99"),
    "cuda_cccl": ("12.9.27", "8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa"),
}


def install(destination):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("this P0 bootstrap supports Linux x86_64 only")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archives = destination / ".archives"
    archives.mkdir(exist_ok=True)
    rows = []
    for component, (version, expected) in COMPONENTS.items():
        name = f"{component}-linux-x86_64-{version}-archive"
        url = f"{BASE}{component}/linux-x86_64/{name}.tar.xz"
        archive = archives / f"{name}.tar.xz"
        if not archive.exists():
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        actual = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {archive}; remove it and retry")
        with tempfile.TemporaryDirectory(prefix="tensor-cuda-") as temporary:
            with tarfile.open(archive) as bundle:
                bundle.extractall(temporary, filter="data")
            shutil.copytree(Path(temporary) / name, destination, dirs_exist_ok=True)
        rows.append({"component": component, "version": version, "url": url, "sha256": actual})
    lib64 = destination / "lib64"
    if not lib64.exists():
        lib64.symlink_to("lib", target_is_directory=True)
    report = {"cuda_home": str(destination), "components": rows,
              "manifest_url": BASE + "redistrib_12.9.1.json"}
    (destination / "bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(install(parser.parse_args().out), indent=2))
