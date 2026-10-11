"""Install a pinned NVRTC distribution and headers; no nvcc or host compiler.

The nvcc redistribution archive supplies CRT headers only. Its executable,
ptxas, nvvm and compiler libraries are never installed. Downloads are checked
against pinned NVIDIA redistribution hashes. Supports Linux/Windows x86-64.
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
import zipfile

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = {
    "cuda_nvrtc": ("12.9.86", "82913658363892dbc0f2638b070476234476e06e084fed60db861cb7e161a6af",
                   "1aa0644fa53c8ca34cdc73db17bcc73530557bdd3f582c7bfdbd7916c8b48f65"),
    "cuda_cudart": ("12.9.79", "1f6ad42d4f530b24bfa35894ccf6b7209d2354f59101fd62ec4a6192a184ce99",
                    "179e9c43b0735ffe67207b3da556eb5a0c50f3047961882b7657d3b822d34ef8"),
    "cuda_cccl": ("12.9.27", "8b1a5095669e94f2f9afd7715533314d418179e9452be61e2fde4c82a3e542aa",
                  "17aaa7c6b8f94a417d8f3261780b7e34b9cbdfab7513bce86768623b06aa28b5"),
    "cuda_nvcc": ("12.9.86", "7a1a5b652e5ef85c82b721d10672fc9a2dbaab44e9bd3c65a69517bf53998c35",
                  "227b109663b5e57d2718bcabb24a4ba0d9d4e52d958e327dc476f7c28691be85"),
}


def install(destination: Path, *, download_dir: Path | None = None) -> dict:
    system = platform.system()
    if system not in ("Linux", "Windows") or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("NVRTC bootstrap supports Linux/Windows x86-64")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    # Keep archives outside the compiler distribution so its footprint is honest.
    downloads = download_dir or destination.parent / "nvrtc-downloads"
    downloads.mkdir(parents=True, exist_ok=True)
    target = "linux-x86_64" if system == "Linux" else "windows-x86_64"
    rows = []
    for component, (version, linux_hash, windows_hash) in COMPONENTS.items():
        name = f"{component}-{target}-{version}-archive"
        extension = ".tar.xz" if system == "Linux" else ".zip"
        url = f"{BASE}{component}/{target}/{name}{extension}"
        archive = downloads / (name + extension)
        if not archive.exists():
            partial = archive.with_suffix(archive.suffix + ".partial")
            try:
                with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as output:
                    shutil.copyfileobj(response, output)
                partial.replace(archive)
            finally:
                partial.unlink(missing_ok=True)
        with archive.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != (linux_hash if system == "Linux" else windows_hash):
            raise RuntimeError(f"SHA-256 mismatch for {archive}; remove it and retry")
        with tempfile.TemporaryDirectory(prefix="tensor-nvrtc-") as temporary:
            if system == "Linux":
                with tarfile.open(archive) as bundle:
                    bundle.extractall(temporary, filter="data")
            else:
                with zipfile.ZipFile(archive) as bundle:
                    bundle.extractall(temporary)
            root = Path(temporary) / name
            shutil.copytree(root / "include", destination / "include", dirs_exist_ok=True)
            notices = destination / "licenses" / component
            notices.mkdir(parents=True, exist_ok=True)
            for item in root.iterdir():
                if item.name.lower().startswith(("license", "eula")):
                    if item.is_dir():
                        shutil.copytree(item, notices / item.name, dirs_exist_ok=True)
                    else:
                        shutil.copy2(item, notices / item.name)
            if component == "cuda_nvrtc":
                # Ship dynamic NVRTC + builtins only, never the static libraries.
                library_dir = root / ("lib" if system == "Linux" else "bin")
                output = destination / "lib"
                output.mkdir(exist_ok=True)
                for item in library_dir.iterdir():
                    if item.is_file() and (".so" in item.name if system == "Linux" else item.suffix == ".dll"):
                        shutil.copy2(item, output / item.name, follow_symlinks=False)
        rows.append({"component": component, "version": version, "url": url,
                     "sha256": digest, "download_bytes": archive.stat().st_size,
                     "installed": "headers and NVRTC libraries" if component == "cuda_nvrtc" else "headers only"})
    paths = [path for path in destination.rglob("*") if path.is_file() and not path.is_symlink()]
    report = {"root": str(destination), "cuda_version": "12.9", "components": rows,
              "installed_bytes": sum(path.stat().st_size for path in paths),
              "files": len(paths), "toolkit_executables": []}
    (destination / "bootstrap.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(install(arguments.out, download_dir=arguments.download_dir), indent=2))
