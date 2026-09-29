"""Install pinned rustc and std into a local Linux x86_64 prefix for P0 ABI checks."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

RELEASE = "1.98.1"
DATE = "2026-09-03"
ARCH = "x86_64-unknown-linux-gnu"
COMPONENTS = {
    "rustc": "e974f036b28565f37c0f3bd92ddefa809bee16c04f9dcf07b9ed96e05aaaf7c4",
    "rust-std": "fa3ff450172a16c026944030230c5069947af93c728d9179971d44e5e0cfb561",
}


def install(destination):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("this P0 bootstrap supports Linux x86_64 only")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    archives = destination / ".archives"
    archives.mkdir(exist_ok=True)
    rows = []
    for component, expected in COMPONENTS.items():
        name = f"{component}-{RELEASE}-{ARCH}"
        url = f"https://static.rust-lang.org/dist/{DATE}/{name}.tar.xz"
        archive = archives / f"{name}.tar.xz"
        if not archive.exists():
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        with archive.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"SHA-256 mismatch for {archive}; remove it and retry")
        with tempfile.TemporaryDirectory(prefix="tensor-rust-") as temporary:
            with tarfile.open(archive) as bundle:
                bundle.extractall(temporary, filter="data")
            subprocess.run(["sh", str(Path(temporary) / name / "install.sh"),
                            f"--prefix={destination}", "--disable-ldconfig"],
                           check=True, capture_output=True, text=True, timeout=120)
        rows.append({"component": component, "version": RELEASE, "url": url, "sha256": actual})
    version = subprocess.run([str(destination / "bin" / "rustc"), "--version"],
                             check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    if not version.startswith(f"rustc {RELEASE} "):
        raise RuntimeError(f"unexpected Rust version: {version}")
    report = {"rustc": str(destination / "bin" / "rustc"), "version": version, "components": rows}
    (destination / "bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    print(json.dumps(install(parser.parse_args().out), indent=2))
