"""Does libtilelang natively link against torch?

tilelang/__init__.py:135 does `import torch  # preload torch to avoid dlopen errors`.
That comment only makes sense if the native library resolves torch symbols at load
time -- i.e. a *binary* link dependency, not merely a Python import.

This parses the PE import directory of the shipped DLL and lists every DLL it
imports from. If torch/c10 appear there, a torch-free binary requires rebuilding
TileLang, not just making imports lazy.

    python experiments/p0/link_check.py                 # this venv
    python experiments/p0/link_check.py <site-packages> # a comparison venv
    python experiments/p0/link_check.py <sp> --all      # every import, not just torch
"""

import struct
import sys
from pathlib import Path

SITE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(sys.prefix) / "Lib" / "site-packages" / "tilelang"
SHOW_ALL = "--all" in sys.argv


def find_libs():
    for pat in ("**/*.dll", "**/*.so", "**/*.pyd"):
        for p in SITE.glob(pat):
            if "tilelang" in p.name.lower() or "tvm" in p.name.lower():
                yield p


def pe_imports(path: Path) -> list[str]:
    """Return the names of DLLs listed in the PE import directory."""
    data = path.read_bytes()
    if data[:2] != b"MZ":
        return []

    pe_off = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe_off:pe_off + 4] != b"PE\0\0":
        return []

    # COFF header -> optional header
    coff = pe_off + 4
    num_sections = struct.unpack_from("<H", data, coff + 2)[0]
    opt_size = struct.unpack_from("<H", data, coff + 16)[0]
    opt = coff + 20
    magic = struct.unpack_from("<H", data, opt)[0]
    pe32plus = magic == 0x20B

    dd = opt + (112 if pe32plus else 96)  # data directory start
    imp_rva, imp_size = struct.unpack_from("<II", data, dd + 8)  # dir[1] = import
    if not imp_rva:
        return []

    # map RVA -> file offset using section table
    sections = []
    sec = opt + opt_size
    for i in range(num_sections):
        base = sec + i * 40
        vsize, vaddr, rawsize, rawptr = struct.unpack_from("<IIII", data, base + 8)
        sections.append((vaddr, vsize, rawptr, rawsize))

    def to_off(rva):
        for vaddr, vsize, rawptr, rawsize in sections:
            if vaddr <= rva < vaddr + max(vsize, rawsize):
                return rawptr + (rva - vaddr)
        return None

    names = []
    off = to_off(imp_rva)
    if off is None:
        return []
    while True:
        entry = data[off:off + 20]
        if len(entry) < 20 or entry == b"\0" * 20:
            break
        name_rva = struct.unpack_from("<I", entry, 12)[0]
        if not name_rva:
            break
        n_off = to_off(name_rva)
        if n_off is None:
            break
        end = data.index(b"\0", n_off)
        names.append(data[n_off:end].decode("ascii", "replace"))
        off += 20
    return names


TORCH_HINTS = ("torch", "c10", "caffe2", "torch_cpu", "torch_cuda", "c10_cuda", "fbgemm")

found_any = False
for lib in sorted(find_libs()):
    imports = pe_imports(lib)
    if not imports:
        continue
    hits = [i for i in imports if any(h in i.lower() for h in TORCH_HINTS)]
    found_any = True
    print(f"\n=== {lib.name} ({lib.stat().st_size / 1e6:.1f} MB) ===")
    print(f"  imports {len(imports)} libraries total")
    if SHOW_ALL:
        for i in sorted(imports):
            print(f"      {i}")
    for h in hits:
        print(f"  TORCH-LINKED -> {h}")
    if not hits:
        print("  (no torch/c10 in the import table)")

if not found_any:
    print("No PE libraries found to inspect.")
