"""AOT region coverage and selection stay compiler-free and fail closed."""

import hashlib
import itertools
import json
import random
import zipfile

import pytest

from tensor.modules.errors import ModuleError
from tensor.modules.portfolio import select, validate
from tensor.modules.project import Project
from tensor.modules.resolver import install
from tests.runtime.test_domain import bounded_manifest


def portfolio():
    return dict(symbols={"n": "int32"},
                shape_domain={"version": 1, "bounds": {"n": {"min": 1, "max": 256}}, "relations": []},
                regions=[dict(bounds={"n": {"min": 1, "max": 32}}, artifacts=["small.tbin"]),
                         dict(bounds={"n": {"min": 33, "max": 256}}, artifacts=["large.tbin"])])


def test_multiaxis_coverage_agrees_with_independent_enumeration():
    rng = random.Random(17)
    for _ in range(150):
        regions = []
        for _ in range(rng.randint(1, 12)):
            bounds = {}
            for name in ("B", "T"):
                low = rng.randint(1, 6)
                bounds[name] = dict(min=low, max=rng.randint(low, 6))
            regions.append(dict(bounds=bounds, artifacts=["kernel.tbin"]))
        p = dict(symbols={"B": "int32", "T": "int32"},
                 shape_domain=dict(version=1, bounds={"B": dict(min=1, max=6),
                     "T": dict(min=1, max=6, multiple_of=2)}, relations=[]), regions=regions)
        if any(all(n % 2 for n in range(r["bounds"]["T"]["min"], r["bounds"]["T"]["max"] + 1))
               for r in regions):
            continue  # Empty regions are rejected separately.
        complete = all(any(all(r["bounds"][k]["min"] <= v <= r["bounds"][k]["max"]
                                      for k, v in (("B", b), ("T", t))) for r in regions)
                       for b, t in itertools.product(range(1, 7), (2, 4, 6)))
        if complete:
            validate(p)
        else:
            with pytest.raises(ModuleError, match="hole"):
                validate(p)


def test_overlaps_use_declared_order_and_guards_reject_before_resolution():
    p = portfolio()
    p["regions"][0]["bounds"]["n"]["max"] = 64
    validate(p)
    assert select(p, {"n": 33})[0] == 0
    assert select(p, {"n": 129})[0] == 1
    for dimensions in ({"n": 0}, {"n": 257}, {"n": True}, {"n": 1.5}, {}, {"n": 1, "T": 1}):
        with pytest.raises(ModuleError):
            select(p, dimensions)


def image(path, low, high):
    manifest = bounded_manifest()
    manifest["shape_domain"]["bounds"]["n"] = dict(min=low, max=high)
    payload = {"kernel.cubin": b"\x7fELFfixture", "kernel.tirx.json": b'{"nodes":[]}'}
    manifest["files"] = {k: hashlib.sha256(v).hexdigest() for k, v in payload.items()}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps(manifest))
        for name, contents in payload.items():
            archive.writestr(name, contents)


def package(tmp_path, p):
    root = tmp_path / "lib"
    root.mkdir()
    manifest = dict(formatVersion=2, name="ops", version="0.1.0", tensorAbi=1,
                    exports={"affine": {"portfolio": p}})
    (root / "tensor.json").write_text(json.dumps(manifest))
    return root


def test_installed_regions_select_images_without_frontend_or_gpu(tmp_path):
    p = portfolio()
    p["evidence"] = "measurements.json"
    root = package(tmp_path, p)
    image(root / "small.tbin", 1, 256)  # A shared broad image is legal.
    image(root / "large.tbin", 33, 256)
    (root / "measurements.json").write_text('{"qualification":"test fixture"}')
    cache = tmp_path / "cache"
    install(root, cache_dir=cache)
    module = Project(root, cache_dir=cache).module()
    for n, expected in ((1, 0), (3, 0), (31, 0), (32, 0), (33, 1), (129, 1), (256, 1)):
        selected = module.resolve("affine", provider="cuda", target="sm_86", dimensions={"n": n})
        assert selected["region"] == expected and selected["selection"] == "packaged"
        assert selected["dimensions"] == {"n": n}
        assert selected["path"].endswith("small.tbin" if expected == 0 else "large.tbin")
    for dimensions in (None, {"n": 257}):
        with pytest.raises(ModuleError):
            module.resolve("affine", target="sm_86", dimensions=dimensions)
    before = module.digest
    (root / "measurements.json").write_text('{"qualification":"changed selection evidence"}')
    install(root, cache_dir=cache)
    assert Project(root, cache_dir=cache).module().digest != before


@pytest.mark.parametrize("problem", ["hole", "narrow", "legacy", "signature"])
def test_invalid_portfolios_fail_installation(tmp_path, problem):
    p = portfolio()
    root = package(tmp_path, p)
    image(root / "small.tbin", 1, 32)
    image(root / "large.tbin", 34 if problem == "narrow" else 33, 256)
    manifest = json.loads((root / "tensor.json").read_text())
    if problem == "hole":
        manifest["exports"]["affine"]["portfolio"]["regions"][1]["bounds"]["n"]["min"] = 34
    if problem == "legacy":
        manifest["formatVersion"] = 1
    if problem == "signature":
        with zipfile.ZipFile(root / "large.tbin") as archive:
            files = {n: archive.read(n) for n in archive.namelist()}
        data = json.loads(files["manifest.json"])
        data["arguments"][0]["dtype"] = "float16"
        data["abi"][0]["dtype"] = "float16"
        files["manifest.json"] = json.dumps(data).encode()
        with zipfile.ZipFile(root / "large.tbin", "w") as archive:
            for name, contents in files.items():
                archive.writestr(name, contents)
    (root / "tensor.json").write_text(json.dumps(manifest))
    with pytest.raises(ModuleError, match="hole|cover|version-2|signatures disagree"):
        install(root, cache_dir=tmp_path / "cache")
