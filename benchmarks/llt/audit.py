"""Verify retained LLT qualification evidence and artifact/source identities."""

import hashlib
import json
import zipfile
from pathlib import Path
from tensor.artifacts.format import read_artifact
from .qualify import ROOT, OUT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    reports = {}
    artifacts = {}
    for name in (
        "cold",
        "gradients",
        "training",
        "leak",
        "generation",
        "attention",
        "decode",
        "loss-memory",
        "systems",
    ):
        report = json.loads((OUT / (name + ".json")).read_text())
        assert report["status"] == "passed", name
        reports[name] = report
        for source, sha in report["environment"]["sources"].items():
            snapshot = OUT / "sources" / sha / Path(source).name
            assert digest(snapshot) == sha, (name, source, "snapshot")
            assert digest(ROOT / source) == sha, (
                name,
                source,
                "current implementation changed",
            )
        for record in report["coverage"]["artifacts"]:
            path = Path(record["path"])
            assert digest(path) == record["sha256"], path
            manifest, images = read_artifact(path)
            assert manifest["target"] == "sm_89", path
            if any(x["dtype"] == "bfloat16" for x in manifest["arguments"]):
                assert manifest["runtime_abi"]["minor"] >= 3
                assert (
                    "bfloat16_storage"
                    in manifest["runtime_abi"]["required_capabilities"]
                )
            # Manifest validation already verifies payload SHA and ABI/capability.
            assert record["target"] == "sm_89", record
            artifacts[str(path)] = record["sha256"]
    for row in reports["training"]["runs"]:
        assert row["steps"] >= 1000
        assert row["resume_maximum_parameter_error"] <= 1e-6
    for report in reports.values():
        assert not report["coverage"]["fallbacks"]

    def samples(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.endswith("samples_ms"):
                    assert len(item) >= 9, key
                    assert all(x > 0 for x in item), key
                else:
                    samples(item)
        elif isinstance(value, list):
            for item in value:
                samples(item)

    for report in reports.values():
        samples(report)
    consumer = json.loads((OUT / "consumer.json").read_text())
    assert consumer["status"] == "passed" and not consumer["compiler_packages"]
    assert consumer["native_plans"] > 0
    wheels = {p.name: digest(p) for p in (ROOT / "build/llt-wheels").glob("*.whl")}
    assert len(wheels) == 2, wheels
    for wheel in (ROOT / "build/llt-wheels").glob("*.whl"):
        source_root = ROOT / (
            "packages/tensor-torch/src"
            if wheel.name.startswith("tensor_torch")
            else "src"
        )
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                source = source_root / name
                if source.is_file() and name.endswith((".py", ".h", ".cpp")):
                    assert archive.read(name) == source.read_bytes(), (wheel.name, name)
    for report in reports.values():
        sha = report.get("source_sha256", report.get("scaling_source_sha256"))
        if sha:
            file = (
                "leak.py"
                if "runs" in report and "source_sha256" in report
                else "scaling.py"
            )
            assert digest(OUT / "sources" / sha / file) == sha
            assert digest(ROOT / "benchmarks/llt" / file) == sha

    summary = {
        "status": "passed",
        "reports": {name: digest(OUT / (name + ".json")) for name in reports},
        "consumer_sha256": digest(OUT / "consumer.json"),
        "wheels": wheels,
        "unique_artifacts": len(artifacts),
        "artifact_sha256": artifacts,
        "validation_sha256": digest(OUT / "validation.json"),
        "pinned_layout_limitation": "reproduced; shared-memory workaround exercised by decode",
        "target": "sm_89",
        "gate": "LLT dependency profile; separate from Tensor 1.0 and LLT quality",
    }
    (OUT / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("LLT dependency evidence audit passed:", len(artifacts), "artifacts")


if __name__ == "__main__":
    main()
