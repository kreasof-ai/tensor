"""Execute a hash-verified Linux/Windows CI artifact ZIP in a clean GPU consumer.

CI evidence must include the run head SHA, successful jobs, and artifact names
and SHA-256 digests returned by GitHub's Actions API. Source hashes are checked
against that exact Git revision, accepting its LF or Windows CRLF checkout.
The consumer may use a newer compatible runtime; its revision is reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import importlib.metadata as metadata
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import zipfile

from phase1_transfer_check import NoCompilerImports, exercise
import sys

EXAMPLES = ("elementwise", "gemm_relu", "dynamic_affine", "dynamic_gemm", "scalar_offset")
ROOT = Path(__file__).resolve().parents[1]


def check(archive: Path, ci_record: Path, platform: str):
    sys.meta_path.insert(0, NoCompilerImports())
    from tensor.artifact import read_artifact

    ci = json.loads(ci_record.read_text())
    revision = ci["headSha"]
    if ci["conclusion"] != "success" or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("acceptance requires a successful CI run and full producer revision")
    name = "tensor-phase2-" + {"linux": "ubuntu-24.04", "windows": "windows-latest"}[platform]
    candidates = [item for item in ci["artifacts"] if item["name"] == name and not item["expired"]]
    if len(candidates) != 1:
        raise ValueError("CI record must identify one unexpired platform artifact")
    identity = candidates[0]
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if "sha256:" + digest != identity["digest"]:
        raise ValueError("downloaded CI archive hash mismatch")
    wheel = "dist/tensor_workspace-0.1.0-py3-none-any.whl"
    prefix = "build/phase2-transfer/"
    members = {prefix + name + ".tbin" for name in EXAMPLES} | {
        prefix + "nvrtc-producer.json", "build/nvrtc-12.9/bootstrap.json", wheel}
    with tempfile.TemporaryDirectory(prefix="tensor-phase2-transfer-") as directory:
        root = Path(directory)
        with zipfile.ZipFile(archive) as bundle:
            module_members = {"build/phase3-transfer/ops.tpack", "build/phase3-transfer/phase3-producer.json"}
            if set(bundle.namelist()) & module_members:
                members |= module_members
            if (set(bundle.namelist()) != members or len(bundle.infolist()) != len(members)
                    or sum(item.file_size for item in bundle.infolist()) > 5 * 1024 * 1024):
                raise ValueError("unexpected CI archive contents or size")
            bundle.extractall(root)
        producer = json.loads((root / prefix / "nvrtc-producer.json").read_text())
        if producer["hostname"] == socket.gethostname():
            raise ValueError("CI producer and GPU consumer must be different hosts")
        if set(producer["artifacts"]) != {name + ".tbin" for name in EXAMPLES}:
            raise ValueError("incomplete producer acceptance matrix")
        artifacts, source_endings = {}, {}
        for name in EXAMPLES:
            artifact = root / prefix / (name + ".tbin")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != producer["artifacts"][artifact.name]:
                raise ValueError(f"producer artifact hash mismatch: {name}")
            manifest, _ = read_artifact(artifact)
            if manifest["target"] != producer["target"]:
                raise ValueError(f"producer target mismatch: {name}")
            source = subprocess.check_output(["git", "show", f"{revision}:examples/{name}.py"], cwd=ROOT)
            hashes = {hashlib.sha256(source).hexdigest(): "LF",
                      hashlib.sha256(source.replace(b"\n", b"\r\n")).hexdigest(): "CRLF"}
            if manifest["source_sha256"] not in hashes:
                raise ValueError(f"source does not match CI producer revision: {name}")
            source_endings[name] = hashes[manifest["source_sha256"]]
            artifacts[name] = artifact
        report = exercise(artifacts)
        import tensor as tx
        contracts = []
        with tx.Device() as device:
            for name, artifact in artifacts.items():
                kernel = device.load(artifact)
                descriptor = kernel.descriptor
                workspace = kernel.workspace_requirements()
                assert device.get_executable(descriptor) is kernel
                assert workspace.byte_size == 0 and workspace.alignment == 1
                assert workspace.device_type == device.device_type and not workspace.flags and not workspace.reserved
                kernel.release()
                try:
                    device.get_executable(descriptor)
                except device.error:
                    pass
                else:
                    raise AssertionError("released executable descriptor remained valid")
                contracts.append({"artifact": name, "workspace_bytes": 0, "released_handle_rejected": True})
            event = device.record_event()
            descriptor = event.descriptor
            assert device.get_event(descriptor) is event
            device.wait(event)
            event.release()
            try:
                device.get_event(descriptor)
            except device.error:
                pass
            else:
                raise AssertionError("released event descriptor remained valid")
        distribution = metadata.distribution("tensor-workspace")
        with zipfile.ZipFile(root / wheel) as packaged:
            matching_wheel = all(distribution.locate_file(name).read_bytes() == packaged.read(name)
                                 for name in packaged.namelist() if name.startswith("tensor/"))
        report["contracts"] = contracts
        report["event_descriptor"] = {"resolved": True, "released_handle_rejected": True}
        report["consumer_matches_producer_wheel"] = matching_wheel
        report.update({"two_hosts": True, "producer_platform": platform, "producer": producer,
            "producer_revision": revision, "ci_run": ci["url"], "archive": identity,
            "archive_sha256": digest, "source_line_endings": source_endings,
            "producer_wheel_sha256": hashlib.sha256((root / wheel).read_bytes()).hexdigest(),
            "consumer_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "consumer_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))})
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--ci-record", type=Path, required=True)
    parser.add_argument("--platform", choices=("linux", "windows"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.archive, args.ci_record, args.platform)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as output:
        output.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "platform": args.platform,
                      "cases": len(result["records"]), "diagnostics": len(result["diagnostics"])}))
