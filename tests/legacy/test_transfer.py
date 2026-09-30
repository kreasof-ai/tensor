"""Reject misleading transfer evidence even when kernel execution passes."""

import pytest
from experiments.p0.artifact_format import ArtifactError
from experiments.p0.transfer_check import audit


def evidence():
    records = []
    for size in (1,127,128,129,1025):
        identity = {"git_revision": "revision", "source_sha256": "source", "lock_sha256": "lock", "git_dirty": False}
        result = {"status": "passed", "size": size, "compiler_imports": [], "compiler_import_guard": True,
                  "producer": {**identity, "hostname": "build"},
                  "consumer": {**identity, "hostname": "gpu", "packages": {"numpy": "2.5.3"}}}
        records.append({"exit_code": 0, "result": result})
    return {"status": "passed", "records": records}


def test_matching_matrix():
    assert audit(evidence())["two_hosts"]


@pytest.mark.parametrize("field,value", [("hostname","build"),("git_revision","other"),
                                         ("source_sha256","other"),("lock_sha256","other"),
                                         ("git_dirty",True),("packages",{"torch":"2.14.0"})])
def test_rejects_wrong_host_or_checkout(field, value):
    report = evidence()
    report["records"][0]["result"]["consumer"][field] = value
    with pytest.raises(ArtifactError):
        audit(report)


def test_requires_coverage_and_import_guard():
    report = evidence()
    with pytest.raises(ArtifactError, match="boundary-size"):
        audit({**report,"records":report["records"][:-1]})
    report["records"][0]["result"]["compiler_import_guard"] = False
    with pytest.raises(ArtifactError, match="isolation"):
        audit(report)
