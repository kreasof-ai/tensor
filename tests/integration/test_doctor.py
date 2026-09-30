"""Doctor checks the usable build path, including GPU-free build hosts."""

import subprocess

from tensor.cli import doctor


def test_explicit_target_allows_gpu_free_build_host(monkeypatch):
    monkeypatch.setattr(doctor, "check_packages", lambda: {"status": "ok"})
    monkeypatch.setattr(doctor, "check_provider", lambda: {"status": "ok"})
    monkeypatch.setattr(doctor, "check_device", lambda ordinal: {"status": "unavailable"})
    monkeypatch.setattr(doctor, "check_toolchain", lambda target, nvcc: {"status": "ok", "target": target})

    report = doctor.diagnose(target="sm_86", compiler="nvcc")

    assert report["status"] == "build_ready"
    assert report["checks"]["toolchain"]["target"] == "sm_86"


def test_no_target_skips_compilation_when_device_is_absent(monkeypatch):
    monkeypatch.setattr(doctor, "check_packages", lambda: {"status": "ok"})
    monkeypatch.setattr(doctor, "check_provider", lambda: {"status": "ok"})
    monkeypatch.setattr(doctor, "check_device", lambda ordinal: {"status": "unavailable"})

    report = doctor.diagnose()

    assert report["status"] == "needs_setup"
    assert report["checks"]["toolchain"]["status"] == "skipped"
    assert "--target" in report["checks"]["toolchain"]["hint"]


def test_nvcc_version_alone_does_not_pass_the_toolchain_check(monkeypatch):
    monkeypatch.setattr(doctor, "_resolve_nvcc", lambda explicit: "/fake/nvcc")

    def run(command, **kwargs):
        if "--version" in command:
            return subprocess.CompletedProcess(command, 0, "nvcc 12.9", "")
        return subprocess.CompletedProcess(command, 1, "", "fatal error: cuda_runtime.h missing")

    monkeypatch.setattr(doctor.subprocess, "run", run)
    check = doctor.check_toolchain("sm_86", None)

    assert check["status"] == "error"
    assert "cuda_runtime.h" in check["detail"]


def test_invalid_device_ordinal_cannot_report_build_ready(monkeypatch):
    monkeypatch.setattr(doctor, "check_packages", lambda: {"status": "ok"})

    report = doctor.diagnose(target="sm_86", device=-1)

    assert report["status"] == "needs_setup"
    assert report["checks"]["device"]["status"] == "error"


def test_runtime_only_install_reports_run_ready(monkeypatch):
    monkeypatch.setattr(doctor, "check_packages", lambda: {"status": "error", "detail": "compiler missing"})
    monkeypatch.setattr(doctor, "check_device", lambda ordinal: {"status": "ok", "arch": "sm_86"})
    monkeypatch.setattr(doctor, "check_toolchain", lambda target, nvcc: {"status": "error"})

    report = doctor.diagnose()

    assert report["status"] == "run_ready"
    assert report["target"] == "sm_86"
