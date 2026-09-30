"""Registry transport acceptance using a real local Simple Index and uploader."""
from contextlib import contextmanager
from email.parser import BytesParser
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import zipfile

import pytest

from tensor.artifacts.modules import ModuleError, Project, add, install, pack
from tensor.artifacts.registry import _record, build_wheel, index_url, publish, read_wheel


def module(path, name, *, dependencies=None):
    path.mkdir(parents=True)
    (path / "tensor.json").write_text(json.dumps({"formatVersion": 1, "name": name,
        "version": "0.1.0", "tensorAbi": 1, "exports": {"kernel": "kernel.py"},
        "dependencies": dependencies or {}}))
    (path / "kernel.py").write_text("raise RuntimeError('registry operations executed source')\n")
    return path


@contextmanager
def local_index(*, html=False):
    state = {"wheels": {}, "requests": [], "yanked": False, "bad_hash": False,
             "corrupt": False, "duplicates": False, "uploads": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def respond(self, data, content_type):
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            state["requests"].append(self.path)
            if self.path.startswith("/simple/"):
                name = self.path.split("/")[2]
                files = []
                for filename, wheel in state["wheels"].items():
                    if filename.split("-")[0].replace("_", "-") != name:
                        continue
                    digest = "0" * 64 if state["bad_hash"] else hashlib.sha256(wheel).hexdigest()
                    files.append({"filename": filename, "url": "/files/" + filename,
                                  "hashes": {"sha256": digest}, "yanked": state["yanked"]})
                if state["duplicates"]:
                    files *= 2
                if html:
                    data = "<!DOCTYPE html>" + "".join(
                        f'<a href="{f["url"]}#sha256={f["hashes"]["sha256"]}"'
                        + (' data-yanked="removed"' if f["yanked"] else '')
                        + f'>{f["filename"]}</a>' for f in files)
                    self.respond(data.encode(), "text/html")
                else:
                    self.respond(json.dumps({"meta": {"api-version": "1.0"}, "name": name,
                                             "files": files}).encode(), "application/vnd.pypi.simple.v1+json")
            elif self.path.startswith("/files/") and self.path[7:] in state["wheels"]:
                wheel = state["wheels"][self.path[7:]]
                self.respond(b"corrupt" if state["corrupt"] else wheel, "application/octet-stream")
            else:
                self.send_error(404)

        def do_POST(self):
            # Parse actual Twine multipart uploads, then expose the received bytes.
            data = self.rfile.read(int(self.headers["Content-Length"]))
            envelope = ("Content-Type: " + self.headers["Content-Type"] + "\nMIME-Version: 1.0\n\n").encode()
            message = BytesParser().parsebytes(envelope + data)
            for part in message.walk():
                filename = part.get_filename()
                if filename:
                    state["wheels"][filename] = part.get_payload(decode=True)
                    state["uploads"].append(filename)
            self.respond(b"OK", "text/plain")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    state["base"] = f"http://127.0.0.1:{server.server_port}"
    state["index"] = state["base"] + "/simple/"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def wheel_closure(tmp_path):
    leaf = module(tmp_path / "leaf", "base-ops")
    root = module(tmp_path / "lib", "my-ops", dependencies={
        "base-ops": {"path": "../leaf", "version": "0.1.0"}})
    report = build_wheel(root, out_dir=tmp_path / "dist")
    return root, leaf, report, Path(report["path"])


def test_deterministic_wheel_relocates_closure_and_is_pip_installable(tmp_path):
    root, leaf, report, wheel = wheel_closure(tmp_path)
    second = build_wheel(root, out_dir=tmp_path / "again")
    assert wheel.read_bytes() == Path(second["path"]).read_bytes()
    archive = tmp_path / "ops.tpack"
    pack(root, archive)
    third = build_wheel(archive, out_dir=tmp_path / "from-archive")
    assert report["sha256"] == third["sha256"]
    target = tmp_path / "site-packages"
    # Exercise a real Python wheel installer, with no dependency resolution.
    uv = shutil.which("uv")
    if uv:
        subprocess.run([uv, "pip", "install", "--python", sys.executable, "--target", str(target),
                        "--no-deps", str(wheel)], check=True, capture_output=True)
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "--target", str(target),
                        "--no-deps", str(wheel)], check=True, capture_output=True)
    payload = target / "tensor_module_payloads/tensor_module_my_ops/module.tpack"
    assert payload.read_bytes() == archive.read_bytes()
    shutil.rmtree(root)
    shutil.rmtree(leaf)
    app = module(tmp_path / "app", "app")
    add(wheel, app, cache_dir=tmp_path / "cache")
    wheel.unlink()
    install(app, cache_dir=tmp_path / "cache", frozen=True, offline=True)
    assert Project(app, cache_dir=tmp_path / "cache").module("base-ops").manifest["name"] == "base-ops"


@pytest.mark.parametrize("html", [False, True])
def test_registry_pins_restore_frozen_graph_and_repair_corrupt_cache(tmp_path, html):
    _, _, report, wheel = wheel_closure(tmp_path)
    app = module(tmp_path / "app", "app")
    cache = tmp_path / "cache"
    with local_index(html=html) as server:
        server["wheels"][wheel.name] = wheel.read_bytes()
        added = add("pypi:Tensor_Module.My_Ops==0.1.0", app, cache_dir=cache, index_url=server["index"])
        assert added["pypi"]["sha256"] == report["sha256"]
        manifest = json.loads((app / "tensor.json").read_text())
        assert manifest["dependencies"]["my-ops"]["pypi"]["index"] == server["index"]
        assert "path" not in manifest["dependencies"]["my-ops"]
        before = (app / "tensor.lock").read_bytes()
        server["requests"].clear()
        install(app, cache_dir=cache, frozen=True)
        assert server["requests"] == []
        # A new host has only the committed project manifest and lock.
        fresh = tmp_path / "fresh"
        install(app, cache_dir=fresh, frozen=True)
        assert len(server["requests"]) == 2
        assert (app / "tensor.lock").read_bytes() == before
        installed = Project(app, cache_dir=fresh)
        leaf = installed.module("base-ops")
        (fresh / "packages" / leaf.digest / "kernel.py").write_bytes(b"corrupt")
        with pytest.raises(ModuleError, match="cached registry dependency"):
            install(app, cache_dir=fresh, frozen=True, offline=True)
        install(app, cache_dir=fresh, frozen=True)
        assert Project(app, cache_dir=fresh).module("base-ops").manifest["name"] == "base-ops"
        assert (app / "tensor.lock").read_bytes() == before
    # No registry is needed once the complete cache is present.
    install(app, cache_dir=fresh, frozen=True, offline=True)


@pytest.mark.parametrize("failure", ["bad_hash", "corrupt", "duplicates", "yanked"])
def test_invalid_registry_releases_preserve_project(tmp_path, failure):
    _, _, _, wheel = wheel_closure(tmp_path)
    app = module(tmp_path / "app", "app")
    install(app, cache_dir=tmp_path / "cache")
    before = [(app / p).read_bytes() for p in ("tensor.json", "tensor.lock")]
    with local_index() as server:
        server["wheels"][wheel.name] = wheel.read_bytes()
        server[failure] = True
        with pytest.raises(ModuleError):
            add("pypi:tensor-module-my-ops==0.1.0", app, cache_dir=tmp_path / "cache", index_url=server["index"])
    assert [(app / p).read_bytes() for p in ("tensor.json", "tensor.lock")] == before


def test_pinned_release_cannot_change_but_can_be_restored_when_yanked(tmp_path):
    _, _, _, wheel = wheel_closure(tmp_path)
    app = module(tmp_path / "app", "app")
    with local_index() as server:
        server["wheels"][wheel.name] = wheel.read_bytes()
        add("pypi:tensor-module-my-ops==0.1.0", app, cache_dir=tmp_path / "cache", index_url=server["index"])
        before = (app / "tensor.lock").read_bytes()
        server["bad_hash"] = True
        with pytest.raises(ModuleError, match="project pin"):
            install(app, cache_dir=tmp_path / "fresh", frozen=True)
        assert not (tmp_path / "fresh/packages").exists()
        server["bad_hash"] = False
        server["yanked"] = True
        install(app, cache_dir=tmp_path / "fresh", frozen=True)
        assert (app / "tensor.lock").read_bytes() == before


def test_wheel_rejects_record_tampering_and_executable_members(tmp_path):
    _, _, _, wheel = wheel_closure(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        files = {name: archive.read(name) for name in archive.namelist()}
    for change in ("record", "payload", "pth", "metadata", "descriptor"):
        altered = dict(files)
        if change == "record":
            key = next(n for n in files if n.endswith("/RECORD"))
            altered[key] = b""
        elif change == "payload":
            key = next(n for n in files if n.endswith("/module.tpack"))
            altered[key] += b"corrupt"
        elif change == "pth":
            altered["evil.pth"] = b"import os; raise RuntimeError('executed')"
        elif change == "metadata":
            key = next(n for n in files if n.endswith("/METADATA"))
            altered[key] = files[key].replace(b"Name: tensor-module-my-ops", b"Name: different-project")
        else:
            key = next(n for n in files if n.endswith("/tensor-module.json"))
            descriptor = json.loads(files[key]); descriptor["module"] = "different-module"
            altered[key] = json.dumps(descriptor).encode()
        if change in {"payload", "metadata", "descriptor"}:
            # A valid outer RECORD must not bypass inner identity/hash checks.
            record_path = next(n for n in files if n.endswith("/RECORD"))
            altered[record_path] = _record({n:d for n,d in altered.items() if n != record_path}, record_path)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            for name, contents in altered.items():
                archive.writestr(name, contents)
        with pytest.raises(ModuleError):
            read_wheel(io.BytesIO(buffer.getvalue()))
    with pytest.raises(ModuleError, match="filename"):
        wrong = tmp_path / "renamed.whl"
        wrong.write_bytes(wheel.read_bytes())
        read_wheel(wrong)


def test_registry_cli_dry_run_add_inspect_and_offline_install(tmp_path, capsys):
    from tensor.cli import main
    root = module(tmp_path / "lib", "my-ops")
    app = module(tmp_path / "app", "app")
    assert main(["publish", str(root), "--dry-run", "--out-dir", str(tmp_path / "dist")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "prepared" and report["dry_run"]
    wheel = Path(report["path"])
    assert main(["inspect", str(wheel)]) == 0
    assert json.loads(capsys.readouterr().out)["root"] == "my-ops"
    common = ["--project", str(app), "--module-cache", str(tmp_path / "cache")]
    with local_index() as server:
        server["wheels"][wheel.name] = wheel.read_bytes()
        assert main(["add", "pypi:tensor-module-my-ops==0.1.0", "--index-url", server["index"], *common]) == 0
        assert json.loads(capsys.readouterr().out)["name"] == "my-ops"
    assert main(["install", "--frozen", "--offline", *common]) == 0
    assert json.loads(capsys.readouterr().out)["frozen"]
    with pytest.raises(SystemExit):
        main(["add", "pypi:my-ops>=0.1", *common])
    assert "registry references" in capsys.readouterr().err


def test_registry_rejects_credential_urls_and_remote_http(tmp_path):
    for url in ("http://example.com/simple/", "https://user:secret@example.com/simple/",
                "file:///tmp/index", "https://example.com/simple/?token=secret"):
        with pytest.raises(ModuleError):
            index_url(url)
    root = module(tmp_path / "lib", "my-ops")
    with pytest.raises(ModuleError):
        publish(root, out_dir=tmp_path / "dist", repository_url="https://user:secret@example.com/upload")
    assert not (tmp_path / "dist").exists()


def test_actual_twine_upload_and_registry_round_trip(tmp_path, monkeypatch):
    if importlib.util.find_spec("twine") is None:
        pytest.skip("publisher extra required for actual Twine upload")
    root = module(tmp_path / "lib", "my-ops")
    app = module(tmp_path / "app", "app")
    monkeypatch.setenv("TWINE_USERNAME", "__token__")
    monkeypatch.setenv("TWINE_PASSWORD", "local-test-token")
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.null.Keyring")
    with local_index() as server:
        report = publish(root, out_dir=tmp_path / "dist", repository_url=server["base"] + "/legacy/")
        assert report["status"] == "published"
        assert server["uploads"] == [Path(report["path"]).name]
        added = add("pypi:tensor-module-my-ops==0.1.0", app, index_url=server["index"], cache_dir=tmp_path / "cache")
        assert added["pypi"]["sha256"] == report["sha256"]
    install(app, cache_dir=tmp_path / "cache", frozen=True, offline=True)
