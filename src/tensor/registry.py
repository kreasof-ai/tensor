"""PyPI transport for Tensor packages; wheels carry data, never install hooks.

The consumer uses only the standard library. Twine is an optional publisher
dependency, imported by its command line only when an upload is requested.
"""
from __future__ import annotations

import base64
import csv
from email.parser import BytesParser
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import zipfile

from tensor.modules import (HASH, MAX_BYTES, MAX_FILES, VERSION, ModuleError,
                            _archive, _canonical, _json, _path, pack)

PYPI_INDEX = "https://pypi.org/simple/"
INDEX_BYTES = 4 * 1024 * 1024


def distribution_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", value):
        raise ModuleError("invalid Python distribution name")
    return re.sub(r"[-_.]+", "-", value).lower()


def _url(value):
    if not isinstance(value, str) or any(c.isspace() or ord(c) < 32 for c in value):
        raise ModuleError("invalid registry URL")
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError as exc:
        raise ModuleError("invalid registry URL") from exc
    if (not parts.hostname or parts.username is not None or parts.password is not None
            or parts.scheme not in {"https", "http"}
            or (parts.scheme == "http" and parts.hostname not in {"localhost", "127.0.0.1", "::1"})):
        raise ModuleError("registry URLs require HTTPS (HTTP is allowed on loopback); credentials belong in Twine")
    if port is not None and not 0 < port <= 65535:
        raise ModuleError("invalid registry port")
    return value


def index_url(value):
    _url(value)
    parts = urlsplit(value)
    if parts.query or parts.fragment:
        raise ModuleError("index URLs cannot contain queries or fragments")
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/") + "/", "", ""))


def origin(value):
    if (not isinstance(value, dict) or set(value) != {"index", "distribution", "sha256"}
            or not isinstance(value["sha256"], str) or not HASH.fullmatch(value["sha256"])):
        raise ModuleError("pypi dependencies need index, distribution and wheel sha256")
    return {"index": index_url(value["index"]), "distribution": distribution_name(value["distribution"]),
            "sha256": value["sha256"]}


def reference(value):
    if not isinstance(value, str) or not value.startswith("pypi:") or value.count("==") != 1:
        raise ModuleError("registry references use pypi:distribution-name==major.minor.patch")
    name, version = value[5:].split("==")
    name = distribution_name(name)
    if not VERSION.fullmatch(version):
        raise ModuleError("registry versions must be exact major.minor.patch")
    return name, version


def _filename(name, version):
    return f"{name.replace('-', '_')}-{version}-py3-none-any.whl"


def _record(files, record_path):
    text = io.StringIO(newline="")
    writer = csv.writer(text, lineterminator="\n")
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + digest, str(len(data))))
    writer.writerow((record_path, "", ""))
    return text.getvalue().encode()


def _wheel_files(payload, graph, packages, name):
    root = packages[graph["root"]]
    version = root.manifest["version"]
    stem = name.replace("-", "_")
    info = f"{stem}-{version}.dist-info"
    resource = f"tensor_module_payloads/{stem}/module.tpack"
    metadata = (f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
                f"Summary: Tensor module {graph['root']}\nDescription-Content-Type: text/plain\n\n"
                f"Tensor module {graph['root']} {version}, runtime ABI {root.manifest['tensorAbi']}.\n"
                "Contains a verified Tensor package and its complete dependency closure.\n"
                "Install the payload with tensor add; Tensor selects device-compatible exports.\n").encode()
    descriptor = {"format": "tensor.pypi", "formatVersion": 1, "distribution": name,
                  "module": graph["root"], "version": version, "module_sha256": root.digest,
                  "payload": resource, "payload_sha256": hashlib.sha256(payload).hexdigest()}
    files = {resource: payload, f"{info}/METADATA": metadata,
             f"{info}/WHEEL": b"Wheel-Version: 1.0\nGenerator: tensor\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
             f"{info}/tensor-module.json": _canonical(descriptor)}
    files[f"{info}/RECORD"] = _record(files, f"{info}/RECORD")
    return files


def build_wheel(source=".", *, out_dir="dist", distribution=None, cache_dir=None):
    """Build an installable data wheel with a deterministic, verified closure."""
    source = Path(source)
    with tempfile.TemporaryDirectory(prefix="tensor-wheel-") as directory:
        if source.is_dir():
            archive = Path(directory) / "module.tpack"
            pack(source, archive, cache_dir=cache_dir)
        else:
            archive = source
        graph, packages = _archive(archive)
        root = packages[graph["root"]]
        name = distribution_name(distribution if distribution is not None else "tensor-module-" + graph["root"])
        version = root.manifest["version"]
        files = _wheel_files(archive.read_bytes(), graph, packages, name)
        output = Path(out_dir) / _filename(name, version)
        output.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with output.open("xb") as stream:
                created = True
                with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as wheel:
                    for member, data in sorted(files.items()):
                        info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0))
                        info.create_system = 3
                        info.compress_type = zipfile.ZIP_DEFLATED
                        info.external_attr = 0o100644 << 16
                        wheel.writestr(info, data)
            read_wheel(output)
        except BaseException:
            if created:
                output.unlink(missing_ok=True)
            raise
    return {"status": "prepared", "path": str(output.resolve()), "distribution": name,
            "name": graph["root"], "version": version, "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "module_sha256": root.digest,
            "modules": graph["packages"]}


def read_wheel(path, *, distribution=None, version=None):
    """Validate our data-only wheel profile, without importing or extracting it."""
    try:
        with zipfile.ZipFile(path) as wheel:
            infos = wheel.infolist()
            names = [item.filename for item in infos]
            if (len(names) != len(set(names)) or len(names) > MAX_FILES
                    or sum(item.file_size for item in infos) > MAX_BYTES
                    or any(_path(n) != n for n in names)
                    or any((item.external_attr >> 16) & 0o170000 == 0o120000 for item in infos)):
                raise ModuleError("invalid, duplicate or oversized wheel members")
            descriptors = [n for n in names if n.endswith(".dist-info/tensor-module.json")]
            if len(descriptors) != 1 or wheel.getinfo(descriptors[0]).file_size > INDEX_BYTES:
                raise ModuleError("wheel is not a Tensor data package")
            descriptor = _json(wheel.read(descriptors[0]))
            fields = {"format", "formatVersion", "distribution", "module", "version", "module_sha256", "payload", "payload_sha256"}
            if (not isinstance(descriptor, dict) or set(descriptor) != fields
                    or descriptor["format"] != "tensor.pypi" or type(descriptor["formatVersion"]) is not int
                    or descriptor["formatVersion"] != 1):
                raise ModuleError("invalid Tensor wheel descriptor")
            name = distribution_name(descriptor["distribution"])
            release = descriptor["version"]
            if not isinstance(release, str) or not VERSION.fullmatch(release) or name != descriptor["distribution"]:
                raise ModuleError("invalid Tensor wheel identity")
            if (distribution is not None and name != distribution) or (version is not None and release != version):
                raise ModuleError("wheel distribution/version mismatch")
            if isinstance(path, (str, Path)) and Path(path).name != _filename(name, release):
                raise ModuleError("wheel filename disagrees with its metadata")
            info = f"{name.replace('-', '_')}-{release}.dist-info"
            payload_path = f"tensor_module_payloads/{name.replace('-', '_')}/module.tpack"
            expected = {payload_path, *{f"{info}/{n}" for n in ("METADATA", "WHEEL", "RECORD", "tensor-module.json")}}
            if set(names) != expected or descriptor["payload"] != payload_path:
                raise ModuleError("Tensor wheels must contain only their payload and metadata")
            files = {n: wheel.read(n) for n in names}
        record_path = f"{info}/RECORD"
        rows = list(csv.reader(io.StringIO(files[record_path].decode())))
        expected_rows = list(csv.reader(io.StringIO(_record({n:d for n,d in files.items() if n != record_path}, record_path).decode())))
        if sorted(rows) != sorted(expected_rows):
            raise ModuleError("wheel RECORD hash/size mismatch")
        metadata = BytesParser().parsebytes(files[f"{info}/METADATA"])
        for key, expected_value in (("Metadata-Version", "2.1"), ("Name", name), ("Version", release)):
            if metadata.get_all(key) != [expected_value]:
                raise ModuleError("wheel METADATA identity mismatch")
        if metadata.get_all("Requires-Dist"):
            raise ModuleError("Tensor transport wheels cannot have Python dependencies")
        wheel_metadata = BytesParser().parsebytes(files[f"{info}/WHEEL"])
        for key, expected_value in (("Wheel-Version", "1.0"), ("Root-Is-Purelib", "true"), ("Tag", "py3-none-any")):
            if wheel_metadata.get_all(key) != [expected_value]:
                raise ModuleError("unsupported Tensor wheel compatibility tags")
        payload = files[payload_path]
        if hashlib.sha256(payload).hexdigest() != descriptor["payload_sha256"]:
            raise ModuleError("wheel payload hash mismatch")
        graph, packages = _archive(io.BytesIO(payload))
        root = packages[graph["root"]]
        if (graph["root"] != descriptor["module"] or root.manifest["version"] != release
                or root.digest != descriptor["module_sha256"]):
            raise ModuleError("wheel identity disagrees with its Tensor package")
        return graph, packages
    except (zipfile.BadZipFile, KeyError, TypeError, UnicodeDecodeError, csv.Error) as exc:
        raise ModuleError(f"invalid Tensor wheel: {exc}") from exc


class _Redirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _download(url, limit, *, accept=None):
    _url(url)
    headers = {"User-Agent": "tensor/0.1.0", "Accept-Encoding": "identity"}
    if accept:
        headers["Accept"] = accept
    try:
        with build_opener(_Redirects()).open(Request(url, headers=headers), timeout=30) as response:
            _url(response.url)
            data = response.read(limit + 1)
            if len(data) > limit:
                raise ModuleError("registry response exceeds size limit")
            return data, response.headers.get_content_type(), response.url
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        raise ModuleError(f"registry download failed: {exc}") from exc


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.files = []

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attrs = dict(attrs)
        href = attrs.get("href")
        if not href:
            return
        parts = urlsplit(href)
        digest = parse_qs(parts.fragment).get("sha256", [None])[0]
        self.files.append({"filename": unquote(parts.path.rsplit("/", 1)[-1]), "url": href,
                           "hashes": {"sha256": digest}, "yanked": "data-yanked" in attrs})


def fetch(name, version, *, index=PYPI_INDEX, wheel_sha256=None):
    """Fetch an exact Tensor wheel through the JSON or HTML Simple Index API."""
    name, index = distribution_name(name), index_url(index)
    if not isinstance(version, str) or not VERSION.fullmatch(version):
        raise ModuleError("registry versions must be exact major.minor.patch")
    if wheel_sha256 is not None and (not isinstance(wheel_sha256, str) or not HASH.fullmatch(wheel_sha256)):
        raise ModuleError("invalid pinned wheel hash")
    data, content_type, base = _download(index + name + "/", INDEX_BYTES,
        accept="application/vnd.pypi.simple.v1+json, application/vnd.pypi.simple.v1+html;q=0.2, text/html;q=0.1")
    if content_type in {"application/vnd.pypi.simple.v1+json", "application/json"}:
        page = _json(data)
        if (not isinstance(page, dict) or not isinstance(page.get("meta"), dict)
                or not str(page["meta"].get("api-version", "")).startswith("1.")
                or not isinstance(page.get("name"), str) or distribution_name(page["name"]) != name
                or not isinstance(page.get("files"), list)):
            raise ModuleError("invalid Simple Index JSON response")
        files = page["files"]
    elif content_type in {"text/html", "application/vnd.pypi.simple.v1+html"}:
        links = _Links()
        try:
            links.feed(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ModuleError("invalid Simple Index HTML response") from exc
        files = links.files
    else:
        raise ModuleError("unsupported Simple Index response content type")
    if len(files) > MAX_FILES or any(not isinstance(item, dict) for item in files):
        raise ModuleError("invalid or excessive Simple Index file list")
    filename = _filename(name, version)
    candidates = [item for item in files if item.get("filename") == filename
                  and (not item.get("yanked", False) or wheel_sha256 is not None)]
    if len(candidates) != 1:
        raise ModuleError(f"need one unambiguous Tensor data wheel for {name}=={version} (new installs exclude yanked releases)")
    item = candidates[0]
    digest = item.get("hashes", {}).get("sha256") if isinstance(item.get("hashes"), dict) else None
    if not isinstance(digest, str) or not HASH.fullmatch(digest) or not isinstance(item.get("url"), str):
        raise ModuleError("registry wheel needs a download URL and SHA-256")
    if wheel_sha256 is not None and digest != wheel_sha256:
        raise ModuleError("registry wheel hash disagrees with the project pin")
    url = urljoin(base, item["url"])
    parts = urlsplit(url)
    url = urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))
    wheel, _, _ = _download(url, MAX_BYTES)
    if hashlib.sha256(wheel).hexdigest() != digest:
        raise ModuleError("downloaded wheel SHA-256 mismatch")
    graph, packages = read_wheel(io.BytesIO(wheel), distribution=name, version=version)
    return graph, packages, {"index": index, "distribution": name, "sha256": digest}


def publish(source=".", *, out_dir="dist", distribution=None, cache_dir=None,
            dry_run=False, repository="pypi", repository_url=None):
    """Prepare a wheel and optionally upload it using Twine's authentication."""
    if repository not in {"pypi", "testpypi"}:
        raise ModuleError("repository must be pypi or testpypi; use repository_url for a custom index")
    if repository_url is not None:
        _url(repository_url)
    result = build_wheel(source, out_dir=out_dir, distribution=distribution, cache_dir=cache_dir)
    destination = repository_url or ("https://upload.pypi.org/legacy/" if repository == "pypi"
                                     else "https://test.pypi.org/legacy/")
    result.update({"dry_run": dry_run, "repository": destination})
    if dry_run:
        return result
    if importlib.util.find_spec("twine") is None:
        raise ModuleError(f"wheel prepared at {result['path']}; upload requires tensor-workspace[publish] or Twine")
    commands = [[sys.executable, "-m", "twine", "check", "--strict", result["path"]],
                [sys.executable, "-m", "twine", "upload", "--non-interactive",
                 "--repository-url", destination, result["path"]]]
    for command in commands:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ModuleError(f"Twine failed; prepared wheel is retained at {result['path']}") from exc
        if completed.returncode:
            # Twine can echo configured credentials/URLs on failure; do not relay them.
            raise ModuleError(f"Twine {command[3]} failed (exit {completed.returncode}); wheel retained at {result['path']}. Check Twine credentials and repository settings.")
    result["status"] = "published"
    return result
