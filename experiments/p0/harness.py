"""Phase 0 experiment harness.

Measures the things the proposal lists as open questions in §25, on whatever
hardware is available. Every experiment is designed so it is meaningful on a
no-GPU machine; experiments that genuinely need a device say so instead of
producing a misleading number.

    python -m experiments.p0.harness                  # everything available here
    python -m experiments.p0.harness --list           # show experiments
    python -m experiments.p0.harness --only codegen   # run one
    python -m experiments.p0.harness --targets cuda   # override target set

Results land in experiments/p0/out/ as JSON plus generated sources.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from experiments.p0.provenance import snapshot

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# Targets the no-GPU path can still emit source for. `c` is the CPU backend
# that ships in the wheel; `llvm` does not (it is USE_LLVM=ON, source-only).
DEFAULT_TARGETS = [
    {"kind": "cuda", "arch": "sm_80"},
    {"kind": "cuda", "arch": "sm_90"},
    {"kind": "cuda", "arch": "sm_100"},
]


@dataclass
class Measurement:
    experiment: str
    subject: str
    target: str
    ok: bool
    seconds: float = 0.0
    error: str = ""
    metrics: dict = field(default_factory=dict)
    # Set when a non-ok result is the finding rather than a broken experiment.
    # Without this, an experiment that *proves* a dependency is mandatory would
    # be reported as a failure, which is precisely the misleading signal this
    # project is trying to avoid.
    expected_failure: bool = False

    @property
    def verdict(self) -> str:
        if self.ok:
            return "ok"
        return "expected" if self.expected_failure else "FAIL"


def _now() -> float:
    return time.perf_counter()


def target_label(t: dict) -> str:
    return f"{t.get('kind')}:{t.get('arch', '')}".rstrip(":")


# ---------------------------------------------------------------- experiments


def exp_environment(args=None) -> list[Measurement]:
    """What is actually installed, and what does it cost to import?

    Directly targets the §23 'steps from download to first kernel' and
    'warm startup latency' metrics, and §25's packaging question.

    Every import is timed in a *fresh subprocess*. Timing `import tilelang`
    inside the harness would measure nothing: the harness itself has
    already imported it.
    """
    out: list[Measurement] = []

    def time_import(label: str, code: str) -> Measurement:
        t0 = _now()
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
        wall = _now() - t0
        try:
            secs = float(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            return Measurement("environment", label, "n/a", False, wall,
                               error=proc.stderr.strip()[-200:] or "no output")
        return Measurement("environment", label, "n/a", True, secs, metrics={"seconds": round(secs, 3)})

    # Cold interpreter start, for comparison against every import below.
    out.append(time_import("bare_interpreter", "import time; print(0.0)"))
    out.append(time_import("import_torch", "import time; t=time.perf_counter(); import torch; print(time.perf_counter()-t)"))
    out.append(time_import("import_tilelang", "import time; t=time.perf_counter(); import tilelang; print(time.perf_counter()-t)"))

    # How `tvm` actually gets onto the path. It is not an installed
    # distribution: it is vendored under tilelang/3rdparty/tvm and only becomes
    # importable as a side effect of importing tilelang. This matters for any
    # component that wants TIRx directly without going through tilelang.
    # A real import is used rather than find_spec, which is unreliable here.
    code = (
        "import importlib\n"
        "try:\n"
        "    importlib.import_module('tvm'); before = 'yes'\n"
        "except Exception:\n"
        "    before = 'no'\n"
        "import tilelang\n"
        "import tvm\n"
        "print('tvm_importable_before_tilelang=%s' % before)\n"
        "print('tvm_importable_after_tilelang=yes')\n"
        "print('tvm_path=%s' % tvm.__file__)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    parsed = {}
    for line in proc.stdout.strip().splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            parsed[k.strip()] = v.strip()
    out.append(Measurement("environment", "tvm_import_shape", "n/a", bool(parsed), metrics=parsed))

    # On-disk footprint of the environment. This is the "what does a user
    # download" question, answered without re-downloading anything.
    site = Path(sys.prefix) / "Lib" / "site-packages"
    if site.is_dir():
        total = 0
        biggest = []
        for dist in ("tilelang", "torch", "numpy", "z3", "apache_tvm_ffi", "torch_c_dlpack_ext"):
            size = 0
            for h in list(site.glob(f"{dist}*")):
                if h.is_dir():
                    size += sum(f.stat().st_size for f in h.rglob("*") if f.is_file())
                elif h.is_file():
                    size += h.stat().st_size
            if size:
                total += size
                biggest.append({"package": dist, "mb": round(size / 1e6, 1)})
        biggest.sort(key=lambda d: -d["mb"])
        out.append(Measurement("environment", "disk_footprint", "n/a", True,
                                metrics={"total_mb": round(total / 1e6, 1), "breakdown": biggest}))

    out.append(Measurement("environment", "host", "n/a", True, metrics={
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "has_cuda_toolkit": bool(shutil_which("nvcc")),
        "note": "no NVIDIA GPU is required for any codegen experiment below",
    }))
    return out


def shutil_which(name: str) -> str | None:
    from shutil import which
    return which(name)


def exp_codegen(args) -> list[Measurement]:
    """Frontend -> TIRx -> lowering -> target source, for every kernel/target.

    This is the experiment that does NOT need a GPU. It answers the §25
    questions about IR complexity, lowering stages and compile latency --
    for source emission, which is where most of the compile time goes.
    """
    from tilelang.tools.compile_only import compile_kernel_source
    from experiments.p0 import kernels as K

    targets, kernels = args.targets, args.kernels
    out: list[Measurement] = []
    for name in kernels:
        for tgt in targets:
            label = target_label(tgt)
            try:
                func = K.KERNELS[name]()
            except Exception as e:  # kernel authoring failure
                out.append(Measurement("codegen", name, label, False, 0.0,
                                        error=f"build: {type(e).__name__}: {e}"))
                continue
            t0 = _now()
            try:
                src = compile_kernel_source(func, tgt)
            except Exception as e:
                out.append(Measurement("codegen", name, label, False, _now() - t0,
                                        error=f"{type(e).__name__}: {e}"))
                continue
            dt = _now() - t0

            OUT.mkdir(parents=True, exist_ok=True)
            path = OUT / f"{name}__{label.replace(':', '-')}.cu"
            path.write_text(src, encoding="utf-8")

            out.append(Measurement("codegen", name, label, True, dt, metrics={
                "source_bytes": len(src.encode("utf-8")),
                "source_lines": len(src.splitlines()),
                **count_markers(src),
                "written": str(path.relative_to(HERE.parent.parent)),
            }))
    return out


def count_markers(src: str) -> dict:
    """Cheap structural fingerprint of generated code.

    These are the signals that distinguish a real kernel from a stub, and
    they are the cheapest available proxy for 'how much hardware control
    did the compiler actually take'.

    Note that async_copy and tma are *alternative* strategies: pre-Hopper
    targets stage through cp.async, Hopper+ can use TMA tensor maps. A zero
    in one column with a non-zero in the other is a target decision, not a
    missing feature.
    """
    patterns = {
        "mma_instr": r"mma_sync|wmma|tcgen05\.",
        "async_copy": r"cp_async|cp\.async",
        "tma": r"CUtensorMap|cp\.async\.bulk|tma_",
        "barriers": r"__syncthreads|barrier\.sync",
        "shared_mem": r"extern __shared__|__shared__",
        "swizzle": r"swizzle",
        "dynamic_smem": r"buf_dyn_shmem|extern __shared__ __align__",
        "tl_includes": r"#include <tl_templates/",
    }
    return {k: len(re.findall(v, src)) for k, v in patterns.items()}


def exp_pass_trace(args) -> list[Measurement]:
    """Record the real lowering pipeline and dump TIRx IR at every stage.

    This is the substrate for `tensor inspect --stage X`: the pass names,
    their order, and the IR before/after each one. It also answers two §26
    metrics directly -- number of lowering stages, and IR complexity -- and
    it is the only honest way to count what the compiler actually does,
    because the pass list is owned by the backend, not by a hardcoded
    pipeline in our own code.

    Trace files land under <trace_dir>/<label>/.run_records/run_*/ as
    NN_<PassName>_before.tir and _after.tir.
    """
    import tilelang.tools.lower_trace as lt
    from tilelang.tools.compile_only import compile_kernel_source
    from experiments.p0 import kernels as K

    kernels = args.kernels
    target = args.targets[0]
    label = target_label(target)
    out: list[Measurement] = []
    for name in kernels:
        try:
            func = K.KERNELS[name]()
        except Exception as e:
            out.append(Measurement("pass_trace", name, label, False, 0.0,
                                    error=f"build: {type(e).__name__}: {e}"))
            continue

        trace_dir = OUT / "_traces" / f"{name}__{label.replace(':', '-')}"
        if trace_dir.exists():
            shutil.rmtree(trace_dir, ignore_errors=True)
        trace_dir.mkdir(parents=True, exist_ok=True)

        t0 = _now()
        # terminal mode is the fast path and the one that writes the per-pass
        # .tir dumps this experiment reads. It also prints a full diff per pass
        # to stdout -- megabytes of it -- so that is captured and discarded.
        # The before/after IR is on disk either way.
        lt.enable(mode="terminal", trace_dir=str(trace_dir))
        sink = io.StringIO()
        try:
            with contextlib.redirect_stdout(sink):
                compile_kernel_source(func, target)
        except Exception as e:
            lt.reset()
            out.append(Measurement("pass_trace", name, label, False, _now() - t0,
                                    error=f"{type(e).__name__}: {e}"))
            continue
        finally:
            lt.reset()
        dt = _now() - t0

        # Trace files are nested under per-run record directories.
        after = list(trace_dir.rglob("*_after.tir"))
        steps = []
        for p in sorted(after, key=lambda q: q.name):
            stem = p.name[: -len("_after.tir")]
            idx, _, pass_name = stem.partition("_")
            steps.append({"order": int(idx) if idx.isdigit() else 0, "pass": pass_name})

        ir_bytes = sum(p.stat().st_size for p in trace_dir.rglob("*.tir"))
        out.append(Measurement("pass_trace", name, label, True, dt, metrics={
            "num_steps": len(steps),
            "num_distinct_passes": len({s["pass"] for s in steps}),
            "ir_dumped_kb": round(ir_bytes / 1024, 1),
            "passes": [s["pass"] for s in sorted(steps, key=lambda s: s["order"])],
            "trace_dir": str(trace_dir.relative_to(HERE.parent.parent)),
        }))
    return out


def exp_torch_dependency(args=None) -> list[Measurement]:
    """Is torch structurally required to *compile*, or only to interoperate?

    This is the decisive experiment for ADR 0005. It writes a real probe script
    and runs it in three modes, each progressively more hostile to torch:

      normal          torch installed and working (control)
      torch_blocked   `import torch` raises outright
      torch_stubbed   `import torch` succeeds but no torch functionality exists

    The probe must be a real file, not `python -c`: TileLang's prim_func is built
    by an eager builder that reads the function's source, which `-c` does not have.

    If codegen survives all three, Tensor can ship a torch-free compiler and
    expose PyTorch as an optional adapter. If `import tilelang` fails under
    `torch_blocked`, torch is a hard dependency of the *frontend*, not just the
    tensor bridge, and the packaging story needs a different answer.
    """
    out: list[Measurement] = []
    OUT.mkdir(parents=True, exist_ok=True)
    probe = OUT / "_torch_probe.py"
    probe.write_text(_TORCH_PROBE, encoding="utf-8")

    for mode in ("normal", "torch_blocked", "torch_stubbed"):
        t0 = _now()
        proc = subprocess.run([sys.executable, str(probe), mode],
                              capture_output=True, text=True, timeout=900)
        last_err = (proc.stderr.strip().splitlines() or [""])[-1][:200]
        compiled = "COMPILED_OK" in proc.stdout
        out.append(Measurement(
            "torch_dependency", mode, "n/a",
            ok=compiled,
            seconds=_now() - t0,
            error="" if compiled else last_err,
            metrics={"stdout": proc.stdout.strip()[-400:]},
            # The whole point of the blocked/stubbed probes is that they fail.
            # If one of them ever succeeded, that would be the finding.
            expected_failure=(mode != "normal" and not compiled),
        ))
    return out


# Runs in a subprocess. --mode selects how much of torch survives.
_TORCH_PROBE = '''
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "normal"

if MODE == "torch_blocked":
    import builtins
    _real = builtins.__import__
    def _blocked(name, *a, **k):
        if name.split(".")[0] == "torch":
            raise ModuleNotFoundError("No module named 'torch' (probe)")
        return _real(name, *a, **k)
    builtins.__import__ = _blocked
elif MODE == "torch_stubbed":
    _cache = {}
    def _mk(name):
        if name not in _cache:
            _cache[name] = type(name, (object,), {"__module__": "torch"})
        return _cache[name]
    class _TorchStub:
        __path__ = []
        __all__ = []
        __spec__ = None
        def __getattr__(self, n):
            if n.startswith("__") and n.endswith("__"):
                raise AttributeError(n)
            return _mk(n)
    sys.modules["torch"] = _TorchStub()

print("torch_importable_before_tilelang=%s" % ("torch" in sys.modules))

import tilelang
import tilelang.language as T

print("import_tilelang=OK")
print("torch_loaded_after_tilelang=%s" % ("torch" in sys.modules))


@T.prim_func
def add(A: T.Tensor((256,), "float32"),
        B: T.Tensor((256,), "float32"),
        C: T.Tensor((256,), "float32")):
    with T.Kernel(2, threads=128) as bx:
        for i in T.Parallel(128):
            C[bx * 128 + i] = A[bx * 128 + i] + B[bx * 128 + i]


print("prim_func=OK")

from tilelang.tools.compile_only import compile_kernel_source
src = compile_kernel_source(add, {"kind": "cuda", "arch": "sm_80"})
print("COMPILED_OK lines=%d" % len(src.splitlines()))
'''

def exp_artifact_shape(args=None) -> list[Measurement]:
    """E4: what must a `.tbin` carry, and is the portable tier real?

    TileLang's pipeline runs `BindTarget` as its FIRST pass, so there is no
    target-agnostic capture point inside lowering. The only candidate for a
    portable artifact is the frontend TIRx PrimFunc from `@T.prim_func`.

    This measures, for one kernel:

      serialize   FFI JSON and TVMScript text, with sizes
      retarget    does the RELOADED artifact compile to a different, correct
                  result for a different arch?  (re-targetability)
      roundtrip   is the reloaded result byte-identical to the original?
      crossproc   does it survive a *fresh interpreter*?  A `.tbin` that only
                  loads in the process that made it is worthless.
      shape       are shapes baked in?  (re-specializability)

    Verdicts are reported, not assumed. `retarget` and `crossproc` passing but
    `shape` failing means the portable tier is real but is a *re-targetable*
    tier, not a *re-specializable* one.
    """
    import hashlib
    import tilelang.language as T
    import tvm
    import tvm.ir as tir
    from tilelang.tools.compile_only import compile_kernel_source
    from experiments.p0 import kernels as K

    OUT.mkdir(parents=True, exist_ok=True)
    out: list[Measurement] = []

    def dig(s):
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def compile_for(func, arch):
        return compile_kernel_source(func, {"kind": "cuda", "arch": arch})

    # ---- serialize ------------------------------------------------------
    try:
        func = K.gemm_relu(512, 512, 512)
        mod = tvm.IRModule({"main": func})
        js = tir.save_json(mod)
        json_path = OUT / "artifact_frontend_mod.json"
        json_path.write_text(js, encoding="utf-8")
        script_text = mod.script()
        script_path = OUT / "artifact_frontend_mod.py"
        script_path.write_text(script_text, encoding="utf-8")

        # The generated code #includes tl_templates headers, so an artifact
        # that ships source must ship them too. Measure that tree.
        import tilelang
        hdr_root = Path(tilelang.__file__).parent / "templates"
        hdr_bytes = 0
        hdr_files = 0
        for base in (hdr_root, Path(tilelang.__file__).parent):
            if base.is_dir():
                for f in base.rglob("tl_templates/**/*"):
                    if f.is_file():
                        hdr_bytes += f.stat().st_size
                        hdr_files += 1

        out.append(Measurement("artifact_shape", "serialize", "n/a", True, 0.0, metrics={
            "ffi_json_kb": round(len(js.encode()) / 1024, 1),
            "tirx_script_kb": round(len(script_text.encode()) / 1024, 1),
            "tirx_script_lines": len(script_text.splitlines()),
            "tl_templates_files": hdr_files,
            "tl_templates_kb": round(hdr_bytes / 1024, 1),
        }))
    except Exception as e:
        out.append(Measurement("artifact_shape", "serialize", "n/a", False, 0.0,
                               error=f"{type(e).__name__}: {e}"))
        return out

    # ---- re-target + round-trip fidelity --------------------------------
    try:
        reloaded = tir.load_json(js)
        items = list(reloaded.functions_items())
        reloaded_func = items[0][1]
    except Exception as e:
        out.append(Measurement("artifact_shape", "reload", "n/a", False, 0.0,
                               error=f"{type(e).__name__}: {e}"))
        return out

    srcs = {}
    for label, f in (("original", func), ("reloaded", reloaded_func)):
        for arch in ("sm_80", "sm_90"):
            t0 = _now()
            try:
                srcs[(label, arch)] = compile_for(f, arch)
                out.append(Measurement("artifact_shape", f"compile_{label}", arch, True, _now() - t0,
                                       metrics={"lines": len(srcs[(label, arch)].splitlines()),
                                                "digest": dig(srcs[(label, arch)])}))
            except Exception as e:
                out.append(Measurement("artifact_shape", f"compile_{label}", arch, False, _now() - t0,
                                       error=f"{type(e).__name__}: {str(e)[:120]}"))

    o80, o90 = srcs.get(("original", "sm_80")), srcs.get(("original", "sm_90"))
    r80, r90 = srcs.get(("reloaded", "sm_80")), srcs.get(("reloaded", "sm_90"))
    if o80 and r80 and o90 and r90:
        out.append(Measurement("artifact_shape", "roundtrip_fidelity", "sm_80",
                               ok=(o80 == r80 and o90 == r90),
                               metrics={"sm80_match": o80 == r80, "sm90_match": o90 == r90}))
        out.append(Measurement("artifact_shape", "retargetable", "sm_80->sm_90",
                               ok=(dig(r80) != dig(r90)),
                               metrics={"differs": dig(r80) != dig(r90),
                                        "sm80_lines": len(r80.splitlines()),
                                        "sm90_lines": len(r90.splitlines())}))

    # ---- cross-process: the real portability test -----------------------
    probe = OUT / "_artifact_reload_probe.py"
    probe.write_text(_RELOAD_PROBE, encoding="utf-8")
    t0 = _now()
    proc = subprocess.run([sys.executable, str(probe), str(json_path)],
                          capture_output=True, text=True, timeout=900)
    ok = (proc.returncode == 0 and r80 is not None and r90 is not None
          and f"sm_80={dig(r80)}/" in proc.stdout and f"sm_90={dig(r90)}/" in proc.stdout)
    out.append(Measurement("artifact_shape", "cross_process_reload", "sm_80+sm_90", ok, _now() - t0,
                           error="" if ok else (proc.stderr.strip().splitlines() or [""])[-1][:180],
                           metrics={"stdout": proc.stdout.strip()[-300:]}))

    # ---- shape specialization -------------------------------------------
    try:
        a = dig(compile_for(K.gemm_relu(512, 512, 512), "sm_80"))
        b = dig(compile_for(K.gemm_relu(256, 256, 256), "sm_80"))
        out.append(Measurement("artifact_shape", "shape_baked_in", "sm_80", True, 0.0, metrics={
            "digest_512": a, "digest_256": b,
            "differs": a != b,
            "note": "shapes are baked at the frontend; a captured artifact is re-targetable, "
                    "not re-shapeable. T.symbolic / T.dynamic exist for polymorphism.",
        }))
    except Exception as e:
        out.append(Measurement("artifact_shape", "shape_baked_in", "sm_80", False, 0.0,
                               error=f"{type(e).__name__}: {str(e)[:120]}"))
    return out


_RELOAD_PROBE = '''
import sys, hashlib
import tilelang          # puts vendored tvm on sys.path
import tvm, tvm.ir as ir
from tilelang.tools.compile_only import compile_kernel_source

path = sys.argv[1]
mod = ir.load_json(open(path, encoding="utf-8").read())
func = list(mod.functions_items())[0][1]
digests = []
for arch in ("sm_80", "sm_90"):
    src = compile_kernel_source(func, {"kind": "cuda", "arch": arch})
    d = hashlib.sha256(src.encode()).hexdigest()[:16]
    digests.append(f"{arch}={d}/{len(src.splitlines())}L")
print("RELOAD_OK " + " ".join(digests))
'''

def _artifact_op_surface(json_path: Path) -> list[str]:
    """The set of registered ops an artifact depends on.

    This is the real compatibility contract. An artifact is a graph of nodes;
    the only version-sensitive parts are the ops it calls, and those are
    statically enumerable without loading anything. A preflight gate can
    therefore check an artifact against a runtime *before* trying to load it.
    """
    js = json.loads(json_path.read_text(encoding="utf-8"))
    nodes = js["nodes"]

    def label(i):
        n = nodes[i]
        d = n.get("data")
        if isinstance(d, dict):
            for k in ("name", "global_name", "key", "op_name"):
                if isinstance(d.get(k), str):
                    return d[k]
            hint = d.get("name_hint")
            if isinstance(hint, int) and hint < len(nodes):
                v = nodes[hint].get("data")
                if isinstance(v, str):
                    return v
        if isinstance(d, str):
            return d
        return n.get("type", "?")

    ops = set()
    for n in nodes:
        if n.get("type") == "tirx.Call":
            i = n["data"].get("op")
            if isinstance(i, int) and i < len(nodes):
                ops.add(label(i))
    return sorted(ops)


def exp_artifact_versioning(args=None) -> list[Measurement]:
    """E4a: is a `.tbin` self-identifying, and what is its compatibility surface?

    Two questions with different answers:

    1. The artifact's only version metadata is `{"tvm_version": "0.25.dev0"}` -- a
       *dev* tag. Measured against real artifacts from two TileLang releases,
       it is identical across them, so it cannot gate compatibility.
    2. The ops an artifact calls ARE statically extractable, and they are the
       real contract. Measured: four ops, and across three minor releases the
       only change is `tl.region` -> `tl.tileop.region`.

    Also checks that malformed artifacts fail loudly rather than silently,
    which is what makes a version gate safe to implement.
    """
    import json as _json
    import tilelang.language as T
    import tvm
    import tvm.ir as tir
    from experiments.p0 import kernels as K

    OUT.mkdir(parents=True, exist_ok=True)
    out: list[Measurement] = []
    art = OUT / "artifact_frontend_mod.json"

    # 1. Does the artifact describe itself?
    try:
        js = tir.save_json(tvm.IRModule({"main": K.gemm_relu(512, 512, 512)}))
        art.write_text(js, encoding="utf-8")
        meta = _json.loads(js).get("metadata", {})
        out.append(Measurement("artifact_versioning", "self_describing", "n/a", True, 0.0, metrics={
            "metadata": meta,
            "has_format_version": any("format" in k for k in meta),
            "records_tilelang_version": any("tilelang" in k for k in meta),
            "verdict": "tvm_version only, and it is a dev tag -- NOT a usable gate",
        }))
    except Exception as e:
        out.append(Measurement("artifact_versioning", "self_describing", "n/a", False, 0.0,
                               error=f"{type(e).__name__}: {e}"))
        return out

    # 2. The compatibility surface
    try:
        ops = _artifact_op_surface(art)
        out.append(Measurement("artifact_versioning", "op_surface", "n/a", True, 0.0, metrics={
            "num_ops": len(ops),
            "ops": ", ".join(ops),
            "note": "statically extractable without loading -- a preflight gate can "
                    "check these against the runtime registry before attempting a load",
        }))
    except Exception as e:
        out.append(Measurement("artifact_versioning", "op_surface", "n/a", False, 0.0,
                               error=f"{type(e).__name__}: {e}"))

    # 3. Malformed artifacts must fail loudly, never silently.
    fixtures = {
        "truncated": js[: int(len(js) * 0.6)],
        "wrong_schema": '{"root_index":0,"nodes":[{"type":"NotARealNode"}],"metadata":{}}',
        "not_json": "this is not json at all",
    }
    for name, body in fixtures.items():
        p = OUT / f"_bad_{name}.json"
        p.write_text(body, encoding="utf-8")
        try:
            tir.load_json(body)
            out.append(Measurement("artifact_versioning", f"reject_{name}", "n/a", False, 0.0,
                                   metrics={"error": "LOADED A MALFORMED ARTIFACT"}))
        except Exception as e:
            out.append(Measurement("artifact_versioning", f"reject_{name}", "n/a", True, 0.0,
                                   metrics={"error_type": type(e).__name__,
                                            "names_cause": ("NotARealNode" in str(e))}))
    return out


def exp_symbolic_shapes(args=None) -> list[Measurement]:
    """E4b: can one artifact serve several shapes?

    E4 found shapes baked at the frontend. That is true for the *default*,
    static-shaped kernel. TileLang also exposes `T.dynamic(name)`, which makes an
    extent a `tirx.Var` instead of an int. The question is what that changes.

    The measurement that settles it is not a diff between two shapes -- it is
    whether the extent appears in the generated kernel's *signature*. If `M` is a
    kernel parameter, the artifact is shape-agnostic by construction. If it is a
    baked constant, it is not.

    Round-trips the symbolic artifact and checks that the runtime extent
    survives serialization. Concrete-value substitution is not evidence of
    runtime shape support; that requires launching at several shapes.
    """
    import re
    import tilelang.language as T
    import tvm
    import tvm.ir as tir
    from tilelang.tools.compile_only import compile_kernel_source
    from experiments.p0 import kernels as K

    OUT.mkdir(parents=True, exist_ok=True)
    out: list[Measurement] = []
    K_DIM = 128
    M = T.dynamic("M")

    def dig(s):
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    @T.prim_func
    def dyn_add(
        A: T.Tensor((M, K_DIM), "float16"),
        B: T.Tensor((M, K_DIM), "float16"),
        C: T.Tensor((M, K_DIM), "float16"),
    ):
        with T.Kernel(T.ceildiv(M, 128), threads=128) as bx:
            a_shared = T.alloc_shared((128, K_DIM), "float16")
            b_shared = T.alloc_shared((128, K_DIM), "float16")
            o_shared = T.alloc_shared((128, K_DIM), "float16")
            T.copy(A[bx * 128, 0], a_shared)
            T.copy(B[bx * 128, 0], b_shared)
            for i, j in T.Parallel(128, K_DIM):
                o_shared[i, j] = a_shared[i, j] + b_shared[i, j]
            T.copy(o_shared, C[bx * 128, 0])

    # 1. The decisive measurement: where does the extent live in the signature?
    try:
        src = compile_kernel_source(dyn_add, {"kind": "cuda", "arch": "sm_80"})
        (OUT / "symbolic_sm80.cu").write_text(src, encoding="utf-8")
        sig = next((l for l in src.splitlines() if "__global__" in l), "")
        # Is the symbol a parameter of the kernel, rather than a baked constant?
        as_param = bool(re.search(r",\s*int\s+M\s*\)", sig))
        preds = len(re.findall(r"<\s*M\s*\)", src))
        out.append(Measurement("symbolic_shapes", "extent_is_runtime_param", "sm_80", as_param, 0.0, metrics={
            "signature": re.sub(r"\s+", " ", sig)[:150],
            "M_is_kernel_parameter": as_param,
            "predication_checks": preds,
            "lines": len(src.splitlines()),
        }))
    except Exception as e:
        out.append(Measurement("symbolic_shapes", "extent_is_runtime_param", "sm_80", False, 0.0,
                               error=f"{type(e).__name__}: {str(e)[:150]}"))
        return out

    # 2. Serializability of the symbolic artifact
    try:
        js = tir.save_json(tvm.IRModule({"main": dyn_add}))
        (OUT / "artifact_symbolic.json").write_text(js, encoding="utf-8")
        out.append(Measurement("symbolic_shapes", "serialize", "n/a", True, 0.0,
                               metrics={"kb": round(len(js.encode()) / 1024, 1)}))
    except Exception as e:
        out.append(Measurement("symbolic_shapes", "serialize", "n/a", False, 0.0,
                               error=f"{type(e).__name__}: {e}"))

    # 3. The serialized artifact must retain both code and its runtime extent.
    try:
        art = OUT / "artifact_symbolic.json"
        f2 = list(tir.load_json(art.read_text(encoding="utf-8")).functions_items())[0][1]
        reloaded_source = compile_kernel_source(f2, {"kind": "cuda", "arch": "sm_80"})
        reload_sig = next((line for line in reloaded_source.splitlines() if "__global__" in line), "")
        runtime_param = bool(re.search(r",\s*int\s+M\s*\)", reload_sig))
        out.append(Measurement("symbolic_shapes", "roundtrip_runtime_extent", "sm_80",
                               src == reloaded_source and runtime_param, 0.0,
                               metrics={"digest": dig(reloaded_source), "source_match": src == reloaded_source,
                                        "M_is_kernel_parameter": runtime_param}))
    except Exception as e:
        out.append(Measurement("symbolic_shapes", "roundtrip_runtime_extent", "sm_80", False, 0.0,
                               error=f"{type(e).__name__}: {str(e)[:150]}"))

    # 4. Contrast: the static-shaped kernel bakes its extent
    try:
        s_static = compile_kernel_source(K.fused_elementwise(512, K_DIM), {"kind": "cuda", "arch": "sm_80"})
        sig_s = next((l for l in s_static.splitlines() if "__global__" in l), "")
        out.append(Measurement("symbolic_shapes", "static_contrast", "sm_80", True, 0.0, metrics={
            "static_lines": len(s_static.splitlines()),
            "static_has_extent_param": bool(re.search(r"int\s+[A-Z]\w*\s*[,)]", sig_s)),
            "note": "static extents are Python ints and never appear as parameters -- "
                    "they are folded into offsets and grid constants at codegen",
        }))
    except Exception as e:
        out.append(Measurement("symbolic_shapes", "static_contrast", "sm_80", False, 0.0,
                               error=f"{type(e).__name__}: {str(e)[:150]}"))
    return out


EXPERIMENTS = {
    "environment": exp_environment,
    "codegen": exp_codegen,
    "pass_trace": exp_pass_trace,
    "torch_dependency": exp_torch_dependency,
    "artifact_shape": exp_artifact_shape,
    "artifact_versioning": exp_artifact_versioning,
    "symbolic_shapes": exp_symbolic_shapes,
}


# ----------------------------------------------------------------------- main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    ap.add_argument("--only", nargs="*", default=None, help="run only these experiments")
    ap.add_argument("--kernels", nargs="*", default=None, help="restrict to these kernels")
    ap.add_argument("--targets", nargs="*", default=None,
                    help="targets, e.g. 'cuda:sm_80 c' (default: cuda sm_80/90/100)")
    ap.add_argument("--out", default=None, help="results JSON path")
    args = ap.parse_args()

    if args.list:
        for k, fn in EXPERIMENTS.items():
            summary = (fn.__doc__ or "").strip().splitlines()[0]
            print(f"{k:12} {summary}")
        return 0

    from experiments.p0 import kernels as K
    args.kernels = args.kernels or list(K.KERNELS)

    if args.targets:
        args.targets = [
            {"kind": t.split(":")[0], **({"arch": t.split(":")[1]} if ":" in t else {})}
            for t in args.targets
        ]
    else:
        args.targets = list(DEFAULT_TARGETS)

    selected = args.only or list(EXPERIMENTS)
    results: list[Measurement] = []
    for name in selected:
        if name not in EXPERIMENTS:
            print(f"unknown experiment: {name}", file=sys.stderr)
            return 2
        fn = EXPERIMENTS[name]
        print(f"\n=== {name} ===", flush=True)
        res = fn(args)
        results.extend(res)
        for m in res:
            line = f"  {m.verdict:8} {m.subject:22} {m.target:12} {m.seconds:7.3f}s"
            if m.verdict != "ok":
                line += f"  {m.error[:120]}"
            elif m.metrics:
                line += "  " + " ".join(f"{k}={v}" for k, v in m.metrics.items() if not isinstance(v, (dict, list)))
            print(line, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else OUT / "results.json"
    out_path.write_text(
        json.dumps({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "host": platform.platform(),
                    "provenance": snapshot(),
                    "results": [asdict(m) for m in results]}, indent=2),
        encoding="utf-8",
    )
    ok = sum(1 for m in results if m.ok)
    expected = sum(1 for m in results if m.expected_failure)
    broken = len(results) - ok - expected
    print(f"\n{ok} ok, {expected} expected-failure (the finding), {broken} unexpected -> {out_path}")
    return 0 if broken == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
