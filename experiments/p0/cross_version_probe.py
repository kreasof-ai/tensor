"""E4a cross-version probe.

Loads a serialized TIRx artifact and re-lowers it, in whatever TileLang version
happens to be running. Run it with a different interpreter to answer: does a
`.tbin` survive a TileLang upgrade?

    <python> cross_version_probe.py <artifact.json>

Prints a single machine-readable line starting with XV_RESULT so the caller can
compare digests across versions.

What it deliberately does NOT do: catch a version mismatch and paper over it. If
the load fails, that is the result.
"""

import hashlib
import json
import sys


def main():
    path = sys.argv[1]
    raw = open(path, encoding="utf-8").read()

    # 0. What does the artifact claim about itself?
    try:
        meta = json.loads(raw).get("metadata", {})
    except Exception:
        meta = {"<unparseable>": True}
    print(f"XV_ARTIFACT_META {json.dumps(meta)}")

    # 1. Runtime identity
    try:
        import tilelang
        tl_ver = getattr(tilelang, "__version__", "?")
    except Exception as e:
        print(f"XV_RESULT FAIL import_tilelang {type(e).__name__}: {str(e)[:120]}")
        return
    try:
        import tvm
        tvm_ver = getattr(tvm, "__version__", "?")
    except Exception:
        tvm_ver = "?"

    # Does this runtime even HAVE the IR the artifact was written in?
    has_tirx = False
    try:
        import tvm.tirx  # noqa: F401
        has_tirx = True
    except Exception:
        pass
    print(f"XV_RUNTIME tilelang={tl_ver} tvm={tvm_ver} tirx_available={has_tirx}")

    if not has_tirx:
        print("XV_RESULT FAIL no_tirx_in_runtime")
        return

    # 2. Load
    try:
        import tvm.ir as ir
        mod = ir.load_json(raw)
        fn = list(mod.functions_items())[0][1]
    except Exception as e:
        print(f"XV_RESULT FAIL load {type(e).__name__}: {str(e)[:160]}")
        return

    # 3. Re-lower. A load that succeeds but lowers differently is WORSE than a
    #    load that fails, so the digest is reported, not just the success.
    try:
        from tilelang.tools.compile_only import compile_kernel_source
        out = []
        for arch in ("sm_80", "sm_90"):
            try:
                src = compile_kernel_source(fn, {"kind": "cuda", "arch": arch})
                d = hashlib.sha256(src.encode()).hexdigest()[:16]
                out.append(f"{arch}={d}/{len(src.splitlines())}L")
            except Exception as e:
                out.append(f"{arch}=ERR:{type(e).__name__}")
        print(f"XV_RESULT OK {' '.join(out)}")
    except Exception as e:
        print(f"XV_RESULT FAIL lower {type(e).__name__}: {str(e)[:160]}")


if __name__ == "__main__":
    main()
