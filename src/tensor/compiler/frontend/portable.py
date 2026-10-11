"""Restore a verified portable frontend for compilation."""

from __future__ import annotations

from pathlib import Path
import runpy


def export_spec(path, checked=None):
    from tensor.artifacts.portable import preflight
    from tensor.compiler.errors import BuildError

    path = Path(path)
    if path.suffix == ".py":
        # Packaged helpers live beside the export, not in the invoking project.
        import sys

        original = sys.path[:]
        bytecode = sys.dont_write_bytecode
        local = {p.stem for p in path.resolve().parent.glob("*.py")} | {
            p.name
            for p in path.resolve().parent.iterdir()
            if p.is_dir() and (p / "__init__.py").is_file()
        }
        saved = {
            name: value
            for name, value in sys.modules.items()
            if name.split(".")[0] in local
        }
        try:
            for name in saved:
                del sys.modules[name]
            sys.dont_write_bytecode = True
            sys.path.insert(0, str(path.resolve().parent))
            namespace = runpy.run_path(str(path.resolve()))
            export = namespace.get("tensor_export")
            if not callable(export):
                raise BuildError("source must define tensor_export()")
            return export()
        finally:
            sys.path[:] = original
            sys.dont_write_bytecode = bytecode
            for name in list(sys.modules):
                if name.split(".")[0] in local:
                    del sys.modules[name]
            sys.modules.update(saved)
    manifest, files = checked or preflight(path)
    import tvm

    for name in manifest["op_set"]:
        try:
            tvm.ir.Op.get(name)
        except Exception as exc:
            raise BuildError(f"portable operator is unavailable: {name}") from exc
    try:
        module = tvm.ir.load_json(files["kernel.tirx.json"].decode())
        if not isinstance(module, tvm.IRModule) or len(module.functions) != 1:
            raise BuildError("portable profile requires one frontend function")
        kernel = next(iter(module.functions.values()))
        if not isinstance(kernel, tvm.tirx.PrimFunc):
            raise BuildError("portable export must be a TIRx PrimFunc")
        return {"kernel": kernel, "outputs": manifest.get("outputs", [])}
    except BuildError:
        raise
    except Exception as exc:
        raise BuildError(f"portable frontend deserialization failed: {exc}") from exc
