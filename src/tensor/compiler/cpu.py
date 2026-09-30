"""Scoped Linux CPU producer reusing TileLang's C lowering and Tensor ABI 1.

The CPU provider is a contract-validation implementation, with no claim of
optimized CPU scheduling. It deliberately uses its own native compiler.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
import zipfile

from tensor.runtime.abi import DTYPES, runtime_requirement, workspace_requirement
from tensor.artifacts.format import FORMAT, FORMAT_VERSION, validate_manifest
from tensor.compiler.build import BuildError, _cached_image, _cache_root, _header_hash, _notices, _op_set, _write_cache
from tensor.cli.doctor import check_packages
from tensor.runtime.signature import SCALAR_TYPES

C_TYPES = {"bool": "bool", "float32": "float", "float64": "double",
           **{name: name + "_t" for name in ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")}}


def _wrapper(entrypoint, arguments, abi, symbols):
    indices = {item["name"]: index for index, item in enumerate(abi)}
    frontend = {item["name"]: item for item in arguments}
    lines = ['#include <tensor/abi.h>', '#include <cstring>', '#include <cstdio>', '#include <cmath>',
        'static int32_t fail(TensorErrorV1* e, int32_t code, const char* msg) {',
        '  if(e) { e->code=code; std::snprintf(e->message,sizeof(e->message),"%s",msg); } return code;', '}',
        'extern "C" int32_t tensor_kernel_v1(const TensorCallV1* call, TensorErrorV1* error) {',
        '  if(error) { error->code=0; error->message[0]=0; }',
        '  if(!call || call->abi_version!=TENSOR_ABI_VERSION || call->struct_size<sizeof(TensorCallV1))',
        '    return fail(error,TENSOR_ERROR_ABI,"unsupported call ABI");',
        f'  if(call->flags || call->argument_count!={len(abi)} || !call->arguments)',
        '    return fail(error,TENSOR_ERROR_ARGUMENT,"invalid argument count or flags");',
        '  if(call->stream.device_type!=TENSOR_DEVICE_CPU || call->stream.device_ordinal || call->stream.handle)',
        '    return fail(error,TENSOR_ERROR_PROVIDER,"CPU calls need the synchronous CPU stream");']
    values = []
    # Decode scalar dimensions before checking buffer extents, irrespective of parameter order.
    for index, item in enumerate(abi):
        name, dtype = item["name"], item["dtype"]
        if dtype not in C_TYPES:
            raise BuildError(f"CPU profile does not support {dtype}")
        kind = 1 if item["kind"] == "buffer" else 2
        lines += [f'  const auto& arg{index}=call->arguments[{index}];',
                  f'  if(arg{index}.kind!={kind} || arg{index}.dtype!={DTYPES[dtype]})',
                  f'    return fail(error,TENSOR_ERROR_ARGUMENT,"{name}: wrong argument kind or dtype");']
        if kind == 2:
            lines += [f'  {C_TYPES[dtype]} value{index};',
                      f'  std::memcpy(&value{index},&arg{index}.scalar,sizeof(value{index}));']
            width = ctypes.sizeof(SCALAR_TYPES[dtype])
            if width < 8:
                lines += [f'  if((arg{index}.scalar >> {width * 8}) != 0)',
                          f'    return fail(error,TENSOR_ERROR_ARGUMENT,"{name}: noncanonical scalar payload");']
            if dtype in ("float32", "float64"):
                lines += [f'  if(!std::isfinite(value{index}))',
                          f'    return fail(error,TENSOR_ERROR_ARGUMENT,"{name}: nonfinite scalar");']
            if dtype == "bool":
                lines += [f'  if(arg{index}.scalar>1) return fail(error,TENSOR_ERROR_ARGUMENT,"invalid bool");']
            if name in symbols:
                lines += [f'  if(value{index}<=0) return fail(error,TENSOR_ERROR_ARGUMENT,"invalid dimension");']
        values.append(f'reinterpret_cast<{C_TYPES[dtype]}*>(arg{index}.buffer.address)' if kind == 1 else f'value{index}')
    for index, item in enumerate(abi):
        if item["kind"] != "buffer":
            continue
        descriptor = frontend[item["name"]]
        shape = descriptor["shape"]
        extents = []
        for extent in shape:
            if type(extent) is int:
                extents.append(str(extent))
            elif isinstance(extent, dict) and set(extent) == {"var"} and extent["var"] in indices:
                extents.append(f'value{indices[extent["var"]]}')
            else:
                raise BuildError("CPU profile accepts constant or direct symbolic buffer extents")
        lines += [f'  const auto& buffer{index}=arg{index}.buffer;',
            f'  if(buffer{index}.dtype!={DTYPES[item["dtype"]]} || buffer{index}.device_type!=TENSOR_DEVICE_CPU ||',
            f'     buffer{index}.device_ordinal || !buffer{index}.address || buffer{index}.address%{descriptor["alignment"]} ||',
            f'     buffer{index}.rank!={len(shape)} || !buffer{index}.shape || !buffer{index}.strides)',
            f'    return fail(error,TENSOR_ERROR_ARGUMENT,"{item["name"]}: invalid buffer descriptor");',
            f'  uint64_t bytes{index}=sizeof({C_TYPES[item["dtype"]]});']
        for axis in reversed(range(len(shape))):
            lines += [f'  if(buffer{index}.shape[{axis}]!={extents[axis]} || buffer{index}.shape[{axis}]<=0 ||',
                f'     buffer{index}.strides[{axis}]!=static_cast<int64_t>(bytes{index}) ||',
                f'     static_cast<uint64_t>(buffer{index}.shape[{axis}])>UINT64_MAX/bytes{index})',
                f'    return fail(error,TENSOR_ERROR_ARGUMENT,"{item["name"]}: shape, stride or size mismatch");',
                f'  bytes{index}*=static_cast<uint64_t>(buffer{index}.shape[{axis}]);']
        lines += [f'  if(buffer{index}.byte_size<bytes{index})',
                  f'    return fail(error,TENSOR_ERROR_ARGUMENT,"{item["name"]}: insufficient buffer storage");']
    lines += [f'  if({entrypoint}({", ".join(values)})!=0)',
              '    return fail(error,TENSOR_ERROR_PROVIDER,"CPU kernel failed");', '  return 0;', '}']
    return "\n".join(lines)


def build_cpu(source_path, output_path, *, target=None, cache_dir=None, compiler=None, nvcc=None, nvrtc_home=None):
    started = time.perf_counter()
    source_path, output_path = Path(source_path), Path(output_path)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise BuildError("CPU profile currently supports Linux x86-64")
    if target not in (None, "cpu-linux-x86_64") or compiler not in (None, "native") or nvcc or nvrtc_home:
        raise BuildError("CPU builds use --compiler native and target cpu-linux-x86_64")
    if output_path.exists() or not source_path.is_file() or source_path.suffix not in (".py", ".tbin"):
        raise BuildError("source needs an existing .py or portable .tbin file and output must not exist")
    from tensor.artifacts.portable import export_spec, preflight
    checked = preflight(source_path) if source_path.suffix == ".tbin" else None
    packages = check_packages()
    if packages["status"] != "ok":
        raise BuildError(packages["detail"])
    cxx = shutil.which("c++")
    if not cxx:
        raise BuildError("CPU producer needs a native C++ compiler (c++)")
    import tilelang
    import tvm
    from importlib.metadata import version
    from tilelang.backend import create_backend_context
    from tilelang.engine.lower import get_device_call, device_codegen_without_compile
    from tensor.compiler.lowering import frontend_arguments
    try:
        spec = export_spec(source_path, checked)
        if not isinstance(spec, dict) or "kernel" not in spec or set(spec) - {"kernel", "outputs"}:
            raise BuildError("CPU export needs kernel and optional outputs")
        kernel = spec["kernel"]
        symbols = {}
        arguments = frontend_arguments(kernel, symbols)
        context = create_backend_context("c", target_host="c", execution_backend="cython")
        with context.target, tilelang.transform.PassContext(config={"tirx.disable_vectorize": True}):
            mod = context.lower(tvm.IRModule({str(kernel.attrs["global_symbol"]): kernel}))
            device = tvm.tirx.transform.Filter(get_device_call(is_device_c=True))(mod)
            if len(device.functions) != 1:
                raise BuildError("CPU profile requires one kernel")
            name, function = list(device.functions.items())[0]
            entrypoint = str(name.name_hint)
            code = str(device_codegen_without_compile(device, context).inspect_source())
        ir = tvm.ir.save_json(tvm.IRModule({str(kernel.attrs["global_symbol"]): kernel}))
        buffers = {str(buffer.data.name): buffer for buffer in kernel.buffer_map.values()}
        scalars = {str(parameter.name): str(parameter.dtype) for parameter in kernel.params if parameter not in kernel.buffer_map}
        abi = []
        for parameter in function.params:
            key, dtype = str(parameter.name), str(parameter.dtype)
            if dtype == "handle" and key in buffers:
                abi.append({"kind": "buffer", "name": str(buffers[key].name), "dtype": str(buffers[key].dtype)})
            elif dtype in C_TYPES and scalars.get(key, symbols.get(key)) == dtype:
                abi.append({"kind": "scalar", "name": key, "dtype": dtype})
            else:
                raise BuildError(f"unsupported CPU parameter {key}: {dtype}")
        code += "\n" + _wrapper(entrypoint, arguments, abi, symbols)
    except BuildError:
        raise
    except Exception as exc:
        raise BuildError(f"CPU source lowering failed: {type(exc).__name__}: {exc}") from exc
    root = Path(tilelang.__file__).parent
    includes = (root / "src", Path(__file__).resolve().parents[1] / "include")
    identity = {"name": "native", "version": subprocess.check_output([cxx, "--version"], text=True, timeout=30).strip(),
                "platform": platform.platform(), "options": ["-std=c++17", "-O2", "-fPIC", "-shared"]}
    manifest = {"format": FORMAT, "format_version": FORMAT_VERSION, "provider": "cpu", "kind": "native",
        "target": "cpu-linux-x86_64", "entrypoint": "tensor_kernel_v1", "compiler": identity,
        "runtime_abi": runtime_requirement(arguments, symbols), "arguments": arguments, "abi": abi,
        "workspace": workspace_requirement(),
        "symbols": symbols, "outputs": spec.get("outputs", []),
        "launch": {"grid": [1,1,1], "block": [1,1,1], "shared_memory_bytes": 0},
        "source_sha256": checked[0]["source_sha256"] if checked else hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "tilelang_version": version("tilelang"), "tvm_ffi_version": version("apache-tvm-ffi"), "op_set": _op_set(ir),
        "files": {"kernel.so": "0"*64, "kernel.tirx.json": "0"*64}}
    if checked:
        manifest["frontend_artifact_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
    validate_manifest(manifest)
    key = hashlib.sha256(json.dumps({"manifest": manifest, "code": code, "headers": _header_hash(includes)}, sort_keys=True).encode()).hexdigest()
    cache = _cache_root(cache_dir)
    binary = _cached_image(cache, key, "so")
    hit = binary is not None
    compile_seconds = 0.0
    if binary is None:
        began = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="tensor-cpu-build-") as directory:
            path = Path(directory)
            (path / "kernel.cpp").write_text(code, encoding="utf-8")
            command = [cxx, *identity["options"], *[f"-I{p}" for p in includes], str(path / "kernel.cpp"), "-o", str(path / "kernel.so")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise BuildError(f"CPU compilation failed:\n{result.stderr}")
            binary = (path / "kernel.so").read_bytes()
        compile_seconds = time.perf_counter() - began
        try:
            _write_cache(cache, key, binary, "so")
        except OSError:
            pass
    files = {"kernel.so": binary, "kernel.tirx.json": ir.encode(), **_notices(root)}
    manifest["files"] = {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with output_path.open("xb") as stream:
            created = True
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as bundle:
                for name, content in files.items():
                    bundle.writestr(name, content)
                bundle.writestr("manifest.json", json.dumps(manifest, sort_keys=True, indent=2))
    except BaseException:
        if created:
            output_path.unlink(missing_ok=True)
        raise
    return {"status": "built", "provider": "cpu", "compiler": "native", "target": manifest["target"],
            "path": str(output_path.resolve()), "bytes": output_path.stat().st_size, "arguments": arguments,
            "seconds": time.perf_counter()-started, "compile_seconds": compile_seconds, "nvcc_seconds": 0,
            "cache_hit": hit, "cache_key": key}
