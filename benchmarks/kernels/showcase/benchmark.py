"""Reproducible CUDA kernel measurements, schedule search and matched controls.

Run from the repository with a producer Python containing Torch, Triton,
TileLang 0.1.14 and apache-tvm-ffi 0.1.12. See README.md for commands.
"""
# Benchmark trials retain compiler/runtime failures and continue to other candidates.
# ruff: noqa: BLE001

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import tilelang
import torch
import torch.nn.functional as F
import triton

import tensor as tx
from benchmarks.kernels.showcase import kernels
from benchmarks.kernels.showcase import triton_kernels as tk
from tensor.artifacts.format import read_artifact
from tensor.compiler.cuda.matmul import tiled_matmul
from tensor.compiler.cuda.schedules import matmul_legal, matmul_space
from tensor.compiler.search import ScheduleSearch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_info(path):
    def run(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True
        ).strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def graph_time(call, iterations, repetitions):
    """Use repeated nodes to amortize graph replay; retain every event sample."""
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
        for _ in range(iterations):
            call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start, stop = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) * 1000 / iterations)
    return {
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
    }


def cases(quick):
    result = []
    mm = [(128, 1024, 1024), (512, 2048, 2048), (1024, 1024, 1024), (129, 1000, 1024)]
    fa = [
        (1, 8, 512, 64, False),
        (1, 8, 1024, 64, True),
        (1, 8, 1024, 128, False),
        (1, 8, 129, 64, True),
    ]
    mla = [(1, 16, 1024), (4, 16, 1024), (1, 32, 4096)]
    for shape in mm[:1] if quick else mm:
        result.append({"workload": "matmul", "shape": list(shape)})
    for b, h, s, d, causal in fa[:1] if quick else fa:
        result.append(
            {"workload": "flash_attention", "shape": [b, h, s, d], "causal": causal}
        )
    for shape in mla[:1] if quick else mla:
        result.append(
            {
                "workload": "mla_decode",
                "shape": list(shape),
                "qk_dim": 576,
                "v_dim": 512,
            }
        )
    for encoding in (2,) if quick else (2, 12, 14):
        for m in (1, 32):
            result.append(
                {"workload": "gguf", "shape": [m, 1024, 1024], "encoding": encoding}
            )
    for index, case in enumerate(result):
        suffix = (
            f"-q{case['encoding']}"
            if "encoding" in case
            else "-causal"
            if case.get("causal")
            else ""
        )
        case["id"] = case["workload"] + "-" + "x".join(map(str, case["shape"])) + suffix
    return result


def configs(case, count, target):
    kind = case["workload"]
    if kind == "matmul":
        m, _n, k = case["shape"]
        fixed = {
            "block_m": 32,
            "block_n": 64,
            "block_k": 32,
            "stages": 2,
            "threads": 128,
            "orientation": "activation",
        }
        seed = {
            "family": "matmul",
            "block_m": 128 if m >= 128 else 32,
            "block_n": 32,
            "block_k": 64,
            "stages": 3,
            "threads": 128,
            "orientation": "weight",
        }
        search = ScheduleSearch(
            [seed],
            spaces=matmul_space(m, target=target),
            legal=lambda c: matmul_legal(c, target=target, depth=k),
        )
        return fixed, search
    if kind == "flash_attention":
        fixed = {"block_m": 32, "block_n": 64, "threads": 128}
        space = {
            "attention": {
                "block_m": (16, 32, 64),
                "block_n": (32, 64, 128),
                "threads": (128, 256),
            }
        }
        seeds = [
            {"family": "attention", **fixed},
            {"family": "attention", "block_m": 64, "block_n": 64, "threads": 128},
        ]
    elif kind == "mla_decode":
        fixed = {"splits": 8, "block_n": 32, "threads": 128, "value_dim": 576}
        space = {
            "mla": {
                "splits": (4, 8, 16, 32, 64),
                "block_n": (16, 32, 64),
                "threads": (64, 128, 256),
                "value_dim": (512, 576),
            }
        }
        seeds = [
            {"family": "mla", **fixed},
            {
                "family": "mla",
                "splits": 32,
                "block_n": 32,
                "threads": 128,
                "value_dim": 512,
            },
            {
                "family": "mla",
                "splits": 64,
                "block_n": 16,
                "threads": 64,
                "value_dim": 512,
            },
        ]
        return fixed, ScheduleSearch(
            seeds,
            spaces=space,
            legal=lambda c: (
                c["block_n"] >= c["threads"] // 4
                and c["splits"] * c["block_n"] <= case["shape"][2]
            ),
        )
    elif case["shape"][0] == 1:
        fixed = {"threads": 128, "unroll": 4}
        space = {"gemv": {"threads": (64, 128, 256), "unroll": (1, 2, 4, 8)}}
        seeds = [{"family": "gemv", **fixed}]
    else:
        fixed = {
            "block_m": 32,
            "block_n": 64,
            "block_k": 32,
            "stages": 2,
            "threads": 128,
            "packed_pairs": False,
        }
        space = {
            "dequant": {
                "block_m": (16, 32, 64),
                "block_n": (32, 64, 128),
                "block_k": (32, 64, 128),
                "stages": (2, 3),
                "threads": (128, 256),
                "packed_pairs": (False, True),
            }
        }
        seeds = [{"family": "dequant", **fixed}]
    return fixed, ScheduleSearch(seeds, spaces=space)


def source(case, config):
    kind, shape = case["workload"], case["shape"]
    if kind == "flash_attention":
        text = (ROOT / "examples/flash_attention.py").read_text()
        values = dict(zip(("BATCH", "HEADS", "SEQ_LEN", "HEAD_DIM"), shape))
        values.update(
            IS_CAUSAL=case["causal"],
            BLOCK_M=config["block_m"],
            BLOCK_N=config["block_n"],
        )
        for name, value in values.items():
            text, n = re.subn(
                rf"^{name} = .*$", f"{name} = {value!r}", text, flags=re.MULTILINE
            )
            assert n == 1
        return text.replace("threads=128", f"threads={config['threads']}")
    if kind == "matmul":
        expr = f"matmul(*{shape!r}, dtype='float16', **{config!r})"
        imports = "from tensor.compiler.cuda.matmul import matmul"
    elif kind == "gguf":
        expr = f"gguf(*{shape!r}, encoding={case['encoding']}, **{config!r})"
        imports = "from benchmarks.kernels.showcase.kernels import gguf"
    else:
        expr = f"mla_partial(*{shape!r}, **{config!r})"
        imports = "from benchmarks.kernels.showcase.kernels import mla_partial"
    # build() imports this specialization; module dependencies are hashed in results.
    return imports + f"\ndef tensor_export():\n    return {{'kernel': {expr}}}\n"


def load_prim(path):
    from tensor.compiler.frontend.portable import export_spec

    return export_spec(path, None)["kernel"]


class TensorCall:
    def __init__(self, device, paths, arglists):
        self.kernels = [device.load(path) for path in paths]
        self.buffers = [[device.from_dlpack(t) for t in args] for args in arglists]

    def __call__(self):
        for kernel, buffers in zip(self.kernels, self.buffers):
            kernel.launch(*buffers)

    def close(self):
        for buffers in self.buffers:
            for buffer in buffers:
                buffer.release()
        for kernel in self.kernels:
            kernel.release()


def arguments(case, data, config):
    if case["workload"] != "mla_decode":
        return [[*data["inputs"], data["out"]]]
    b, h, _ = case["shape"]
    splits = config["splits"]
    parts = torch.empty((b, h, splits, 512), device="cuda", dtype=torch.float32)
    lse = torch.empty((b, h, splits), device="cuda", dtype=torch.float32)
    return [[*data["inputs"], parts, lse], [parts, lse, data["out"]]]


def build_tensor(case, config, directory, device, data, target):
    directory.mkdir(parents=True, exist_ok=True)
    src = directory / "kernel.py"
    src.write_text(source(case, config))
    paths = [directory / "kernel.tbin"]
    sources = [src]
    if case["workload"] == "mla_decode":
        merge = directory / "merge.py"
        merge.write_text(
            "from benchmarks.kernels.showcase.kernels import mla_merge\n"
            f"def tensor_export():\n    return {{'kernel': mla_merge({case['shape'][0]}, {case['shape'][1]}, {config['splits']})}}\n"
        )
        sources.append(merge)
        paths.append(directory / "merge.tbin")
    builds = [
        tx.build(
            src, path, target=target, cache_dir=directory.parents[1] / "compiler-cache"
        )
        for src, path in zip(sources, paths)
    ]
    manifests = [read_artifact(path)[0] for path in paths]
    call = TensorCall(device, paths, arguments(case, data, config))
    evidence = {
        "artifacts": [str(p) for p in paths],
        "artifact_sha256": [sha(p) for p in paths],
        "sources": [str(p) for p in sources],
        "source_sha256": [sha(p) for p in sources],
        "builds": builds,
        "prelowering": [m.get("cuda_prelowering") for m in manifests],
        "shared_memory_bytes": [m["launch"]["shared_memory_bytes"] for m in manifests],
    }
    return call, evidence


def tilelang_call(case, config, data, directory, target):
    directory.mkdir(parents=True, exist_ok=True)
    src = directory / "kernel.py"
    src.write_text(source(case, config))
    if case["workload"] == "matmul":
        prims = [tiled_matmul(*case["shape"], "float16", "float16", **config)]
    else:
        prims = [load_prim(src)]
    if case["workload"] == "mla_decode":
        prims.append(
            kernels.mla_merge(case["shape"][0], case["shape"][1], config["splits"])
        )
    # AKAS calls a scalar unaligned U16 load helper. Supply its definition to
    # TileLang's ordinary NVCC callback; all scheduling/codegen stays native.
    import tvm_ffi

    from tensor.compiler.cuda.lowering import lower_cuda_intrinsics

    callback_name = "tilelang_callback_cuda_compile"
    original = tvm_ffi.get_global_func(callback_name)
    helper_used = []

    def native_compile(code, cuda_target, pass_config=None):
        expanded = lower_cuda_intrinsics(code)
        helper_used.append(expanded != code)
        return original(expanded, cuda_target, pass_config)

    if case["workload"] == "gguf":
        tvm_ffi.register_global_func(callback_name, native_compile, override=True)
    try:
        compiled = [
            tilelang.compile(
                p, target={"kind": "cuda", "arch": target}, execution_backend="tvm_ffi"
            )
            for p in prims
        ]
    finally:
        if case["workload"] == "gguf":
            tvm_ffi.register_global_func(callback_name, original, override=True)
    args = arguments(case, data, config)

    def call():
        for kernel, values in zip(compiled, args):
            kernel(*values)

    codes = [k.get_kernel_source() for k in compiled]
    for i, code in enumerate(codes):
        (directory / f"kernel-{i}.cu").write_text(code)
    return call, {
        "config": config,
        "cuda_sha256": [hashlib.sha256(c.encode()).hexdigest() for c in codes],
        "compiler": "TileLang native NVCC / TVM-FFI",
        "same_schedule_as_tensor": True,
        "akas_scalar_intrinsic_definitions_supplied": any(helper_used),
    }


def prepare(case):
    kind, shape = case["workload"], case["shape"]
    full_precision_reference = None
    if kind == "matmul":
        m, n, k = shape
        inputs = [
            torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.2,
            torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.2,
        ]
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        reference = lambda: (inputs[0].float() @ inputs[1].float().T).half()
        baseline = lambda: torch.mm(inputs[0], inputs[1].T, out=out)
        label = "torch_cublas"
    elif kind == "flash_attention":
        inputs = [
            torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)
        ]
        out = torch.empty_like(inputs[0])

        def reference():
            scores = (
                inputs[0].float()
                @ inputs[1].float().transpose(-1, -2)
                * shape[-1] ** -0.5
            )
            if case["causal"]:
                mask = torch.ones(
                    (shape[-2], shape[-2]), device="cuda", dtype=torch.bool
                ).tril()
                scores.masked_fill_(~mask, -float("inf"))
            return (scores.softmax(-1) @ inputs[2].float()).half()

        def baseline():
            with torch.nn.attention.sdpa_kernel(
                torch.nn.attention.SDPBackend.FLASH_ATTENTION
            ):
                return F.scaled_dot_product_attention(*inputs, is_causal=case["causal"])

        label = "torch_sdpa_flash"
    elif kind == "mla_decode":
        b, h, s = shape
        inputs = [
            torch.randn((b, h, 576), device="cuda", dtype=torch.float16),
            torch.randn((b, s, 576), device="cuda", dtype=torch.float16),
        ]
        out = torch.empty((b, h, 512), device="cuda", dtype=torch.float16)

        def reference():
            scores = inputs[0].float() @ inputs[1].float().transpose(-1, -2) * 576**-0.5
            return (scores.softmax(-1) @ inputs[1][..., :512].float()).half()

        # This is a materializing FP16 PyTorch MQA baseline, not FlashMLA.
        baseline = lambda: (
            (inputs[0] @ inputs[1].transpose(-1, -2) * 576**-0.5).softmax(-1)
            @ inputs[1][..., :512]
        )
        label = "torch_mqa_materialized"
    else:
        import numpy as np
        from akas.quantization.weights.ggml.definition import weight_storage
        from akas.quantization.weights.ggml.validation import sample_weight

        m, n, k = shape
        size, dtype = weight_storage(case["encoding"], n * k)
        raw = np.empty(size, dtype=dtype)
        decoded = sample_weight(raw, case["encoding"]).reshape(n, k)
        inputs = [
            torch.randn(m * k, device="cuda", dtype=torch.float32) * 0.2,
            torch.from_numpy(raw).cuda(),
        ]
        out = torch.empty(m * n, device="cuda", dtype=torch.float32)
        weight = torch.from_numpy(decoded).cuda()
        # All fused prefill kernels decode to FP16 and use FP16 MMA operands.
        # GEMV uses the original FP32 activation and decoded GGML weight.
        full_precision_reference = lambda: (inputs[0].reshape(m, k) @ weight.T).reshape(
            -1
        )
        reference = (
            full_precision_reference
            if m == 1
            else lambda: (
                inputs[0].reshape(m, k).half().float() @ weight.half().float().T
            ).reshape(-1)
        )
        # Resident dequantized GEMM is a lower bound, explicitly not a fused baseline.
        activation = torch.empty((m, k), device="cuda", dtype=torch.float16)
        resident = weight.half()
        temp = torch.empty((m, n), device="cuda", dtype=torch.float16)

        def baseline():
            activation.copy_(inputs[0].reshape(m, k))
            torch.mm(activation, resident.T, out=temp)
            out.copy_(temp.reshape(-1))

        label = "torch_resident_dequantized_lower_bound"
    return {
        "inputs": inputs,
        "out": out,
        "reference": reference,
        "baseline": baseline,
        "baseline_label": label,
        "full_precision_reference": full_precision_reference,
    }


def validate(call, data, case, thorough=False):
    out = data["out"]
    saved = data["inputs"][0].clone() if thorough else None
    errors = []
    families = ("random", "fresh_scaled", "zero") if thorough else ("random",)
    # Replay the very same captured graph after changing inputs, proving that
    # partials/scratch are overwritten on every invocation.
    graph = None
    if thorough:
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.current_stream()):
            captured_return = call()
    try:
        for family in families:
            if family == "fresh_scaled":
                data["inputs"][0].copy_(saved * -1.5)
            elif family == "zero":
                data["inputs"][0].zero_()
            expected = data["reference"]()
            out.fill_(float("nan"))
            if graph is not None:
                graph.replay()
                returned = captured_return
            else:
                returned = call()
            if isinstance(returned, torch.Tensor):
                actual = returned
            else:
                actual = out
            torch.cuda.synchronize()
            rtol, atol = (2e-2, 3e-3) if case["workload"] != "gguf" else (2e-2, 2e-2)
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            errors.append(
                {
                    "input": family,
                    "max_abs_error": (actual.float() - expected.float())
                    .abs()
                    .max()
                    .item(),
                    "rtol": rtol,
                    "atol": atol,
                }
            )
            if data.get("full_precision_reference"):
                full = data["full_precision_reference"]()
                errors[-1]["max_abs_error_vs_fp32_operands"] = (
                    (actual.float() - full).abs().max().item()
                )
                errors[-1]["relative_l2_error_vs_fp32_operands"] = (
                    (actual.float() - full).norm() / full.norm().clamp_min(1e-12)
                ).item()
    finally:
        if saved is not None:
            data["inputs"][0].copy_(saved)
    return errors


def triton_configs(case):
    kind = case["workload"]
    if kind == "gguf" and case["shape"][0] == 1:
        return [
            {"rows": r, "warps": w, "stages": 2}
            for r, w in [
                (1, 4),
                (2, 4),
                (4, 4),
                (8, 4),
                (16, 4),
                (4, 8),
                (8, 8),
                (16, 8),
            ]
        ]
    if kind == "matmul" or kind == "gguf":
        return [
            {"BM": bm, "BN": bn, "BK": bk, "warps": w, "stages": st}
            for bm, bn, bk, w, st in [
                (32, 64, 32, 4, 3),
                (32, 64, 64, 4, 3),
                (64, 64, 32, 4, 3),
                (64, 128, 32, 4, 3),
                (32, 128, 64, 4, 3),
                (128, 64, 32, 4, 3),
                (16, 64, 64, 4, 3),
                (64, 64, 64, 8, 3),
            ]
        ]
    if kind == "flash_attention":
        return [
            {"BM": bm, "BN": bn, "warps": w, "stages": 2}
            for bm, bn, w in [
                (32, 32, 4),
                (32, 64, 4),
                (64, 32, 4),
                (64, 64, 4),
                (64, 64, 8),
                (128, 32, 4),
                (128, 64, 8),
                (32, 128, 4),
            ]
        ]
    return [
        {"splits": s, "BN": bn, "warps": w, "stages": 2}
        for s, bn, w in [
            (4, 32, 4),
            (8, 32, 4),
            (16, 32, 4),
            (32, 32, 4),
            (8, 64, 4),
            (16, 64, 4),
            (8, 32, 8),
            (16, 32, 8),
            (32, 16, 4),
            (64, 16, 4),
        ]
    ]


def triton_call(case, config, data):
    kind, shape = case["workload"], case["shape"]
    c = dict(config)
    warps, stages = c.pop("warps"), c.pop("stages")
    if kind == "gguf" and shape[0] == 1:
        _, n, k = shape

        def call():
            tk.gguf_gemv[(triton.cdiv(n, c["rows"]),)](
                *data["inputs"],
                data["out"],
                n,
                k,
                case["encoding"],
                c["rows"],
                num_warps=warps,
                num_stages=stages,
            )
    elif kind in ("matmul", "gguf"):
        m, n, k = shape
        func = tk.matmul if kind == "matmul" else tk.gguf
        extra = {} if kind == "matmul" else {"Q": case["encoding"]}

        def call():
            func[(triton.cdiv(m, c["BM"]), triton.cdiv(n, c["BN"]))](
                *data["inputs"],
                data["out"],
                m,
                n,
                k,
                **c,
                **extra,
                num_warps=warps,
                num_stages=stages,
            )
    elif kind == "flash_attention":
        b, h, s, d = shape

        def call():
            tk.attention[(triton.cdiv(s, c["BM"]), b * h)](
                *data["inputs"],
                data["out"],
                h,
                s,
                d,
                case["causal"],
                **c,
                num_warps=warps,
                num_stages=stages,
            )
    else:
        b, h, s = shape
        splits = c["splits"]
        p = torch.empty((b, h, splits, 512), device="cuda", dtype=torch.float32)
        l = torch.empty((b, h, splits), device="cuda", dtype=torch.float32)

        def call():
            tk.mla_partial[(triton.cdiv(h, 16), splits, b)](
                *data["inputs"],
                p,
                l,
                h,
                s,
                splits,
                c["BN"],
                num_warps=warps,
                num_stages=stages,
            )
            tk.mla_merge[(b * h,)](p, l, data["out"], splits, num_warps=4)

    return call


def report(directory, result):
    rows = []
    for case in result["cases"]:
        for provider, measurement in case.get("measurements", {}).items():
            rows.append(
                {
                    "case": case["id"],
                    "provider": provider,
                    **{k: measurement[k] for k in ("median_us", "min_us", "max_us")},
                }
            )
    if rows:
        with (directory / "measurements.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = [
        "# CUDA showcase measurements",
        "",
        f"GPU: {result['environment']['gpu']['name']}. All times are microseconds per complete operation.",
        "CUDA events over repeated operations in CUDA graphs; median of independent replay samples. Compilation, tuning, allocation and validation are excluded.",
        "",
        "| Case | Tensor searched | TileLang fixed | TileLang same schedule | Triton searched | PyTorch | FA2 / GGML | Speedup vs fixed TileLang |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in result["cases"]:
        ms = case.get("measurements", {})

        def val(key, values=ms):
            return f"{values[key]['median_us']:.3f}" if key in values else "—"

        pt = case.get("pytorch_label", "")
        speedup = (
            ms["tilelang_fixed"]["median_us"] / ms["tensor_searched"]["median_us"]
            if {"tilelang_fixed", "tensor_searched"} <= ms.keys()
            else None
        )
        upstream = (
            "upstream_fa2" if case["workload"] == "flash_attention" else "ggml_cuda"
        )
        text.append(
            f"| {case['id']} | {val('tensor_searched')} | {val('tilelang_fixed')} | {val('tilelang_same_schedule')} | {val('triton_searched')} | {val(pt)} | {val(upstream)} | {speedup:.2f}× |"
            if speedup
            else f"| {case['id']} | incomplete | | | | | | |"
        )
    text += [
        "",
        "See results.json for every candidate, compilation failure, correctness error, selected schedule, raw timing sample, version and artifact hash.",
        "",
        "MLA is dense absorbed V3-style MQA with a contiguous shared latent cache, QK=576, V=512, one query token and a complete split/merge. It excludes model projections and paging indirection. Upstream FlashMLA does not support this SM89 GPU.",
        "GGUF uses real raw GGML Q4_0/Q4_K/Q6_K byte layouts generated by AKAS; prefill dequantizes inside the kernel. PyTorch's resident-dequantized comparison excludes weight decoding and is only a lower bound, not a fused GGUF competitor.",
        "GGUF prefill uses FP16 operands and FP32 accumulation: its independent oracle rounds the decoded weight and activation to FP16, then multiplies in FP32. Errors against the original FP32 operands are also reported. GEMV uses the original FP32 operands. GGML may use Q8_1 activations and has a separate relative-L2 qualification.",
        "Tensor matmul uses checked semantic prelowering. Attention/MLA/GGUF are explicit kernels with producer schedule search; there is no semantic attention/GGUF prelowering pass. Matched TileLang controls use the same selected schedule with native NVCC and TVM-FFI.",
        "Search timings select candidates; the table uses separate final measurements in randomized provider order. A fixed TileLang comparison measures the whole search advantage; a same-schedule control measures compiler/runtime differences. Both losses and wins are retained.",
    ]
    (directory / "REPORT.md").write_text("\n".join(text) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--akas", type=Path, default=ROOT.parent / "akas")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--workload", choices=["matmul", "flash_attention", "mla_decode", "gguf"]
    )
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=9)
    parser.add_argument("--ggml-library", type=Path)
    parser.add_argument("--fa-python", type=Path)
    parser.add_argument("--case", help="run one exact case ID")
    args = parser.parse_args()
    if args.candidates < 1 or args.iterations < 1 or args.repetitions < 3:
        parser.error(
            "positive candidates/iterations and at least three repetitions required"
        )
    sys.path.insert(0, str(args.akas.resolve() / "src"))
    args.out.mkdir(parents=True, exist_ok=False)
    snapshot = args.out / "driver-sources"
    snapshot.mkdir()
    for path in [
        *Path(__file__).parent.glob("*.py"),
        Path(__file__).with_name("ggml_bridge.cu"),
    ]:
        (snapshot / path.name).write_bytes(path.read_bytes())
    torch.manual_seed(20261024)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    props = torch.cuda.get_device_properties(0)
    target = f"sm_{props.major}{props.minor}"
    env = {
        "gpu": {
            "name": props.name,
            "sm": target,
            "multiprocessors": props.multi_processor_count,
            "memory_bytes": props.total_memory,
        },
        "versions": {
            name: metadata.version(name)
            for name in ("torch", "triton", "tilelang", "apache-tvm-ffi", "numpy")
        },
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "tensor": git_info(ROOT),
        "akas": git_info(args.akas),
        "nvidia_smi": subprocess.check_output(["nvidia-smi", "-q"], text=True),
        "nvcc": subprocess.check_output(["nvcc", "--version"], text=True),
        "build_environment": {
            k: os.environ.get(k)
            for k in ("CUDA_HOME", "TENSOR_NVRTC_HOME", "CPATH", "LD_LIBRARY_PATH")
        },
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in Path(__file__).parent.glob("*.py")
        },
        "akas_kernel_sha256": {
            str(p.relative_to(args.akas)): sha(p)
            for p in [
                args.akas / "src/akas/ops/linear/cuda.py",
                args.akas / "src/akas/quantization/weights/ggml/cuda.py",
                args.akas / "src/akas/quantization/weights/ggml/kernel.py",
            ]
        },
    }
    if args.ggml_library:
        llama = args.akas / "build/baselines/llama.cpp"
        env["ggml"] = {
            **git_info(llama),
            "bridge_sha256": sha(args.ggml_library),
            "bridge_source_sha256": sha(Path(__file__).with_name("ggml_bridge.cu")),
            "cuda_library_sha256": sha(llama / "build-cuda/bin/libggml-cuda.so.0.27.0"),
        }
    result = {
        "schema": "tensor.cuda-showcase.v1",
        "environment": env,
        "command": sys.argv,
        "iterations": args.iterations,
        "repetitions": args.repetitions,
        "seed": 20261024,
        "timing": "warm CUDA events over repeated nodes in CUDA graph; hot cache; allocations excluded",
        "upstream_flashmla": {
            "status": "unsupported" if props.major == 8 else "not_measured",
            "reason": "V3 dense upstream requires SM90; current main changed model/architecture scope",
            "source": "https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7",
        },
        "cases": [],
    }
    stream = torch.cuda.Stream()
    with (
        torch.inference_mode(),
        torch.cuda.stream(stream),
        tx.Device(stream=stream.cuda_stream) as device,
    ):
        for case in cases(args.quick):
            if args.workload and case["workload"] != args.workload:
                continue
            if args.case and case["id"] != args.case:
                continue
            print(json.dumps({"case": case["id"], "status": "starting"}), flush=True)
            row = {
                **case,
                "candidates": [],
                "baselines": {},
                "measurements": {},
                "errors": [],
            }
            result["cases"].append(row)
            directory = args.out / case["id"]
            directory.mkdir()
            data = prepare(case)
            fixed, search = configs(case, args.candidates, target)
            selected = None
            for index in range(args.candidates):
                try:
                    c = search.next()
                except StopIteration:
                    break
                c = {k: v for k, v in c.items() if k != "family"}
                trial = {"config": c}
                row["candidates"].append(trial)
                call = None
                start = time.perf_counter()
                try:
                    call, evidence = build_tensor(
                        case, c, directory / f"trial-{index:03d}", device, data, target
                    )
                    trial.update(evidence)
                    trial["correctness"] = validate(call, data, case)
                    trial["timing"] = graph_time(call, 30, 5)
                    trial["status"] = "passed"
                    search.record(
                        {"family": next(iter(search.spaces)), **c},
                        trial["timing"]["median_us"] * 1e-6,
                    )
                    if (
                        selected is None
                        or trial["timing"]["median_us"]
                        < selected["timing"]["median_us"]
                    ):
                        selected = trial
                except Exception as exc:
                    trial.update(
                        status="failed", error=str(exc), exception=type(exc).__name__
                    )
                finally:
                    if call:
                        call.close()
                    trial["wall_seconds"] = time.perf_counter() - start
                print(
                    json.dumps(
                        {
                            "case": case["id"],
                            "trial": index,
                            "status": trial["status"],
                            "us": trial.get("timing", {}).get("median_us"),
                            "error": trial.get("error", "")[:200],
                        }
                    ),
                    flush=True,
                )
                write_json(args.out / "results.json", result)
            if selected is None:
                row["errors"].append("No valid Tensor candidate")
                continue
            # Remeasure the leading schedules independently before final choice.
            finalists = sorted(
                (t for t in row["candidates"] if t["status"] == "passed"),
                key=lambda t: t["timing"]["median_us"],
            )[:3]
            for trial in finalists:
                call = TensorCall(
                    device,
                    [Path(p) for p in trial["artifacts"]],
                    arguments(case, data, trial["config"]),
                )
                try:
                    trial["selection_timing"] = graph_time(call, 60, 7)
                finally:
                    call.close()
            selected = min(finalists, key=lambda t: t["selection_timing"]["median_us"])
            row["selected"] = selected
            calls, cleanup = {}, []
            selected_paths = [Path(p) for p in selected["artifacts"]]
            call = TensorCall(
                device, selected_paths, arguments(case, data, selected["config"])
            )
            cleanup.append(call)
            calls["tensor_searched"] = call
            row["tensor_correctness"] = validate(call, data, case, thorough=True)
            try:
                call, evidence = build_tensor(
                    case,
                    {} if case["workload"] == "matmul" else fixed,
                    directory / "tensor_default",
                    device,
                    data,
                    target,
                )
                cleanup.append(call)
                evidence["correctness"] = validate(call, data, case, thorough=True)
                row["baselines"]["tensor_default"] = evidence
                calls["tensor_default"] = call
            except Exception as exc:
                row["errors"].append({"provider": "tensor_default", "error": str(exc)})
            for label, config in (
                ("tilelang_fixed", fixed),
                ("tilelang_same_schedule", selected["config"]),
            ):
                try:
                    call, evidence = tilelang_call(
                        case, config, data, directory / label, target
                    )
                    evidence["same_schedule_as_tensor"] = config == selected["config"]
                    evidence["correctness"] = validate(call, data, case, thorough=True)
                    row["baselines"][label] = evidence
                    calls[label] = call
                except Exception as exc:
                    row["errors"].append(
                        {
                            "provider": label,
                            "error": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
            triton_trials, best = [], None
            row["triton_candidates"] = triton_trials
            for config in triton_configs(case):
                trial = {"config": config}
                triton_trials.append(trial)
                try:
                    call = triton_call(case, config, data)
                    trial["correctness"] = validate(call, data, case)
                    trial["timing"] = graph_time(call, 30, 5)
                    trial["status"] = "passed"
                    if (
                        best is None
                        or trial["timing"]["median_us"] < best[0]["timing"]["median_us"]
                    ):
                        best = trial, call
                except Exception as exc:
                    trial.update(status="failed", error=str(exc))
            if best:
                row["triton_selected"] = best[0]
                row["triton_correctness"] = validate(best[1], data, case, thorough=True)
                calls["triton_searched"] = best[1]
            pt = data["baseline_label"]
            row["pytorch_label"] = pt
            try:
                row["baselines"][pt] = {
                    "correctness": validate(data["baseline"], data, case, thorough=True)
                }
                calls[pt] = data["baseline"]
            except Exception as exc:
                row["errors"].append({"provider": pt, "error": str(exc)})
            order = list(calls)
            random.Random(42).shuffle(order)
            row["measurement_order"] = order
            for label in order:
                row["measurements"][label] = graph_time(
                    calls[label], args.iterations, args.repetitions
                )
            if (case["workload"] == "gguf" and args.ggml_library) or (
                case["workload"] == "flash_attention" and args.fa_python
            ):
                import numpy as np

                external = "ggml" if case["workload"] == "gguf" else "fa2"
                path = directory / "external-inputs.npz"
                np.savez(
                    path,
                    **{
                        f"input{i}": t.cpu().numpy()
                        for i, t in enumerate(data["inputs"])
                    },
                    expected=data["reference"]().cpu().numpy(),
                    case=json.dumps(case),
                )
                command = [
                    str(args.fa_python) if external == "fa2" else sys.executable,
                    str(Path(__file__).with_name("external_baselines.py")),
                    "--kind",
                    external,
                    "--inputs",
                    str(path),
                    "--out",
                    str(directory / "external.json"),
                    "--iterations",
                    str(args.iterations),
                    "--repetitions",
                    str(args.repetitions),
                ]
                if external == "ggml":
                    command += ["--library", str(args.ggml_library)]
                with (directory / "external.log").open("w") as log:
                    proc = subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=False
                    )
                if proc.returncode:
                    row["errors"].append(
                        {
                            "provider": external,
                            "error": (directory / "external.log").read_text()[-4000:],
                        }
                    )
                else:
                    ext = json.loads((directory / "external.json").read_text())
                    row["baselines"][ext["provider"]] = ext
                    row["measurements"][ext["provider"]] = {
                        key: ext[key]
                        for key in ("median_us", "min_us", "max_us", "samples_us")
                    }
            for call in cleanup:
                call.close()
            row["status"] = "passed" if not row["errors"] and best else "partial"
            write_json(args.out / "results.json", result)
            report(args.out, result)
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "status": row["status"],
                        "timings": {
                            k: v["median_us"] for k, v in row["measurements"].items()
                        },
                        "errors": [
                            {
                                "provider": e.get("provider"),
                                "error": e.get("error", "")[-300:],
                            }
                            for e in row["errors"]
                        ],
                    }
                ),
                flush=True,
            )
    result["status"] = (
        "passed"
        if all(r.get("status") == "passed" for r in result["cases"])
        else "partial"
    )
    result["environment"]["nvidia_smi_after"] = subprocess.check_output(
        ["nvidia-smi", "-q"], text=True
    )
    write_json(args.out / "results.json", result)
    report(args.out, result)


if __name__ == "__main__":
    main()
