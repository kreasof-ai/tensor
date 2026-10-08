"""Reproducible LLT backend qualification, with tolerances fixed before runs."""

import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import importlib.metadata
import platform
import statistics
import subprocess
import time
from dataclasses import asdict, replace
import torch
from tensor_torch.llt import Operators, AdamW, KVCache
from .model import Config, Model

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/research/data/llt-readiness"
TOLERANCES = {
    "attention_fp16": {"atol": 0.005, "rtol": 0.005},
    "attention_bf16": {"atol": 0.035, "rtol": 0.035},
    "gradient_bf16": {"atol": 0.05, "rtol": 0.05},
    "model_loss_absolute": 0.15,
    "model_gradient_relative_l2": 0.08,
    "training_loss_absolute": 0.15,
    "training_parameter_rms": 0.03,
    "resume_parameter_absolute": 1e-6,
}


def setup(seed=3):
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False


def save(name, result):
    OUT.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__), Path(__file__).with_name("model.py")]
    files += list((ROOT / "packages/tensor-torch/src/tensor_torch").glob("*.py"))
    files += list(
        (ROOT / "packages/tensor-torch/src/tensor_torch/templates").glob("llt*.py")
    )
    files += list((ROOT / "src/tensor/runtime").glob("*.py"))
    files += [
        ROOT / "src/tensor/providers/cuda.py",
        ROOT / "src/tensor/compiler/nvrtc.py",
        ROOT / "src/tensor/artifacts/format.py",
        ROOT / "src/tensor/include/tensor/abi.h",
        ROOT / "packages/tensor-torch/src/tensor_torch/executor.cpp",
    ]
    sources = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files
    }
    snapshot = OUT / "sources"
    for p in files:
        dest = snapshot / sources[str(p.relative_to(ROOT))] / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(p.read_bytes())
    result.update(
        environment={
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "driver": subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
            ).strip(),
            "tilelang": importlib.metadata.version("tilelang"),
            "tvm_ffi": importlib.metadata.version("apache-tvm-ffi"),
            "nvrtc": "12.9",
            "cuda_timing_policy": "single qualification process; no concurrent GPU experiments",
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(),
            "sm": torch.cuda.get_device_capability(),
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "sources": sources,
        },
        tolerances=TOLERANCES,
    )
    (OUT / (name + ".json")).write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )


def batch(generator, c, b=1, s=33):
    start = torch.randint(c.vocab, (b, 1), generator=generator)
    tokens = (start + torch.arange(s)[None, :]) % c.vocab
    return tokens.cuda(), ((tokens + 1) % c.vocab).cuda()


def train_step(model, optimizer, tokens, target, reference=False):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(tokens, target)
    loss.backward()
    if reference:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.detach()


def gradients(ops):
    results = []
    for architecture in ("llt", "naive"):
        for rotary in (0, 16):
            setup()
            c = Config(architecture=architecture, rotary=rotary)
            model = Model(c, ops).cuda()
            reference = Model(c).cuda()
            reference.load_state_dict(model.state_dict())
            generator = torch.Generator().manual_seed(9)
            tokens, target = batch(generator, c)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(tokens, target)
                expected = reference(tokens, target)
            loss.backward()
            expected.backward()
            error = abs(loss.item() - expected.item())
            assert error < TOLERANCES["model_loss_absolute"], error
            errors = {}
            for (name, p), (_, q) in zip(
                model.named_parameters(), reference.named_parameters()
            ):
                assert p.grad is not None and q.grad is not None, name
                rms = (p.grad - q.grad).float().norm().item() / max(
                    q.grad.float().norm().item(), 1e-8
                )
                errors[name] = rms
                assert rms < TOLERANCES["model_gradient_relative_l2"], (name, rms)
            model.zero_grad(set_to_none=True)
            model.config.checkpoint = True
            with torch.autocast("cuda", dtype=torch.bfloat16):
                replay = model(tokens, target)
            replay.backward()
            # Exact checkpointing is checked against this same Tensor implementation.
            baseline = Model(replace(c, checkpoint=False), ops).cuda()
            baseline.load_state_dict(model.state_dict())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                baseline(tokens, target).backward()
            for (name, p), (_, q) in zip(
                model.named_parameters(), baseline.named_parameters()
            ):
                torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0)
            results.append(
                {
                    "architecture": architecture,
                    "rotary": rotary,
                    "loss_error": error,
                    "gradient_relative_l2": errors,
                    "checkpoint_exact": True,
                }
            )
            print(
                "Gradient fixture passed:", architecture, "rotary", rotary, flush=True
            )
    save("gradients", {"status": "passed", "fixtures": results, "coverage": ops.report})


def training(ops, steps, lr=0.001):
    assert steps >= 1000, "qualification requires at least 1000 steps"
    rows = []
    for architecture in ("llt", "naive"):
        setup()
        c = Config(architecture=architecture, checkpoint=True)
        model = Model(c, ops).cuda()
        reference = Model(c).cuda()
        reference.load_state_dict(model.state_dict())
        optimizer = AdamW(model.parameters(), ops, lr=lr, max_norm=1.0)
        baseline = torch.optim.AdamW(reference.parameters(), lr=lr, foreach=False)
        generator = torch.Generator().manual_seed(51)
        losses = []
        drift = 0.0
        resumed = None
        resume_error = None
        torch.cuda.synchronize()
        started = time.perf_counter()
        for step in range(steps):
            tokens, target = batch(generator, c)
            actual = train_step(model, optimizer, tokens, target)
            expected = train_step(reference, baseline, tokens, target, True)
            a, e = actual.item(), expected.item()
            assert math.isfinite(a) and math.isfinite(e), (step, a, e)
            drift = max(drift, abs(a - e))
            assert abs(a - e) < TOLERANCES["training_loss_absolute"], (step, a, e)
            if step == steps // 2:
                checkpoint_path = (
                    ROOT / "build/llt-qualification" / f"{architecture}-resume.pt"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "generator": generator.get_state(),
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(),
                        "step": step + 1,
                        "scheduler": {"lr": lr},
                        "config": asdict(c),
                    },
                    checkpoint_path,
                )
                state = torch.load(checkpoint_path, weights_only=False)
                resumed = Model(c, ops).cuda()
                resumed.load_state_dict(state["model"])
                resumed_optimizer = AdamW(resumed.parameters(), ops, lr=lr)
                resumed_optimizer.load_state_dict(state["optimizer"])
                resumed_generator = torch.Generator()
                resumed_generator.set_state(state["generator"])
                torch.set_rng_state(state["torch_rng"])
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            elif resumed is not None:
                rt, ry = batch(resumed_generator, c)
                torch.testing.assert_close(tokens, rt, atol=0, rtol=0)
                train_step(resumed, resumed_optimizer, rt, ry)
                if step == steps // 2 + 5:
                    resume_error = max(
                        (p - q).abs().max().item()
                        for p, q in zip(model.parameters(), resumed.parameters())
                    )
                    assert resume_error <= TOLERANCES["resume_parameter_absolute"], (
                        resume_error
                    )
                    del resumed, resumed_optimizer
                    resumed = None
            if step % 100 == 0 or step == steps - 1:
                losses.append({"step": step + 1, "tensor": a, "torch": e})
                print(
                    architecture,
                    "step",
                    step + 1,
                    "Tensor loss",
                    round(a, 5),
                    "Torch loss",
                    round(e, 5),
                    flush=True,
                )
        rms = math.sqrt(
            sum(
                (p - q).square().sum().item()
                for p, q in zip(model.parameters(), reference.parameters())
            )
            / sum(p.numel() for p in model.parameters())
        )
        assert rms < TOLERANCES["training_parameter_rms"], rms
        rows.append(
            {
                "config": asdict(c),
                "steps": steps,
                "learning_rate": lr,
                "loss_samples": losses,
                "maximum_loss_drift": drift,
                "final_parameter_rms_difference": rms,
                "resume_maximum_parameter_error": resume_error,
                "elapsed_seconds": time.perf_counter() - started,
            }
        )
        del model, reference, optimizer, baseline
        gc.collect()
        torch.cuda.empty_cache()
    save("training", {"status": "passed", "runs": rows, "coverage": ops.report})


def cold():
    import shutil

    cache = ROOT / "build/llt-qualification/cold-artifacts"
    if cache.exists():
        shutil.rmtree(cache)
    ops = Operators(cache)
    q = torch.randn(
        1, 4, 49, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    c = torch.randn(1, 1, 49, 64, device="cuda", dtype=q.dtype, requires_grad=True)

    def call():
        out = ops.attention(q, c, c, causal=True, scale=0.125)
        return torch.autograd.grad(out, (q, c), torch.ones_like(out))

    started = time.perf_counter()
    actual = call()
    torch.cuda.synchronize()
    cold_ms = (time.perf_counter() - started) * 1000
    assert not any(a["cache_hit"] for a in ops.report["artifacts"])
    reloaded = Operators(cache)
    out = reloaded.attention(q, c, c, causal=True, scale=0.125)
    replay = torch.autograd.grad(out, (q, c), torch.ones_like(out))
    for a, b in zip(actual, replay):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert all(a["cache_hit"] for a in reloaded.report["artifacts"])
    values = []
    for _ in range(9):
        started = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000)
    save(
        "cold",
        {
            "status": "passed",
            "cold_compile_and_first_backward_wall_ms": cold_ms,
            "warm_wall_samples_ms": values,
            "warm_wall_median_ms": statistics.median(values),
            "reload_exact": True,
            "reload_cache_hits": True,
            "coverage": ops.report,
        },
    )
    print("Cold/cache reload qualified:", round(cold_ms, 2), "ms cold", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("gradients", "training", "cold"))
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()
    setup()
    ops = Operators(ROOT / "build/llt-qualification/artifacts")
    if args.phase == "cold":
        cold()
    elif args.phase == "gradients":
        gradients(ops)
    else:
        training(ops, args.steps, args.lr)


if __name__ == "__main__":
    main()
