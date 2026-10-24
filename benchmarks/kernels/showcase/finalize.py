"""Remeasure selected kernels with interleaved providers; render the showcase.

Selection never uses these final timing samples. Original measurements remain
in the JSON. Run only after benchmark.py has finished using the GPU.
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from benchmarks.kernels.showcase import benchmark as b

torch = b.torch


def render(directory, result):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.subplots_adjust(
        left=0.065, right=0.99, bottom=0.18, top=0.90, hspace=0.44, wspace=0.18
    )
    colors = {
        "tensor_searched": "#176b55",
        "tilelang_fixed": "#7c8b9a",
        "tilelang_same_schedule": "#adbac7",
        "triton_searched": "#e68a30",
        "pytorch": "#a06cbe",
        "upstream": "#368ab4",
    }
    names = {
        "tensor_searched": "Tensor searched",
        "tilelang_fixed": "TileLang fixed",
        "tilelang_same_schedule": "TileLang same schedule",
        "triton_searched": "Triton searched",
        "pytorch": "PyTorch",
        "upstream": "FA2 / GGML",
    }
    for ax, kind, title in zip(
        axes.flat,
        ("matmul", "flash_attention", "mla_decode", "gguf"),
        (
            "FP16 matrix multiplication",
            "Flash Attention forward",
            "Dense MLA decode (576 / 512)",
            "GGUF fused dequant + matmul",
        ),
    ):
        cases = [c for c in result["cases"] if c["workload"] == kind]
        providers = list(colors)
        if kind in ("matmul", "mla_decode"):
            providers.remove("upstream")
        width = 0.8 / len(providers)
        for j, provider in enumerate(providers):
            values, low, high = [], [], []
            for case in cases:
                key = (
                    case["pytorch_label"]
                    if provider == "pytorch"
                    else ("upstream_fa2" if kind == "flash_attention" else "ggml_cuda")
                    if provider == "upstream"
                    else provider
                )
                ms = case["measurements"].get(key)
                value = ms["median_us"] if ms else float("nan")
                values.append(value)
                low.append(value - ms["min_us"] if ms else 0)
                high.append(ms["max_us"] - value if ms else 0)
            xs = np.arange(len(cases)) + (j - (len(providers) - 1) / 2) * width
            ax.bar(
                xs,
                values,
                width,
                label=names[provider],
                color=colors[provider],
                yerr=[low, high],
                error_kw={"elinewidth": 0.6, "capsize": 1},
            )
        labels = []
        for case in cases:
            shape = case["shape"]
            if kind == "matmul":
                labels.append("×".join(map(str, shape)))
            elif kind == "flash_attention":
                labels.append(
                    f"S={shape[2]}, D={shape[3]}\n{'causal' if case['causal'] else 'noncausal'}"
                )
            elif kind == "mla_decode":
                labels.append(f"B={shape[0]}, H={shape[1]}\nS={shape[2]}")
            else:
                labels.append(
                    f"{ {2: 'Q4_0', 12: 'Q4_K', 14: 'Q6_K'}[case['encoding']] }\nM={shape[0]}"
                )
        ax.set_xticks(np.arange(len(cases)), labels, fontsize=8)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_ylabel("Latency (µs), lower is better")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=6,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        "Tensor CUDA kernel measurements · NVIDIA L4 (SM89)", fontsize=17, weight="bold"
    )
    fig.text(
        0.5,
        0.035,
        "Warm CUDA graph replay · interleaved provider samples · error bars show min–max · GGUF PyTorch excludes weight decoding · upstream FlashMLA unsupported",
        ha="center",
        fontsize=9,
    )
    for extension in ("png", "svg"):
        fig.savefig(directory / f"showcase.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--akas", type=Path, default=b.ROOT.parent / "akas")
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--samples", type=int, default=21)
    p.add_argument("--fa-python", type=Path)
    p.add_argument("--ggml-library", type=Path)
    args = p.parse_args()
    sys.path.insert(0, str(args.akas.resolve() / "src"))
    result = json.loads((args.run / "results.json").read_text())
    if "final_measurement" in result:
        raise ValueError("This run already contains final interleaved measurements")
    result["final_measurement"] = {
        "method": "capture each provider once, interleave graph replay samples in randomized order",
        "iterations": args.iterations,
        "samples": args.samples,
        "source_sha256": b.sha(__file__),
        "torch": torch.__version__,
    }
    torch.manual_seed(result["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    stream = torch.cuda.Stream()
    target = result["environment"]["gpu"]["sm"]
    with (
        torch.inference_mode(),
        torch.cuda.stream(stream),
        b.tx.Device(stream=stream.cuda_stream) as device,
    ):
        for case in result["cases"]:
            data = b.prepare(case)
            if "selected" not in case:
                continue
            selected = case["selected"]
            calls, cleanup, graphs = {}, [], {}
            for label, evidence, config in (
                ("tensor_searched", selected, selected["config"]),
                (
                    "tensor_default",
                    case["baselines"]["tensor_default"],
                    {}
                    if case["workload"] == "matmul"
                    else b.configs(case, 1, target)[0],
                ),
            ):
                call = b.TensorCall(
                    device,
                    [Path(p) for p in evidence["artifacts"]],
                    b.arguments(case, data, config),
                )
                calls[label] = call
                cleanup.append(call)
            for label in ("tilelang_fixed", "tilelang_same_schedule"):
                if label in case["baselines"]:
                    calls[label], _ = b.tilelang_call(
                        case,
                        case["baselines"][label]["config"],
                        data,
                        args.run / case["id"] / (label + "-final"),
                        target,
                    )
            if "triton_selected" in case:
                calls["triton_searched"] = b.triton_call(
                    case, case["triton_selected"]["config"], data
                )
            pt = case["pytorch_label"]
            if pt in case["measurements"]:
                calls[pt] = data["baseline"]
            case["final_correctness"] = {}
            for label, call in calls.items():
                case["final_correctness"][label] = b.validate(
                    call, data, case, thorough=True
                )
                for _ in range(5):
                    call()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.iterations):
                        call()
                graphs[label] = graph
            # Warm all providers together before comparing them.
            for _ in range(20):
                for graph in graphs.values():
                    graph.replay()
            stream.synchronize()
            samples = {label: [] for label in graphs}
            orders = []
            rng = random.Random(101)
            for _ in range(args.samples):
                order = list(graphs)
                rng.shuffle(order)
                orders.append(order)
                for label in order:
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graphs[label].replay()
                    end.record()
                    end.synchronize()
                    samples[label].append(
                        start.elapsed_time(end) * 1000 / args.iterations
                    )
            case["initial_measurements"] = case["measurements"]
            case["measurements"] = {
                label: {
                    "median_us": statistics.median(xs),
                    "min_us": min(xs),
                    "max_us": max(xs),
                    "samples_us": xs,
                }
                for label, xs in samples.items()
            }
            case["final_provider_orders"] = orders
            external = (
                "fa2"
                if case["workload"] == "flash_attention" and args.fa_python
                else "ggml"
                if case["workload"] == "gguf" and args.ggml_library
                else None
            )
            if external:
                directory = args.run / case["id"]
                inputs = directory / "final-external-inputs.npz"
                np.savez(
                    inputs,
                    **{
                        f"input{i}": t.cpu().numpy()
                        for i, t in enumerate(data["inputs"])
                    },
                    expected=data["reference"]().cpu().numpy(),
                    case=json.dumps(
                        {
                            k: v
                            for k, v in case.items()
                            if k in ("shape", "causal", "encoding")
                        }
                    ),
                )
                command = [
                    str(args.fa_python) if external == "fa2" else sys.executable,
                    str(Path(__file__).with_name("external_baselines.py")),
                    "--kind",
                    external,
                    "--inputs",
                    str(inputs),
                    "--out",
                    str(directory / "final-external.json"),
                    "--iterations",
                    str(args.iterations),
                    "--repetitions",
                    str(args.samples),
                ]
                if external == "ggml":
                    command += ["--library", str(args.ggml_library)]
                with (directory / "final-external.log").open("w") as log:
                    subprocess.run(
                        command, stdout=log, stderr=subprocess.STDOUT, check=True
                    )
                ext = json.loads((directory / "final-external.json").read_text())
                case["measurements"][ext["provider"]] = {
                    k: ext[k] for k in ("median_us", "min_us", "max_us", "samples_us")
                }
                case["baselines"][ext["provider"]] = ext
            for call in cleanup:
                call.close()
            graphs.clear()
            print(
                json.dumps(
                    {
                        "case": case["id"],
                        "us": {
                            k: round(v["median_us"], 3)
                            for k, v in case["measurements"].items()
                        },
                    }
                ),
                flush=True,
            )
            b.write_json(args.run / "final-progress.json", result)
    b.write_json(args.run / "results.json", result)
    b.report(args.run, result)
    render(args.run, result)


if __name__ == "__main__":
    main()
