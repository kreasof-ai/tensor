"""Summarize retained short-context AR/MTP replays; never run GPU inference."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean

from benchmarks.qwen35.dense_scaling import CONCURRENCIES, REVISION

PROFILES = {
    "native_mtp": "tensor-mtp-v2-w2",
    "vllm_ar": "vllm-ar-stream4",
    "vllm_mtp": "vllm-mtp-w2-stream4",
    "sglang_ar": "sglang-ar-stream4-mem50",
    "sglang_mtp": "sglang-mtp-w2-stream4",
}


def summarize(root: Path):
    rows = []
    for concurrency in CONCURRENCIES:
        row = {"concurrency": concurrency, "profiles": {}}
        workload_hashes = set()
        for name, profile in PROFILES.items():
            path = root / "results" / profile / f"c{concurrency}" / "report.json"
            if not path.exists():
                if name == "native_mtp" and concurrency in (2, 4, 16, 64):
                    continue
                raise ValueError(f"missing required replay: {path}")
            report = json.loads(path.read_text())
            if report["status"] != "completed" or len(report["servers"]) != 1:
                raise ValueError(f"incomplete replay: {path}")
            server = report["servers"][0]
            config = server["configuration"]
            if (
                config["model_revision"] != REVISION
                or config["weight_format"] != "BF16"
                or config["kv_dtype"] != "bfloat16"
                or config["state_dtype"] != "float32"
                or config["prefix_cache"]
                or config["cpu_offload"] != "none"
                or config["speculative"] != name.endswith("mtp")
            ):
                raise ValueError(f"unexpected comparison configuration: {path}")
            points = server["points"]
            if len(points) != 2 or {p["repeat"] for p in points} != {0, 1}:
                raise ValueError(f"two distinct repeats required: {path}")
            for point in points:
                s = point["summary"]
                expected = max(16, 4 * concurrency)
                if (
                    point["concurrency"] != concurrency
                    or point["disposition"] != "measured"
                    or s["failed"]
                    or s["completed"] != expected
                    or s["output_tokens"] != expected * 256
                ):
                    raise ValueError(f"invalid counts or failed point: {path}")
                records_path = path.parent / config["name"] / point["requests_file"]
                if records_path.exists():
                    records_text = records_path.read_text()
                else:
                    records_text = gzip.decompress(
                        Path(str(records_path) + ".gz").read_bytes()
                    ).decode()
                records = [json.loads(line) for line in records_text.splitlines()]
                if len(records) != expected or any(
                    record["status"] != "completed"
                    or record["output_tokens"] != 256
                    or sum(event["delta_tokens"] for event in record["token_events"])
                    != 256
                    for record in records
                ):
                    raise ValueError(f"client stream counts disagree: {records_path}")
                observed_rate = (
                    sum(r["output_tokens"] for r in records) / s["elapsed_seconds"]
                )
                if abs(observed_rate - s["output_tokens_per_second"]) > 1e-6:
                    raise ValueError(f"throughput denominator disagrees: {path}")
            workload_hashes.add(report["workload_sha256"])
            samples = [p["summary"] for p in points]
            rates = [s["output_tokens_per_second"] for s in samples]
            row["profiles"][name] = {
                "profile": profile,
                "report": str(path.relative_to(root)),
                "output_tokens_per_second": mean(rates),
                "repeat_output_tokens_per_second": rates,
                "mean_ttft_ms": 1000 * mean(s["ttft_seconds"]["mean"] for s in samples),
                "failures": sum(s["failed"] for s in samples),
            }
        if len(workload_hashes) != 1:
            raise ValueError(f"C{concurrency}: different workload hashes")
        row["workload_sha256"] = workload_hashes.pop()
        row["mtp_vs_ar"] = {
            engine: row["profiles"][f"{engine}_mtp"]["output_tokens_per_second"]
            / row["profiles"][f"{engine}_ar"]["output_tokens_per_second"]
            for engine in ("vllm", "sglang")
        }
        rows.append(row)
    return {
        "model_revision": REVISION,
        "metric": "mean of two whole HTTP replay output token rates, including fill/drain",
        "native_model_quality_qualified": False,
        "baseline_serial_quality": json.loads(
            (root / "mtp-serial-quality.json").read_text()
        ),
        "points": rows,
    }


def render(summary, path, preview):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
    styles = {
        "native_mtp": ("Native V2 MTP (retained, experimental)", "#d87818", "none"),
        "vllm_ar": ("vLLM AR", "#2874a6", "--"),
        "vllm_mtp": ("vLLM MTP", "#2874a6", "-"),
        "sglang_ar": ("SGLang AR", "#278747", "--"),
        "sglang_mtp": ("SGLang MTP", "#278747", "-"),
    }
    for name, (label, color, line) in styles.items():
        points = [r for r in summary["points"] if name in r["profiles"]]
        x = [r["concurrency"] for r in points]
        for axis, metric in zip(axes, ("output_tokens_per_second", "mean_ttft_ms")):
            axis.plot(
                x,
                [r["profiles"][name][metric] for r in points],
                marker="o",
                markersize=4,
                linestyle=line,
                color=color,
                label=label,
            )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_xticks(CONCURRENCIES, [str(c) for c in CONCURRENCIES])
        axis.set_xlabel("Concurrency")
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Output tok/s (whole replay)")
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_ylabel("Mean client TTFT (ms)")
    axes[1].set_yscale("log")
    fig.suptitle("Qwen3.5-0.8B · one L40S · AR versus one-draft official MTP")
    fig.supxlabel(
        "66–81 input / 256 output tokens · BF16 weights/KV · FP32 state · "
        "two repeats · native qualification open; AR/MTP output checks in report",
        fontsize=8,
    )
    fig.savefig(path)
    preview.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview, dpi=160)
    plt.close(fig)
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("docs/research/data/qwen35-08b-local-scaling")
    )
    parser.add_argument(
        "--plot", type=Path, default=Path("docs/research/qwen35-08b-local-scaling.svg")
    )
    parser.add_argument(
        "--preview", type=Path, default=Path("build/qwen35-08b-local-scaling.png")
    )
    args = parser.parse_args()
    summary = summarize(args.data)
    (args.data / "mtp-comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    render(summary, args.plot, args.preview)


if __name__ == "__main__":
    main()
