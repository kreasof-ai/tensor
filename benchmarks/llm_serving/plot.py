"""Plot retained serving points without inventing failed or missing results."""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from .runner import REPORT_SCHEMA, atomic_json
from .workload import digest


def peak_memory(directory, point):
    path = directory / point["telemetry_file"]
    if not path.exists():
        return math.nan
    peaks = []
    for line in path.read_text().splitlines():
        sample = json.loads(line)
        devices = sample.get("gpu", {}).get("devices", [])
        if devices:
            peaks.append(sum(device["used_mib"] for device in devices) / 1024)
    return max(peaks, default=math.nan)


def load_points(paths, *, allow_unmatched=False):
    reports = [(Path(path), json.loads(Path(path).read_text())) for path in paths]
    if not reports or any(report.get("schema") != REPORT_SCHEMA for _, report in reports):
        raise ValueError("provide serving report.json files with the supported schema")
    identities = {(report["workload_sha256"], report["comparison_group"], digest(report["protocol"]))
                  for _, report in reports}
    if len(identities) != 1:
        raise ValueError("plots require the same workload hash, comparison group and timing protocol")
    matched = set()
    curves, coverage, names = [], [], set()
    for path, report in reports:
        for server in report["servers"]:
            config = server["configuration"]
            name = config["name"]
            if name in names:
                raise ValueError("duplicate server name across reports; use distinct names for configuration variants")
            names.add(name)
            groups = {}
            coverage.append({"name": name, "disposition": server.get("disposition"),
                             "reason": server.get("reason"), "points": server["points"]})
            for point in server["points"]:
                if point["disposition"] != "measured":
                    continue
                groups.setdefault(point["concurrency"], []).append(point)
            if not groups:
                continue
            devices = server.get("hardware_snapshot", {}).get("devices", [])
            observed_hardware = tuple(sorted((device["name"], device["total_mib"], device["driver"])
                                              for device in devices))
            matched.add((*[config[key] for key in ("model_revision", "weight_format", "kv_dtype",
                                                   "state_dtype", "prefix_cache", "speculative",
                                                   "hardware", "cpu_offload")], observed_hardware,
                         report["settings"]["ttft_slo_seconds"], report["settings"]["tpot_slo_seconds"]))
            samples = []
            for concurrency, points in sorted(groups.items()):
                samples.append({"concurrency": concurrency, "points": points,
                                "memory": [peak_memory(path.parent / name, point) for point in points]})
            curves.append({"name": name, "config": config, "samples": samples})
    if not curves:
        raise ValueError("no length/protocol-qualified measured points to plot")
    unmatched = len(matched) > 1
    if unmatched and not allow_unmatched:
        raise ValueError("precision, revision, cache/speculation, hardware, offloading or SLO configuration differs; use separately qualified arms, or --allow-unmatched for a labeled exploratory plot")
    return curves, coverage, unmatched, any(report.get("synthetic") for _, report in reports)


def values(sample, metric, statistic=None):
    results = []
    for point in sample["points"]:
        value = point["summary"][metric]
        if statistic:
            value = value[statistic]
        if value is not None and math.isfinite(value):
            results.append(value)
    return results


def mean_error(observations, scale=1):
    if not observations:
        return math.nan, 0
    return statistics.mean(observations) * scale, (statistics.stdev(observations) * scale
                                                  if len(observations) > 1 else 0)


def plot(paths, out, *, allow_unmatched=False, latency_stat="p95"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if latency_stat not in ("mean", "p50", "p95", "p99"):
        raise ValueError("unsupported latency statistic")
    curves, coverage, unmatched, synthetic = load_points(paths, allow_unmatched=allow_unmatched)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "plot-coverage.json", {"schema": "tensor.llm-serving-plot.v1",
                                            "reports": [str(Path(path)) for path in paths],
                                            "unmatched": unmatched, "synthetic": synthetic,
                                            "latency_statistic": latency_stat,
                                            "servers": coverage})
    panels = [("output_tokens_per_second", None, "Output tokens / s", 1),
              ("ttft_seconds", latency_stat, f"TTFT {latency_stat} (s)", 1),
              ("tpot_seconds", latency_stat, f"TPOT {latency_stat} (ms)", 1000),
              ("latency_seconds", latency_stat, f"Request latency {latency_stat} (s)", 1),
              ("input_tokens_per_second", None, "Input tokens / s (whole run)", 1),
              ("total_tokens_per_second", None, "Input + output tokens / s", 1),
              ("goodput_output_tokens_per_second", None, "SLO-qualified output tokens / s", 1),
              ("memory", None, "Sampled peak GPU memory (GiB)", 1)]
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), layout="constrained")
    colors = {curve["name"]: plt.get_cmap("tab10")(index % 10) for index, curve in enumerate(curves)}
    for ax, (metric, statistic, label, scale) in zip(axes.flat, panels):
        for curve in curves:
            config = curve["config"]
            name = f"{curve['name']} ({config['weight_format']}, KV {config['kv_dtype']})"
            observations = [([value for value in sample["memory"] if math.isfinite(value)]
                             if metric == "memory" else values(sample, metric, statistic))
                            for sample in curve["samples"]]
            pairs = [mean_error(items, scale) for items in observations]
            if not any(math.isfinite(pair[0]) for pair in pairs):
                continue
            ax.errorbar([sample["concurrency"] for sample in curve["samples"]],
                        [pair[0] for pair in pairs], yerr=[pair[1] for pair in pairs],
                        marker="o", capsize=3, color=colors[curve["name"]], label=name)
        ax.set_xscale("log", base=2)
        ticks = sorted({sample["concurrency"] for curve in curves for sample in curve["samples"]})
        ax.set_xticks(ticks, [str(value) for value in ticks])
        ax.set_xlabel("Client concurrency")
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        if not ax.lines:
            ax.text(.5, .5, "Not available", transform=ax.transAxes, ha="center")
    banner = "SYNTHETIC PROTOCOL TEST — NOT GPU PERFORMANCE" if synthetic else "Matched serving workload"
    if unmatched:
        banner += "\nUnmatched formats/configurations; quality equivalence not established"
    fig.suptitle(banner)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", frameon=False)
    for extension in ("svg", "png"):
        fig.savefig(out / f"throughput-latency.{extension}", dpi=160)
    plt.close(fig)

    frontier, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    all_points = []
    for curve in curves:
        pairs = [(mean_error(values(sample, "tpot_seconds", latency_stat), 1000)[0],
                  mean_error(values(sample, "output_tokens_per_second"))[0])
                 for sample in curve["samples"]]
        pairs = [pair for pair in pairs if all(math.isfinite(value) for value in pair)]
        if pairs:
            ax.scatter([pair[0] for pair in pairs], [pair[1] for pair in pairs],
                       color=colors[curve["name"]], label=curve["name"])
            all_points.extend(pairs)
    if not unmatched and all_points:
        nondominated = sorted({point for point in all_points
                               if not any(other[0] <= point[0] and other[1] >= point[1]
                                          and other != point for other in all_points)})
        ax.plot([point[0] for point in nondominated], [point[1] for point in nondominated],
                "k--", alpha=.5, label="Observed frontier (sampled points)")
    ax.set_xlabel(f"Per-request TPOT {latency_stat} (ms)")
    ax.set_ylabel("Output tokens / s")
    ax.set_title(banner)
    ax.grid(alpha=.2)
    if ax.collections or ax.lines:
        ax.legend()
    for extension in ("svg", "png"):
        frontier.savefig(out / f"latency-throughput-frontier.{extension}", dpi=160)
    plt.close(frontier)
    return out
