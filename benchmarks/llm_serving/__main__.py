"""Prepare a workload, run isolated serving engines, or plot retained reports."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from . import workload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="prepare immutable exact token-ID requests")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--tokenizer", help="Hugging Face tokenizer name or local path")
    source.add_argument("--token-pool", type=Path, help="JSON with token_ids and tokenizer provenance")
    prepare.add_argument("--revision", help="pinned tokenizer revision")
    prepare.add_argument("--input-lengths", type=int, nargs="+", default=[512])
    prepare.add_argument("--output-tokens", type=int, default=128)
    prepare.add_argument("--requests", type=int, default=128)
    prepare.add_argument("--warmup", type=int, default=1)
    prepare.add_argument("--seed", type=int, default=20261009)
    prepare.add_argument("--request-rate", type=float, help="Poisson offered arrivals; omitted means closed-loop")
    prepare.add_argument("--out", type=Path, required=True)
    run = commands.add_parser("run", help="run the same requests against each selected engine")
    run.add_argument("--servers", type=Path, required=True)
    run.add_argument("--workload", type=Path, required=True)
    run.add_argument("--engines", nargs="+", help="engine names or manifest labels; default all")
    run.add_argument("--concurrency", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    run.add_argument("--repeats", type=int, default=1)
    run.add_argument("--external", action="store_true", help="connect to one already-running engine")
    run.add_argument("--timeout", type=float, default=3600)
    run.add_argument("--startup-timeout", type=float, default=1800)
    run.add_argument("--telemetry-interval", type=float, default=2)
    run.add_argument("--ttft-slo", type=float, help="goodput TTFT limit in seconds")
    run.add_argument("--tpot-slo", type=float, help="goodput TPOT limit in seconds")
    run.add_argument("--api-key-env", default="LLM_BENCH_API_KEY")
    run.add_argument("--out", type=Path, required=True)
    plot = commands.add_parser("plot", help="plot matched reports; failed cells remain in coverage JSON")
    plot.add_argument("reports", type=Path, nargs="+")
    plot.add_argument("--allow-unmatched", action="store_true", help="label exploratory cross-format/config plots")
    plot.add_argument("--latency-stat", choices=("mean", "p50", "p95", "p99"), default="p95",
                      help="latency statistic for curves and frontier; default p95")
    plot.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if args.tokenizer:
                pool, provenance = workload.token_pool(args.tokenizer, args.revision)
            else:
                data = json.loads(args.token_pool.read_text())
                pool, provenance = data["token_ids"], data["tokenizer"]
            value = workload.prepare(pool, provenance, lengths=args.input_lengths,
                                     output_tokens=args.output_tokens, requests=args.requests,
                                     warmup=args.warmup, seed=args.seed, request_rate=args.request_rate)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("x") as output:
                json.dump(value, output, separators=(",", ":"), allow_nan=False)
                output.write("\n")
            print(f"Prepared {len(value['requests'])} requests: {args.out}\nsha256={value['sha256']}")
        elif args.command == "run":
            from .runner import load_servers, run as run_workload
            report = asyncio.run(run_workload(
                load_servers(args.servers, args.engines), workload.load(args.workload), args.out,
                engines=args.engines, concurrencies=args.concurrency, repeats=args.repeats,
                external=args.external, timeout=args.timeout, startup_timeout=args.startup_timeout,
                interval=args.telemetry_interval, ttft_slo=args.ttft_slo, tpot_slo=args.tpot_slo,
                api_key_env=args.api_key_env))
            print(f"{report['status']}: {args.out / 'report.json'}")
            return 0 if report["status"] == "completed" else 1
        else:
            from .plot import plot as plot_reports
            plot_reports(args.reports, args.out, allow_unmatched=args.allow_unmatched,
                         latency_stat=args.latency_stat)
            print(f"Plots and coverage: {args.out}")
    except (ValueError, OSError, KeyError, ImportError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
