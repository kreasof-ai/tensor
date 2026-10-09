"""Client timing and throughput aggregation with explicit denominators."""
from __future__ import annotations

import math


def distribution(values):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None}

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99)}


def summarize(rows, elapsed, *, ttft_slo=None, tpot_slo=None):
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("elapsed must be positive and finite")
    successful = [row for row in rows if row["status"] == "completed"]
    output = sum(row["output_tokens"] for row in successful)
    prompt = sum(row["prompt_tokens"] for row in successful)
    good = [row for row in successful
            if (ttft_slo is None or row["ttft_seconds"] <= ttft_slo)
            and (tpot_slo is None or row["tpot_seconds"] is not None
                 and row["tpot_seconds"] <= tpot_slo)]
    return {"requests": len(rows), "completed": len(successful),
            "failed": len(rows) - len(successful), "elapsed_seconds": elapsed,
            "prompt_tokens": prompt, "output_tokens": output,
            "observed_output_tokens_including_failures": sum(row["output_tokens"] for row in rows),
            "output_tokens_per_second": output / elapsed,
            "input_tokens_per_second": prompt / elapsed,
            "total_tokens_per_second": (prompt + output) / elapsed,
            "requests_per_second": len(successful) / elapsed,
            "goodput_output_tokens_per_second": sum(row["output_tokens"] for row in good) / elapsed,
            "goodput_requests_per_second": len(good) / elapsed,
            "ttft_seconds": distribution(row["ttft_seconds"] for row in successful),
            "service_ttft_seconds": distribution(row["service_ttft_seconds"] for row in successful),
            "tpot_seconds": distribution(row["tpot_seconds"] for row in successful),
            "latency_seconds": distribution(row["latency_seconds"] for row in successful),
            "client_queue_seconds": distribution(row["client_queue_seconds"] for row in rows),
            "stream_chunk_gap_seconds": distribution(gap for row in successful
                                                       for gap in row["stream_chunk_gaps_seconds"]),
            "single_token_gap_seconds": distribution(gap for row in successful
                                                       for gap in row["single_token_gaps_seconds"])}
