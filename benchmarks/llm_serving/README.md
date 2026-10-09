# Matched LLM serving harness

Run one immutable token-ID workload against independently installed vLLM,
SGLang and llama.cpp servers. The client measures streamed HTTP requests;
it does not import those engines or change Tensor runtime dependencies.

See the [runbook](../../docs/guides/llm-serving-benchmarks.md) for environment
setup, manifests, memory limits, timing definitions and comparison boundaries.
All commands run from the repository root.

The [real L40S pilot](../../docs/research/llm-serving-l40s-pilot.md) retains
Qwen3-0.6B results from all three engines, plots and raw records. It qualifies
the harness separately from the planned 35B long-context benchmark.

```sh
uv venv --python 3.12 build/llm-bench-client
uv pip install --python build/llm-bench-client/bin/python -r benchmarks/llm_serving/requirements.txt

# Exercise all protocols and plotting without models or a GPU.
build/llm-bench-client/bin/python -m benchmarks.llm_serving.smoke --out build/llm-serving-smoke

# Prepare a pilot. Replace the revision with an actual checkpoint commit.
build/llm-bench-client/bin/python -m benchmarks.llm_serving prepare \
  --tokenizer Qwen/Qwen3.5-35B-A3B-FP8 --revision CHECKPOINT_COMMIT \
  --input-lengths 512 --output-tokens 128 --requests 128 --warmup 1 \
  --out build/llm-serving/pilot.json

# Copy servers.example.json, fill version/revision/path fields and audit flags.
build/llm-bench-client/bin/python -m benchmarks.llm_serving run \
  --servers build/llm-serving/servers.json --workload build/llm-serving/pilot.json \
  --concurrency 1 2 4 8 16 32 --repeats 3 --out build/llm-serving/pilot-run

build/llm-bench-client/bin/python -m benchmarks.llm_serving plot \
  build/llm-serving/pilot-run/report.json --allow-unmatched \
  --out build/llm-serving/pilot-plots
```

The smoke plots are prominently labeled **synthetic protocol tests, not GPU
performance**. Real server runs require pinned provenance, tokenizer agreement,
exact prompt/output counts and complete streams. A GGUF versus FP8/BF16 cache
overlay requires `--allow-unmatched` and carries an explicit comparability label.
Failed and unsupported cells are retained in reports and plot coverage.

Use the runbook's explicit 32,000-input/16,000-output workload after the pilot
passes. Neither request generation nor plotting establishes checkpoint quality
or reproduces Netra's undisclosed engine configuration.
