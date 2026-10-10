"""Bounded serving beam search, ranked by complete C8 client elapsed time.

Kernel qualification and serial/model qualification are reported separately.
An experimental winner never becomes a qualified model throughput claim.
"""
from benchmarks.qwen35.modal_hopper_long import app, prepare_long
from benchmarks.qwen35.modal_compact import measure_compact


def assess(result):
    """Only the unchanged, completed load can enter the experimental beam."""
    import math
    report = result.get('client_report', {})
    if (report.get('workload_sha256') != '6e012fe014f8fc86d58d0065862c62e77e1374fccba612a7ab5d54c1679db44a'
            or result.get('device', {}).get('name') != 'NVIDIA H200'):
        return None
    servers = report.get('servers', [])
    if result.get('status') != 'measured-experimental' or len(servers) != 1:
        return None
    points = servers[0].get('points', [])
    if len(points) != 1 or points[0].get('concurrency') != 8:
        return None
    metrics = points[0]['summary']
    if (metrics['completed'], metrics['failed'], metrics['prompt_tokens'],
            metrics['output_tokens']) != (8, 0, 256000, 128000):
        return None
    checks = result.get('same_state_quality', [])
    if (result.get('kernel_tests_exit_code') != 0 or len(checks) != 2
            or not all(c['passed'] for c in checks)):
        return None
    elapsed = metrics['elapsed_seconds']
    if not math.isfinite(elapsed) or elapsed <= 0:
        return None
    if not math.isclose(metrics['output_tokens_per_second'], 128000/elapsed, rel_tol=1e-10):
        return None
    return elapsed


@app.local_entrypoint()
def beam_main(inventory_file: str, out: str = 'build/qwen35-h200-beam',
              candidates: int = 4, width: int = 2):
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    from tensor.compiler.search import ScheduleSearch, key, neighbors

    inventory = json.loads(Path(inventory_file).read_text())
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    prefix_choices=['mma-experts', 'bf16-pairs', 'bf16-async']
    if 'bf16-pairs-attention' in inventory['prefixes']:prefix_choices.append('bf16-pairs-attention')
    spaces = {'native': {'prefix': tuple(prefix_choices),
                         'window': (8, 32, 64, 128), 'fallback': (1, 3)}}
    seeds = inventory.get('seeds', [dict(family='native', prefix='bf16-async',
                                        window=64, fallback=3)])
    search = ScheduleSearch(seeds, spaces=spaces, width=width)
    evidence = []

    def retain(config, result, location):
        elapsed = assess(result)
        row = dict(configuration=config, result=location,
                   kernel_qualified=elapsed is not None,
                   serial_qualified=result.get('verification_serial_quality', {}).get('passed', False),
                   model_qualified=result.get('model_throughput_qualified', False))
        if elapsed is not None:
            search.record(config, elapsed)
            row.update(elapsed_seconds=elapsed, output_tokens_per_second=128000/elapsed)
        evidence.append(row)
        # Refresh the frontier when a better measured configuration arrives;
        # a small budget should not finish exploring an obsolete beam first.
        ranked=sorted(search.results,key=lambda row:row[0])[:width]
        pending=[];queued=set()
        for candidate in [p for _,c in ranked for p in neighbors(c,spaces)]+search.pending:
            identity=key(candidate)
            if identity not in search.seen and identity not in queued:
                queued.add(identity);pending.append(candidate)
        search.pending=pending

    for previous in inventory.get('observations', []):
        retain(previous['configuration'], json.loads(Path(previous['result']).read_text()), previous['result'])
        search.seen.add(key(previous['configuration']))

    def save():
        ranked = sorted((r for r in evidence if r['kernel_qualified']),
                        key=lambda r: r['elapsed_seconds'])
        report = dict(schema='tensor.qwen35-serving-beam.v1', width=width,
                      objective='whole-client elapsed seconds for C8 32K/16K; minimize',
                      spaces=spaces, evidence=evidence, experimental_beam=ranked[:width],
                      qualified_beam=[r for r in ranked if r['serial_qualified'] and r['model_qualified']][:width],
                      seen=[dict(k) for k in sorted(search.seen)],
                      pending=search.pending)
        (root/'beam.json').write_text(json.dumps(report, indent=2)+'\n')
    save()
    for index in range(candidates):
        try:config = search.next()
        except StopIteration:break
        save()
        name = f"{index:02d}-{config['prefix']}-w{config['window']}-f{config['fallback']}"
        candidate = root/name
        candidate.mkdir(exist_ok=False)
        print('Beam candidate', config, flush=True)
        prefix = json.loads(Path(inventory['prefixes'][config['prefix']]).read_text())
        window = config['window']
        if str(window) in inventory['windows']:
            verification = json.loads(Path(inventory['windows'][str(window)]).read_text())
            if prefix['base']['checkpoint'] != verification['base']['checkpoint']:
                raise ValueError('beam checkpoint mismatch')
            ready = dict(prefix, base=dict(prefix['base'], paths=dict(prefix['base']['paths'],
                **{k: verification['base']['paths'][k] for k in ('verify', 'repair')})),
                verification_chunk=window,
                verification_source_identity=verification['source_identity'])
        else:
            ready = prepare_long.remote(prefix, window, config['prefix'])
            saved = root/f'verification{window}.json'
            saved.write_text(json.dumps(ready, indent=2)+'\n')
            inventory['windows'][str(window)] = str(saved)
        ready.update(fallback_proposals=config['fallback'], profile_phases=False)
        (candidate/'prepared.json').write_text(json.dumps(ready, indent=2)+'\n')
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-beam-'+name
        (candidate/'run.json').write_text(json.dumps(dict(run_id=run_id, volume='tensor-qwen35-h200'), indent=2)+'\n')
        try:
            result = measure_compact.remote(ready, run_id)
        except Exception as error:
            evidence.append(dict(configuration=config, kernel_qualified=False,
                                 serial_qualified=False, model_qualified=False, error=str(error)))
        else:
            result_file = candidate/'summary.json'
            result_file.write_text(json.dumps(result, indent=2)+'\n')
            retain(config, result, str(result_file))
            print('Beam elapsed seconds', assess(result), flush=True)
        save()
