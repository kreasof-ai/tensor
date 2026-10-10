"""Qualify larger prefill dense tiles against the unchanged model and workload."""
import modal
from benchmarks.qwen35.modal_h200 import image, volume
from benchmarks.qwen35.modal_compact import app as replay_app, measure_compact

app = modal.App('tensor-qwen35-h200-dense-m128')
app.include(replay_app)


@app.function(image=image, cpu=16, memory=32768, timeout=900,
              volumes={'/cache': volume}, scaledown_window=2)
def prepare(prepared):
    import hashlib, json, os, shutil
    from pathlib import Path
    from tensor.compiler.entry import export_source
    from benchmarks.qwen35.build import build_artifact
    os.chdir('/workspace'); volume.reload()
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path('packages/tensor-llm/src/tensor_llm/qwen35/kernels/hopper_dense.py'))}
    identity = hashlib.sha256(json.dumps(dict(prepared=prepared, sources=hashes),
                                        sort_keys=True).encode()).hexdigest()[:16]
    root = Path('/cache/dense-m128')/identity; root.mkdir(parents=True, exist_ok=True)
    source = Path(prepared['hopper']).resolve(); selected = root/'hopper'
    manifest = json.loads((source/'hopper.json').read_text())
    for row in manifest['kernels'].values():
        path = (source/row['path']).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('source Hopper artifact checksum mismatch')
    shutil.copytree(source, selected, dirs_exist_ok=True)
    original = json.loads((Path(prepared['prefill'])/'prefill.json').read_text())
    count = 0
    for key, row in manifest['kernels'].items():
        if row['kind'] != 'fp8_linear': continue
        schedule = dict(original['kernels'][key]['parameters'], block_m=128,
                        columns=128, threads=256, packed_gather=True, stages=1,
                        mma_reduction=32, mma_reorder=True, async_mma=False)
        artifact = selected/row['path']; entry = artifact.with_suffix('.py')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.hopper_dense',
                         'make_kernel', schedule, dependencies=('tensor.compiler.entry',)))
        artifact.unlink(); build_artifact(entry, artifact, target='sm_90a')
        row['sha256'] = hashlib.sha256(artifact.read_bytes()).hexdigest(); count += 1
        print('Prepared M128 prefill dense', schedule, flush=True)
    if not count: raise ValueError('prefill dense projections absent')
    manifest['dense_schedule'] = dict(block_m=128, columns=128, threads=256,
                                     producer_sources=hashes)
    (selected/'hopper.json').write_text(json.dumps(manifest, indent=2)+'\n')
    result = dict(prepared, hopper=str(selected), source_identity=identity,
                  dense_prefill_block_m=128, dense_prefill_source_hashes=hashes)
    (root/'prepared.json').write_text(json.dumps(result, indent=2)+'\n')
    volume.commit(); return result


@app.local_entrypoint()
def main(prepared_file: str, out: str='build/qwen35-h200-dense-m128', prepare_only: bool=False,
         resident_speculative_graphs: bool=False):
    import json
    from datetime import datetime, timezone
    from pathlib import Path
    root = Path(out); root.mkdir(parents=True, exist_ok=True)
    base = json.loads(Path(prepared_file).read_text())
    if resident_speculative_graphs:base=dict(base,resident_speculative_graphs=True)
    ready = prepare.remote(base)
    (root/'prepared.json').write_text(json.dumps(ready, indent=2)+'\n')
    if prepare_only: return
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-dense-m128-'+ready['source_identity']
    (root/'run.json').write_text(json.dumps(dict(run_id=run_id, volume='tensor-qwen35-h200'), indent=2)+'\n')
    result = measure_compact.remote(ready, run_id)
    (root/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
