"""Build an opt-in compact expert schedule beside an existing prefill bundle."""
import hashlib
import json
from pathlib import Path
import shutil
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.artifacts import identity
from tensor_llm.qwen35.compact_prefill import implementation_hashes
from .build import build_artifact, needs_build


def produce(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out:
        raise ValueError('compact output must be separate from the control bundle')
    out.mkdir(parents=True, exist_ok=True)
    for p in source.iterdir():
        if p.is_file(): shutil.copy2(p, out/p.name)
    manifest = json.loads((source/'prefill.json').read_text())
    rows = manifest['slots']*manifest['chunk']
    m = manifest['schedule']['block_m']
    record = dict(max_tiles=(rows*8+m-1)//m+255, block_m=m, projections={})
    def add(kind, parameters, module, factory, specialization):
        key = identity(kind, parameters)
        entry = out/(key+'.py')
        artifact = entry.with_suffix('.tbin')
        text = export_source(module, factory, specialization, dependencies=('tensor.compiler.entry',))
        if needs_build(entry, artifact, text, manifest['target']):
            entry.write_text(text)
            artifact.unlink(missing_ok=True)
            build_artifact(entry, artifact, target=manifest['target'])
        manifest['kernels'][key] = dict(kind=kind, parameters=parameters, path=artifact.name,
                                        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        return key
    p = dict(rows=rows, block_m=m)
    record['tile_map'] = add('expert_tile_map', p, 'tensor_llm.qwen35.kernels.compact_experts',
                             'tile_map_kernel', p)
    for logical, row in list(manifest['kernels'].items()):
        if row['kind'] != 'fp8_experts': continue
        p = row['parameters']
        schedule = dict(rows=rows, k=p['k'], o=p['o'], routed_input=p['routed_input'],
                        block_m=m, threads=256 if m>=32 else 128, compact=True)
        selected = add('compact_fp8_experts', p, 'tensor_llm.qwen35.kernels.prefill',
                       'expert_kernel', schedule)
        record['projections'][logical] = selected
    manifest['compact_experts'] = record
    manifest['compact_expert_implementation'] = implementation_hashes()
    (out/'prefill.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(produce(args.source, args.out))
