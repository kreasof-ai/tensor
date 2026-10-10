"""Add compact recurrent-state replay to a checksummed frozen verifier bundle."""
import hashlib
import json
import shutil
from pathlib import Path
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.artifacts import identity
from tensor_llm.qwen35.speculative.recompute import implementation_hashes
from .build import build_artifact


def produce(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    manifest = json.loads((source/'prefill.json').read_text())
    if source == out or manifest.get('chunk') not in (64,128) or 'state_recompute' in manifest:
        raise ValueError('use a separate bundle and a frozen 64- or 128-token verifier')
    out.mkdir(parents=True, exist_ok=True)
    for row in manifest['kernels'].values():
        path = (source/row['path']).resolve()
        if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('frozen verifier checksum mismatch')
        destination = out/row['path']; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    if 'attention_workspace' in manifest:
        child = (source/manifest['attention_workspace']).resolve()
        if not child.is_relative_to(source): raise ValueError('workspace outside source bundle')
        shutil.copytree(child, out/manifest['attention_workspace'], dirs_exist_ok=True)
    p = dict(slots=manifest['slots'], chunk=manifest['chunk'])
    for factory in ('save_scan', 'save_conv', 'restore_scan', 'restore_conv'):
        kind = 'recompute_'+factory
        name = identity(kind, p); entry = out/(name+'.py'); artifact = entry.with_suffix('.tbin')
        entry.write_text(export_source('tensor_llm.qwen35.kernels.recompute', factory, p,
                                      dependencies=('tensor.compiler.entry',)))
        artifact.unlink(missing_ok=True); build_artifact(entry, artifact, target='sm_90')
        manifest['kernels'][name] = dict(kind=kind,parameters=p,path=artifact.name,
                                        sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
    manifest['state_recompute'] = 'accepted-prefix-recompute'
    manifest['state_recompute_implementation'] = implementation_hashes()
    manifest['state_recompute_producer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'prefill.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return out
