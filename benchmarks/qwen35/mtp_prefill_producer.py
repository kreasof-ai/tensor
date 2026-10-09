"""Add native MTP input adapters to a selected chunked prefill bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from tensor.compiler.build import build_artifact
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.checkpoint import Qwen35Checkpoint
from tensor_llm.qwen35.artifacts import identity
from tensor_llm.qwen35.mtp.prefill import implementation_hashes


def produce(checkpoint, prefill, out):
    checkpoint = Qwen35Checkpoint(checkpoint, branch='mtp')
    prefill, out = Path(prefill).resolve(), Path(out).resolve()
    if out == prefill or out.is_relative_to(prefill): raise ValueError('use a separate MTP prefill output')
    manifest = json.loads((prefill/'prefill.json').read_text())
    if manifest['schema'] != 'tensor.qwen35-prefill.v1': raise ValueError('unsupported prefill schema')
    out.mkdir(parents=True, exist_ok=True)
    (out/'prefill.json').unlink(missing_ok=True)
    for row in manifest['kernels'].values():
        source = (prefill/row['path']).resolve()
        if not source.is_relative_to(prefill) or hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('prefill artifact checksum mismatch')
        dest = out/row['path']; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    r = manifest['slots']*manifest['chunk']
    for kind, p in (('mtp_join', dict(r=r, c=2048, eps=checkpoint.config.epsilon)),
                    ('mtp_cast', dict(r=r, c=2048)), ('bf16_linear', dict(r=r, k=4096, o=2048))):
        key = identity(kind, p); entry = out/(key+'.py'); artifact = entry.with_suffix('.tbin')
        module, factory, args = ('tensor_llm.qwen35.kernels.mtp', 'make_kernel', (kind, p))
        if kind == 'bf16_linear':
            module, factory, args = ('tensor_llm.qwen35.kernels.matmul', 'bf16_head_kernel', (dict(p, columns=128, depth=128),))
        entry.write_text(export_source(module, factory, *args, dependencies=('tensor.compiler.entry',)))
        artifact.unlink(missing_ok=True)
        build_artifact(entry, artifact, target='sm_89', compiler='nvrtc', nvrtc_home='build/nvrtc-12.9')
        manifest['kernels'][key] = dict(kind=kind, parameters=p, path=artifact.name,
                                      sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        print('MTP prefill adapter', kind, flush=True)
    manifest.update(draft='qwen35-mtp', implementation=implementation_hashes(),
        target_prefill_lineage=dict(path=str(prefill),
            manifest_sha256=hashlib.sha256((prefill/'prefill.json').read_bytes()).hexdigest()),
        speculative_verification=False, full_stress_target_reached=False)
    (out/'prefill.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--prefill', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(produce(a.checkpoint, a.prefill, a.out))
