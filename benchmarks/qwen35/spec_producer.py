"""Produce small-chunk greedy verification and recurrent rollback artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .build import build_artifact, needs_build
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.artifacts import identity
from tensor_llm.qwen35.speculative.verifier import implementation_hashes


def produce(prefill, out):
    prefill, out = Path(prefill).resolve(), Path(out).resolve()
    if out == prefill or out.is_relative_to(prefill): raise ValueError('use a separate verification output')
    manifest = json.loads((prefill/'prefill.json').read_text())
    slots, chunk = manifest['slots'], manifest['chunk']
    if chunk < 2 or chunk > 128: raise ValueError('verification chunk must be between 2 and 128')
    out.mkdir(parents=True, exist_ok=True)
    for row in manifest['kernels'].values():
        source = (prefill/row['path']).resolve()
        if not source.is_relative_to(prefill) or hashlib.sha256(source.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('prefill artifact checksum mismatch')
        shutil.copyfile(source, out/row['path'])
    def add(kind, p, module, factory, *args):
        key = identity(kind, p); entry = out/(key+'.py'); artifact = entry.with_suffix('.tbin')
        source = export_source(module, factory, *args, dependencies=('tensor.compiler.entry',))
        if needs_build(entry, artifact, source, manifest.get('target','sm_89')):
            entry.write_text(source); artifact.unlink(missing_ok=True)
            build_artifact(entry, artifact, target=manifest.get('target','sm_89'))
        manifest['kernels'][key] = dict(kind=kind, parameters=p, path=artifact.name,
                                      sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        print('verification', kind, p, flush=True)
    for kind in ('gdn_scan','gdn_conv'):
        p = dict(slots=slots, chunk=chunk)
        add('spec_'+kind, p, 'tensor_llm.qwen35.kernels.speculative', 'make_kernel', kind, p)
    for shape in ((32,128,128),(8192,3)):
        p = dict(slots=slots, chunk=chunk, shape=list(shape))
        if chunk>8:
            add('restore',p,'tensor_llm.qwen35.kernels.hopper_restore','restore_kernel',p)
        else:add('restore', p, 'tensor_llm.qwen35.kernels.speculative', 'make_kernel', 'restore', p)
    p = dict(r=slots*chunk, k=2048, o=248320)
    add('verify_head', p, 'tensor_llm.qwen35.kernels.speculative', 'head_kernel',
        dict(p, block_m=16 if slots*chunk<=16 else (64 if slots*chunk>=64 else 32), columns=128, depth=128, threads=128))
    p = dict(r=slots*chunk, vocab=248320)
    add('argmax', p, 'tensor_llm.qwen35.kernels.decode', 'make_kernel', 'argmax', p)
    manifest.update(verification='greedy-prefix-snapshots', implementation=implementation_hashes(),
                    speculative_verification=True, model_throughput_qualified=False,
                    prefill_lineage=dict(path=str(prefill),
                        manifest_sha256=hashlib.sha256((prefill/'prefill.json').read_bytes()).hexdigest()))
    (out/'prefill.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefill', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); produce(a.prefill, a.out)
