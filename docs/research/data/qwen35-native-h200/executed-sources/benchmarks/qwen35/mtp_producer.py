"""Produce an experimental MTP draft bundle from selected native target kernels."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil

from .build import build_artifact
from tensor.compiler.entry import export_source
from tensor_llm.qwen35.checkpoint import Qwen35Checkpoint
from tensor_llm.qwen35.artifacts import identity
from tensor_llm.qwen35.mtp.decode import mtp_implementation_hashes


def produce(checkpoint, decoder, out):
    checkpoint = Qwen35Checkpoint(checkpoint, branch='mtp')
    decoder, out = Path(decoder).resolve(), Path(out).resolve()
    if out == decoder or out.is_relative_to(decoder):
        raise ValueError('MTP output must be separate from the target bundle')
    manifest = json.loads((decoder/'inference.json').read_text())
    if (manifest['schema'] != 'tensor.qwen35-batch.v1'
            or manifest['config'] != json.loads(json.dumps(asdict(checkpoint.config)))):
        raise ValueError('target bundle does not match the checkpoint')
    out.mkdir(parents=True, exist_ok=True)
    (out/'inference.json').unlink(missing_ok=True)
    lineage = dict(path=str(decoder), manifest_sha256=hashlib.sha256((decoder/'inference.json').read_bytes()).hexdigest(),
                   implementation=manifest['implementation'])
    for row in manifest['kernels'].values():
        for part in (row, *(row[n] for n in ('logical', 'quantize', 'merge') if n in row)):
            source = (decoder/part['path']).resolve()
            if (not source.is_relative_to(decoder)
                    or hashlib.sha256(source.read_bytes()).hexdigest() != part['sha256']):
                raise ValueError('target artifact checksum mismatch')
            dest = out/part['path']; dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
    r = manifest['slots']
    for kind, p, module, factory, args in (
            ('mtp_join', dict(r=r, c=2048, eps=checkpoint.config.epsilon),
             'tensor_llm.qwen35.kernels.mtp', 'make_kernel', None),
            ('mtp_cast', dict(r=r, c=2048), 'tensor_llm.qwen35.kernels.mtp', 'make_kernel', None),
            ('bf16_linear', dict(r=r, k=4096, o=2048),
             'tensor_llm.qwen35.kernels.matmul', 'bf16_head_kernel', None)):
        key = identity(kind, p)
        args = (kind, p) if factory == 'make_kernel' else (dict(p, columns=128, depth=128),)
        entry = out/(key+'.py'); artifact = entry.with_suffix('.tbin')
        entry.write_text(export_source(module, factory, *args, dependencies=('tensor.compiler.entry',)))
        artifact.unlink(missing_ok=True)
        build_artifact(entry, artifact, target=manifest['target'])
        manifest['kernels'][key] = dict(kind=kind, parameters=p, path=artifact.name,
                                      sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        print('MTP adapter', kind, flush=True)
    manifest.update(draft='qwen35-mtp', target_config=asdict(checkpoint.config),
        config=asdict(replace(checkpoint.config, layers=('full_attention',))),
        implementation=mtp_implementation_hashes(), target_decoder_lineage=lineage,
        mtp_weight_bytes=checkpoint.weight_bytes, speculative_verification=False,
        model_throughput_qualified=False, full_stress_target_reached=False)
    (out/'inference.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--decoder', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    print(produce(a.checkpoint, a.decoder, a.out))
