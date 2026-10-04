"""Audit an installed four-distribution CUDA LFM2 consumer against saved logits."""
import argparse
import hashlib
import importlib.abc
import importlib.metadata as metadata
import json
import sys
from pathlib import Path

FORBIDDEN = {'tilelang', 'tvm', 'tvm_ffi', 'torch', 'triton', 'wgpu'}


class BlockCompilerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN or fullname == 'tensor.compiler' or fullname.startswith('tensor.compiler.'):
            raise ImportError('compiler/framework import in standalone consumer: ' + fullname)


def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def consume(model, bundle, reference, out):
    installed = {d.metadata['Name'].lower().replace('_', '-') for d in metadata.distributions()}
    assert installed == {'tensor-workspace', 'tensor-llm', 'numpy', 'regex'}, installed
    sys.meta_path.insert(0, BlockCompilerImports())
    import numpy as np
    import tensor
    from tensor_llm import LFM2
    from tensor_llm.provenance import implementation_hashes
    fixture = json.loads((reference / 'validation.json').read_text())
    assert fixture['status'] == 'passed' and digest(model) == fixture['model_sha256']
    assert digest(bundle / 'inference.json') == fixture['bundle_sha256']
    cases = fixture['cases']
    observations = []
    with tensor.Device() as device, LFM2(model, bundle, device) as runner:
        for case in cases:
            runner.reset()
            for stage, values in (('prefill', case['prompt']), ('decode', case['decode'])):
                actual = runner.forward(values)
                expected = np.load(reference / f"{case['depth']}-{stage}.npy")
                np.testing.assert_array_equal(actual, expected)
                observations.append(dict(depth=case['depth'], stage=stage, bitwise_equal=True))
        generated = runner.generate('What is the capital of France?', max_tokens=16)
        assert generated == fixture['generation']
        long=fixture.get('long_generation')
        if long:
            assert runner.generate(long['prompt'],max_tokens=long['max_tokens'],chat=long['chat'])==long['result']
        report = dict(schema='tensor.lfm2-cuda-transfer-consumer.v1', status='passed',
                      model_sha256=fixture['model_sha256'], bundle_sha256=fixture['bundle_sha256'],
                      implementation=implementation_hashes(), adapter=device.info,
                      distributions=sorted(installed), steps=observations, generation=generated,
                      long_generation_equal=bool(long),
                      forbidden_imports=sorted(FORBIDDEN & {name.split('.')[0] for name in sys.modules}),
                      compiler_imports=sorted(name for name in sys.modules if name == 'tensor.compiler' or name.startswith('tensor.compiler.')),
                      consumer_source_sha256=digest(__file__))
    assert not report['forbidden_imports']
    assert not report['compiler_imports']
    out.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'bundle', 'reference', 'out'): p.add_argument('--' + name, required=True, type=Path)
    args = p.parse_args(); consume(args.model, args.bundle, args.reference, args.out)
