"""Qualify shared request state and compare singleton latency with a frozen runner.

This measures serialized execution, not batched throughput. Use --consumer in
an installed environment to block producer/framework imports and replay logits.
"""
import argparse
import gc
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tilelang', 'tvm', 'tvm_ffi', 'torch', 'triton', 'wgpu'}:
            raise ImportError('producer/framework import prohibited: ' + fullname)


if '--consumer' in sys.argv:
    sys.meta_path.insert(0, Guard())

import numpy as np
import tensor
from tensor_llm import LFM2


def frozen_runner(directory):
    spec = importlib.util.spec_from_file_location('tensor_llm_before', directory / '__init__.py',
                                                submodule_search_locations=[str(directory)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.LFM2


def qualify(model, bundle, out, *, before_package=None, before_bundle=None,
            oracle=True, expected=None, repeats=9):
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').unlink(missing_ok=True)
    prefixes = [[256] + [(j + 13 * i) % 256 for j in range(n - 1)]
                for i, n in enumerate((1, 2, 31, 32, 127, 128, 129, 257))]
    context = 768
    report = dict(schema='tensor.lfm2-shared-requests.v1', status='running',
        protocol='eight independent handles; serialized interleaved execution; not batching',
        model_sha256=digest(model), bundle_sha256=digest(bundle / 'inference.json'),
        harness_sha256=digest(__file__), context=context, prefixes=prefixes, steps=[],
        limitations=['buffer accounting excludes driver-owned graph storage',
                     'singleton timing covers this checkpoint and short contexts only'])
    manifest = json.loads((bundle / 'inference.json').read_text())
    report['implementation'] = manifest['implementation']
    report['artifacts'] = {key: row['sha256'] for key, row in manifest['kernels'].items()}
    if expected:
        source = json.loads((expected / 'report.json').read_text())
        if (source['status'] != 'passed' or source['model_sha256'] != report['model_sha256']
                or source['bundle_sha256'] != report['bundle_sha256']):
            raise ValueError('consumer replay requires the same qualified model and bundle')
        report['expected_report_sha256'] = digest(expected / 'report.json')
    with tensor.Device() as device, LFM2(model, bundle, device, context=context,
                                        max_requests=8) as engine:
        report['adapter'] = device.info
        singleton_bytes = engine.allocated_bytes
        requests = [engine] + [engine.new_request() for _ in range(7)]
        report['memory'] = dict(weight_bytes=engine.weight_bytes, shared_bytes=engine.shared_bytes,
            private_bytes=[r.private_bytes for r in requests],
            singleton_buffer_bytes=singleton_bytes, eight_request_buffer_bytes=engine.allocated_bytes)
        assert all(r.weights is engine.weights and r.workspaces is engine.workspaces for r in requests)
        assert engine.allocated_bytes == engine.shared_bytes + sum(r.private_bytes for r in requests)
        # Independent runs use the same math; an eager Torch oracle checks that
        # the unchanged math also implements the declared numeric contract.
        independent = LFM2(model, bundle, device, context=context)
        reference = None
        if oracle:
            from benchmarks.lfm2.torch_reference import Reference
            reference = Reference(model)
        try:
            controls = []
            for index, prefix in enumerate(prefixes):
                independent.reset()
                if reference:reference.reset()
                values = []
                for stage, tokens in enumerate((prefix, [65], [66, 67])):
                    actual = independent.forward(tokens)
                    values.append(actual)
                    row = dict(request=index, stage=stage, finite=bool(np.isfinite(actual).all()))
                    assert row['finite']
                    if reference:
                        for start in range(0, len(tokens), 128):
                            want = reference.forward(tokens[start:start + 128])
                        error = float(np.linalg.norm(actual - want) / np.linalg.norm(want))
                        row['independent_relative_rms'] = error
                        assert error < .01, row
                    report['steps'].append(row)
                controls.append(values)
            for stage, order in enumerate((range(8), reversed(range(8)), (3, 0, 6, 1, 7, 2, 5, 4))):
                for index in order:
                    tokens = (prefixes[index], [65], [66, 67])[stage]
                    actual = requests[index].forward(tokens)
                    np.testing.assert_array_equal(actual, controls[index][stage],
                                                  err_msg=f'request {index}, stage {stage}')
                    name = f'request-{index}-stage-{stage}.npy'
                    if expected:
                        if digest(expected / name) != source['saved_logits'][name]:
                            raise ValueError('consumer reference logits checksum mismatch')
                        np.testing.assert_array_equal(actual, np.load(expected / name, allow_pickle=False))
                    np.save(out / name, actual)
            report['saved_logits'] = {p.name: digest(p) for p in sorted(out.glob('request-*.npy'))}
            # Reset/close/re-admission must not damage a live neighbor.
            other = requests[1]
            position = requests[2].position
            # Match call boundaries: the contract rounds prefill operands to FP16.
            independent.reset();independent.forward(prefixes[2]);independent.forward([65]);independent.forward([66, 67])
            want = independent.forward([68])
            other.reset();other.close()
            assert requests[2].position == position
            np.testing.assert_array_equal(requests[2].forward([68]), want)
            with engine.new_request() as replacement:
                np.testing.assert_array_equal(replacement.forward(prefixes[1]), controls[1][0])
                try:other.forward([65])
                except RuntimeError:pass
                else:raise AssertionError('closed request was accepted')
            report['gates'] = dict(shared_weights=True, shared_scratch=True,
                independent_states=True, interleaved_bitwise_equal=True,
                reset_close_reuse_isolated=True, stale_handle_rejected=True)
        finally:
            independent.close()
            del reference
            gc.collect()
            if oracle:
                import torch
                torch.cuda.empty_cache()
        for request in requests[1:]:request.close()
        assert engine.allocated_bytes == singleton_bytes
        if before_package:
            if not before_bundle:raise ValueError('before-package requires before-bundle')
            before_type = frozen_runner(before_package)
            report['before_sources'] = {p.name: digest(p) for p in sorted(before_package.glob('*.py'))}
            report['before_bundle_sha256'] = digest(before_bundle / 'inference.json')
            samples = []
            with before_type(model, before_bundle, device, context=context) as before:
                assert before.allocated_bytes == singleton_bytes
                for depth in (128, 512):
                    prefix = [256] + [j % 256 for j in range(depth - 1)]
                    before.reset();engine.reset()
                    np.testing.assert_array_equal(before.forward(prefix), engine.forward(prefix))
                    for token in (65, 66, 67):
                        np.testing.assert_array_equal(before.forward([token]), engine.forward([token]))
                    timings = {'before': [], 'shared': []}
                    for repetition in range(-3, repeats):
                        order = (('before', before), ('shared', engine))
                        if repetition % 2:order = tuple(reversed(order))
                        for name, runner in order:
                            runner.reset()
                            start = time.perf_counter();runner.forward(prefix)
                            prefill = time.perf_counter() - start
                            start = time.perf_counter()
                            for token in range(64):runner.forward([token])
                            decode = time.perf_counter() - start
                            if repetition >= 0:
                                timings[name].append(dict(prefill_seconds=prefill, decode_seconds=decode))
                    row = dict(depth=depth, generated=64, repetitions=repeats, warmups=3,
                               samples=timings)
                    for phase in ('prefill', 'decode'):
                        old = statistics.median(s[phase + '_seconds'] for s in timings['before'])
                        new = statistics.median(s[phase + '_seconds'] for s in timings['shared'])
                        row[phase + '_ratio_shared_over_before'] = new / old
                    samples.append(row)
            report['singleton_comparison'] = samples
            report['singleton_timing_protocol'] = 'alternating paired order, host logits, 3 warmups/9 measured repetitions; excludes load, reset, tokenization and sampling'
        report['status'] = 'passed'
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'memory', 'gates')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'bundle', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('before-package', 'before-bundle', 'expected'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--consumer', action='store_true')
    args = parser.parse_args()
    qualify(args.model, args.bundle, args.out, before_package=args.before_package,
            before_bundle=args.before_bundle, oracle=not args.consumer, expected=args.expected)
