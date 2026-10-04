"""Validate and compare the CUDA transfer of the Vulkan inference optimizations."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import tensor
from tensor_llm import GGUF, LFM2, Tokenizer
from tensor_llm.provenance import implementation_hashes


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def metrics(actual, expected):
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    delta = actual - expected
    return dict(relative_rms=float(np.linalg.norm(delta) / np.linalg.norm(expected)),
                cosine=float(np.dot(actual, expected) / (np.linalg.norm(actual) * np.linalg.norm(expected))),
                max_error=float(np.max(np.abs(delta))),
                top1_equal=bool(np.argmax(actual) == np.argmax(expected)),
                finite=bool(np.isfinite(actual).all()))


def tokens(model, depths, generated):
    tokenizer = Tokenizer(GGUF(model).metadata)
    pattern = tokenizer.encode('Tensor compares packed language model inference using the same fixed tokens. ', add_bos=False)
    return [dict(depth=depth, prompt=[tokenizer.bos] + [pattern[i % len(pattern)] for i in range(depth - 1)],
                 decode=[pattern[i % len(pattern)] for i in range(generated)]) for depth in depths]


def validate(model, bundle, out, *, depths=(1, 32, 127, 128, 129, 384, 512, 2048, 4095, 4096, 8192), runner_type=LFM2):
    """Independent mixed-precision Torch operators, reset, and cached continuation."""
    from benchmarks.lfm2.torch_reference import Reference
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cases = tokens(model, depths, 1)
    observations = []
    eager = Reference(model)
    with tensor.Device() as device, runner_type(model, bundle, device) as runner:
        first = None
        for case in cases:
            runner.reset(); eager.reset()
            for stage, ids in (('prefill', case['prompt']), ('decode', case['decode'])):
                actual = runner.forward(ids)
                for start in range(0, len(ids), 128): expected = eager.forward(ids[start:start + 128])
                check = metrics(actual, expected)
                row = dict(depth=case['depth'], stage=stage, position=runner.position, **check)
                assert check['finite'] and check['relative_rms'] < .01 and check['cosine'] > .9999 and check['top1_equal'], row
                observations.append(row)
                np.save(out / f"{case['depth']}-{stage}.npy", actual)
                print('validate', row, flush=True)
                if first is None: first = actual.copy()
        runner.reset(); np.testing.assert_array_equal(runner.forward(cases[0]['prompt']), first)
        examples=[cases[min(3,len(cases)-1)]['prompt']]
        if getattr(runner,'long_plan',None) and cases[-1]['depth']>=4096:examples.append(cases[-1]['prompt'])
        for example in examples:
            runner.reset(); graph = runner.forward(example)
            graph_decode = runner.forward(cases[0]['decode'])
            runner.reset(); runner.graphs_enabled = False
            eager_plan = runner.forward(example)
            np.testing.assert_array_equal(graph, eager_plan)
            np.testing.assert_array_equal(graph_decode, runner.forward(cases[0]['decode']))
            runner.graphs_enabled = True
        gpu = runner.generate('What is the capital of France?', max_tokens=16)
        host = runner.generate('What is the capital of France?', max_tokens=16, gpu_greedy=False)
        assert gpu == host
        long_generation=None
        if getattr(runner,'long_plan',None) and any(c['depth']>=4096 for c in cases):
            case=next(c for c in cases if c['depth']>=4096)
            prompt=runner.tokenizer.decode(case['prompt'][1:])
            # Adjacent pieces can merge when text is re-tokenized. Length, rather
            # than the synthetic token pattern, controls this graph-selection check.
            while len(runner.tokenizer.encode(prompt)) < 4096:
                prompt += ' Tensor compares packed language model inference using the same fixed tokens. '
            prompt_tokens=len(runner.tokenizer.encode(prompt))
            assert prompt_tokens + 4 <= runner.context
            long_gpu=runner.generate(prompt,max_tokens=4,chat=False)
            long_host=runner.generate(prompt,max_tokens=4,chat=False,gpu_greedy=False)
            assert long_gpu==long_host
            long_generation=dict(prompt=prompt,prompt_tokens=prompt_tokens,max_tokens=4,chat=False,result=long_gpu)
        report = dict(schema='tensor.lfm2-cuda-transfer-validation.v1', status='passed',
                      model_sha256=digest(model), bundle_sha256=digest(Path(bundle) / 'inference.json'),
                      implementation=runner.manifest['implementation'], adapter=device.info, steps=observations,
                      cases=cases,
                      reference_source_sha256=digest(Path(__file__).with_name('torch_reference.py')),
                      saved_logits={p.name:digest(p) for p in sorted(out.glob('*.npy'))},
                      long_generation=long_generation,
                      gates=dict(relative_rms_max=.01, cosine_min=.9999, matching_argmax=True,
                                 reset_bitwise_equal=True, graph_eager_bitwise_equal=True,
                                 gpu_host_greedy_equal=True), generation=gpu,
                      reference='independent Torch FP16 prefill/FP32 decode operators; no shared kernel templates')
    del eager; gc.collect()
    (out / 'validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def compare(model, bundles, reference, out, *, depths=(128, 512, 2048, 8192), generated=64, repeats=5, runner_types=None):
    if repeats < 1 or generated < 1 or any(d < 1 for d in depths): raise ValueError('positive benchmark sizes required')
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    cases = tokens(model, depths, generated)
    spec = dict(model=str(Path(model).resolve()), out=str((out / 'native').resolve()),
                context=max(depths) + generated,
                benchmarks=[dict(name=f"pp{c['depth']}-tg{generated}", prompt=c['prompt'], decode=c['decode'], repeats=repeats, warmups=3) for c in cases])
    specpath = out / 'reference-spec.json'; specpath.write_text(json.dumps(spec, indent=2) + '\n')
    with (out / 'native.log').open('w') as log:
        subprocess.run([str(Path(reference).resolve()), str(specpath)], check=True, stdout=log, stderr=log)
    native = json.loads((out / 'native/reference.json').read_text())
    if len(native['benchmarks']) != len(cases): raise ValueError('incomplete native benchmark')
    runners = {}; observations = []; profiles = {}
    with tensor.Device() as device:
        try:
            for name, path in bundles.items():
                if runner_types and name in runner_types:cls=runner_types[name]
                elif name == 'previous':
                    from benchmarks.lfm2.decode_optimization import OptimizedLFM2
                    cls = OptimizedLFM2
                else: cls = LFM2
                runner = cls(model, path, device, context=max(depths) + generated)
                runners[name] = runner
                profiles[name] = dict(bundle_sha256=digest(Path(path) / 'inference.json'),
                                      cuda_profile=runner.cuda_profile, rows=runner.rows,
                                      allocated_bytes=runner.allocated_bytes, launches_per_decode=len(runner.plans[1]),
                                      launches_per_prefill={r: len(p) for r, p in runner.plans.items() if r > 1},
                                      manifest=runner.manifest)
            names = list(runners)
            for case, baseline in zip(cases, native['benchmarks']):
                samples = {name: [] for name in names}
                for repeat in range(-3, repeats):
                    for name in names[repeat % len(names):] + names[:repeat % len(names)]:
                        runner = runners[name]; runner.reset()
                        start = time.perf_counter(); runner.forward(case['prompt']); prefill = time.perf_counter() - start
                        times = []; start = time.perf_counter()
                        for token in case['decode']:
                            begin = time.perf_counter(); runner.forward([token]); times.append(time.perf_counter() - begin)
                        seconds = time.perf_counter() - start
                        if repeat >= 0: samples[name].append(dict(prefill_seconds=prefill, decode_seconds=seconds, decode_latencies=times))
                row = dict(prompt_tokens=case['depth'], decode_tokens=generated, runners={})
                for name, raw in [*samples.items(), ('llama.cpp', baseline['samples'])]:
                    prefill = float(np.median([s['prefill_seconds'] for s in raw]))
                    decode = float(np.median([s['decode_seconds'] for s in raw]))
                    row['runners'][name] = dict(prefill_tokens_per_second=case['depth'] / prefill,
                                                decode_tokens_per_second=generated / decode,
                                                prefill_seconds=prefill, decode_seconds=decode,
                                                decode_milliseconds=decode / generated * 1000, samples=raw)
                observations.append(row)
                print('compare', {**row, 'runners': {name: {k: v for k, v in values.items() if k != 'samples'} for name, values in row['runners'].items()}}, flush=True)
            checkout = Path(reference).resolve().parent / 'llama-cpp'
            commit = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip() if checkout.exists() else None
            report = dict(schema='tensor.lfm2-cuda-transfer-benchmark.v1', status='passed', adapter=device.info,
                          platform=platform.platform(), model_file=Path(model).name, model_sha256=digest(model),
                          implementation=implementation_hashes(), profiles=profiles, cases=observations,
                          harness_sha256={name:digest(Path(__file__).with_name(name)) for name in
                                          ('cuda_transfer.py','reference.cpp','torch_reference.py','fp16_decode.py','decode_optimization.py')},
                          native=dict(commit=commit, helper_sha256=digest(reference), configuration={k: v for k, v in native.items() if k not in ('benchmarks', 'validation', 'tokenization')}),
                          protocol='one sequence, fixed shared tokens, completed host FP32 logits; 3 warmups per runner; rotated sequential Tensor runners; native measured separately on the same GPU before Tensor; loading/reset/tokenization/sampling excluded',
                          repeats=repeats, gpu_concurrency=1)
        finally:
            for runner in runners.values(): runner.close()
    (out / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('validate', 'compare'))
    p.add_argument('--model', required=True, type=Path); p.add_argument('--out', required=True, type=Path)
    p.add_argument('--bundle', type=Path); p.add_argument('--base', type=Path); p.add_argument('--previous', type=Path)
    p.add_argument('--reference', type=Path); p.add_argument('--depths', type=int, nargs='+')
    p.add_argument('--generated', type=int, default=64); p.add_argument('--repeats', type=int, default=5)
    args = p.parse_args()
    if args.action == 'validate':
        if not args.bundle: p.error('--bundle required for validation')
        validate(args.model, args.bundle, args.out, **({'depths': tuple(args.depths)} if args.depths else {}))
    else:
        if not args.bundle or not args.base or not args.reference: p.error('--bundle, --base and --reference required')
        bundles = dict(default=args.base, optimized=args.bundle)
        if args.previous: bundles['previous'] = args.previous
        compare(args.model, bundles, args.reference, args.out, generated=args.generated, repeats=args.repeats,
                **({'depths': tuple(args.depths)} if args.depths else {}))


if __name__ == '__main__': main()
