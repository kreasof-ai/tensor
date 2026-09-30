"""Build Tensor attention artifacts, then benchmark them in a fresh consumer.

This demonstrates the existing .tbin runtime, not a torch.compile backend.
The consumer forbids TileLang/TVM imports and forces PyTorch's FlashAttention
SDPA backend; failures cannot silently fall back to a math implementation.
"""
from __future__ import annotations

from pathlib import Path as _RepositoryPath
import sys as _repository_sys
_repository_sys.path.insert(0, str(_RepositoryPath(__file__).resolve().parents[2]))


import argparse
import hashlib
import importlib.abc
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SHAPES = ((1, 8, 128, 64), (1, 8, 129, 64), (1, 8, 512, 64),
          (1, 8, 1024, 64), (2, 4, 257, 64), (1, 8, 512, 128))


def specialize(shape, causal):
    text = (ROOT / 'examples/flash_attention.py').read_text()
    values = dict(zip(('BATCH', 'HEADS', 'SEQ_LEN', 'HEAD_DIM'), shape))
    values['IS_CAUSAL'] = causal
    for name, value in values.items():
        text, count = re.subn(rf'^{name} = .*$', f'{name} = {value!r}', text, flags=re.M)
        if count != 1:
            raise ValueError(f'attention example needs exactly one {name} declaration')
    return text


def produce(out, *, quick=False):
    import tensor as tx
    out.mkdir(parents=True, exist_ok=False)
    (out / 'sources').mkdir()
    (out / 'artifacts').mkdir()
    with tx.Device() as device:
        info = device.info
    rows = []
    for shape in SHAPES[:2] if quick else SHAPES:
        for causal in (False, True):
            label = 'b%d-h%d-s%d-d%d-' % shape + ('causal' if causal else 'noncausal')
            source = out / 'sources' / (label + '.py')
            artifact = out / 'artifacts' / (label + '.tbin')
            source.write_text(specialize(shape, causal), encoding='utf-8')
            built = tx.build(source, artifact, target=info['arch'], compiler='nvrtc', cache_dir=out / 'compiler-cache')
            row = {'name': label, 'shape': list(shape), 'causal': causal,
                   'artifact': artifact.relative_to(out).as_posix(),
                   'artifact_sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(),
                   'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                   'build_seconds': built['seconds'], 'nvrtc_seconds': built['compile_seconds'],
                   'compiler': built['compiler'], 'cache_hit': built['cache_hit']}
            rows.append(row)
            print(json.dumps({'built': label, 'seconds': row['build_seconds']}), flush=True)
    producer = {'device': info, 'rows': rows, 'torch_compile_backend': False,
                'source_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True))}
    (out / 'producer.json').write_text(json.dumps(producer, indent=2) + '\n', encoding='utf-8')


class NoFrontendImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tilelang', 'tvm', 'tvm_ffi'}:
            raise AssertionError('attention consumer attempted compiler import: ' + fullname)


def consume(out, *, iterations=100, repetitions=7):
    sys.meta_path.insert(0, NoFrontendImports())
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    import tensor as tx
    from tensor.artifacts.format import read_artifact
    torch.manual_seed(41)
    torch.backends.cuda.matmul.allow_tf32 = False
    producer = json.loads((out / 'producer.json').read_text())
    stream = torch.cuda.Stream()
    results = []

    def graph_timing(call):
        for _ in range(20):
            call()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(iterations):
                call()
        samples = []
        for _ in range(repetitions):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record(stream)
            graph.replay()
            end.record(stream)
            end.synchronize()
            samples.append(begin.elapsed_time(end) * 1000 / iterations)
        return {'median_us': statistics.median(samples), 'samples_us': samples}

    def submission_timing(call):
        enqueue, completion = [], []
        for _ in range(repetitions):
            stream.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                call()
            queued = time.perf_counter()
            stream.synchronize()
            completed = time.perf_counter()
            enqueue.append((queued - start) * 1e6 / iterations)
            completion.append((completed - start) * 1e6 / iterations)
        return {'enqueue_median_us': statistics.median(enqueue),
                'enqueue_samples_us': enqueue, 'batch_completion_median_us': statistics.median(completion),
                'batch_completion_samples_us': completion}

    with torch.inference_mode(), torch.cuda.stream(stream), tx.Device(stream=stream.cuda_stream) as device:
        for index, case in enumerate(producer['rows']):
            artifact = out / case['artifact']
            assert hashlib.sha256(artifact.read_bytes()).hexdigest() == case['artifact_sha256']
            manifest, _ = read_artifact(artifact)
            assert manifest['compiler']['name'] == 'nvrtc'
            assert manifest['workspace'] == {'bytes': 0, 'alignment': 1}
            assert manifest['target'] == device.info['arch']
            with device.load(artifact) as kernel:
                shape, causal = case['shape'], case['causal']
                q, k, v = (torch.randn(shape, device='cuda', dtype=torch.float16) for _ in range(3))
                output = torch.full_like(q, float('nan'))
                originals = q.clone(), k.clone()
                borrowed = tuple(device.from_dlpack(t) for t in (q, k, v, output))
                assert [t.pointer for t in borrowed] == [t.data_ptr() for t in (q, k, v, output)]
                candidate = lambda: kernel.launch(*borrowed)
                baseline = lambda: F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=causal)
                correctness = []
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    for family in ('random', 'large_logits', 'zero_logits'):
                        if family == 'large_logits':
                            q.copy_(originals[0] * 3); k.copy_(originals[1] * 3)
                        elif family == 'zero_logits':
                            q.zero_(); k.zero_()
                        else:
                            q.copy_(originals[0]); k.copy_(originals[1])
                        scores = q.float() @ k.float().transpose(-1, -2) / shape[-1] ** .5
                        if causal:
                            mask = torch.ones((shape[-2], shape[-2]), dtype=torch.bool, device='cuda').tril()
                            scores.masked_fill_(~mask, float('-inf'))
                        expected = (scores.softmax(-1) @ v.float()).half()
                        output.fill_(float('nan'))
                        candidate()
                        reference = baseline()
                        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-3)
                        torch.testing.assert_close(reference, expected, rtol=2e-2, atol=2e-3)
                        correctness.append({'family': family,
                            'tensor_max_abs_error': (output - expected).abs().max().item(),
                            'pytorch_max_abs_error': (reference - expected).abs().max().item()})
                    q.copy_(originals[0]); k.copy_(originals[1])
                    gpu_candidate = graph_timing(candidate)
                    gpu_baseline = graph_timing(baseline)
                    host_candidate = submission_timing(candidate)
                    host_baseline = submission_timing(baseline)
                if index == 0:
                    # A clean Tensor/NumPy process can replay this .tbin via the CLI.
                    inputs = out / 'runtime-inputs'; inputs.mkdir(exist_ok=True)
                    import numpy as np
                    for name, tensor in (('q', q), ('k', k), ('v', v)):
                        np.save(inputs / (name + '.npy'), tensor.cpu().numpy())
                    candidate()
                    np.save(inputs / 'expected.npy', output.cpu().numpy())
                row = {**case, 'correctness': correctness, 'tensor_gpu': gpu_candidate,
                       'pytorch_flash_gpu': gpu_baseline, 'tensor_prepared_submission': host_candidate,
                       'pytorch_submission': host_baseline,
                       'gpu_speedup': gpu_baseline['median_us'] / gpu_candidate['median_us'],
                       'submission_speedup': host_baseline['batch_completion_median_us'] / host_candidate['batch_completion_median_us'],
                       'shared_memory_bytes': manifest['launch']['shared_memory_bytes'],
                       'workspace': manifest['workspace']}
                results.append(row)
                print(json.dumps({'case': case['name'], 'correct': True,
                                  'tensor_gpu_us': gpu_candidate['median_us'],
                                  'pytorch_flash_gpu_us': gpu_baseline['median_us'],
                                  'gpu_speedup': row['gpu_speedup'],
                                  'tensor_submission_us': host_candidate['batch_completion_median_us'],
                                  'pytorch_submission_us': host_baseline['batch_completion_median_us']}), flush=True)
                for buffer in borrowed:
                    buffer.release()
    assert not {'tilelang', 'tvm', 'tvm_ffi'} & sys.modules.keys()
    report = {'status': 'passed', 'device': producer['device'], 'torch_version': torch.__version__,
              'producer': producer, 'rows': results, 'iterations': iterations, 'repetitions': repetitions,
              'warmup': 20, 'rtol': 2e-2, 'atol': 2e-3, 'compiler_imports': False,
              'torch_compile_backend': False, 'baseline': 'PyTorch SDPA forced FLASH_ATTENTION',
              'gpu_timing': 'CUDA events around CUDA graph replay; median per operation',
              'submission_timing': 'wall-clock batch enqueue and completion; warm pre-borrowed Tensor buffers',
              'allocation_policy': 'Tensor reuses a preallocated output; PyTorch SDPA returns allocated outputs (graph allocator during capture)',
              'scope': 'specialized FP16 forward self-attention; contiguous BHSD; causal/noncausal; no mask/dropout/GQA/backward'}
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--quick', action='store_true', help='only sequence lengths 128 and 129, head dimension 64')
    parser.add_argument('--run-only', action='store_true', help='execute existing artifacts; compiler imports prohibited')
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--repetitions', type=int, default=7)
    args = parser.parse_args()
    if args.iters < 1 or args.repetitions < 3:
        parser.error('iterations must be positive and repetitions at least three')
    if args.run_only:
        consume(args.out_dir, iterations=args.iters, repetitions=args.repetitions)
    else:
        produce(args.out_dir, quick=args.quick)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--run-only', '--out-dir', str(args.out_dir),
                        '--iters', str(args.iters), '--repetitions', str(args.repetitions)], check=True)
