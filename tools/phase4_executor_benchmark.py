"""Compare Python and C++ execution of the same cached kernel in one process."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
from tensor_torch import bridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True,
                        help='cached phase4 pointwise-129 artifact, with two FP32 inputs')
    parser.add_argument('--out', type=Path, required=True)
    opts = parser.parse_args()
    if bridge._executor is None:
        raise RuntimeError('install the matching C++ executor wheel')
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        a, b = (torch.ones(129, device='cuda') for _ in range(2))
        kernel = bridge.Kernel(opts.artifact)
        reference = lambda a, b: torch.relu(a*2+b)
        native = bridge._executor
        try:
            bridge._executor = None
            python_plan = bridge.LaunchPlan(kernel, (a,b), reference)
            python_fixed = kernel.prepare(a,b)
        finally:
            bridge._executor = native
        cpp_plan = bridge.LaunchPlan(kernel, (a,b), reference)
        cpp_fixed = kernel.prepare(a,b)
        plans = {'python_allocating': lambda: python_plan(a,b),
                 'cpp_allocating': lambda: cpp_plan(a,b),
                 'python_prepared': python_fixed, 'cpp_prepared': cpp_fixed}
        for name, call in plans.items():
            for _ in range(100):
                output = call()
            torch.testing.assert_close(output if output is not None else call.outputs[0], reference(a,b))
        samples = {name: [] for name in plans}
        names = list(plans)
        for repetition in range(12):
            # Rotate order to distribute clock drift and allocator effects.
            for name in names[repetition % 4:] + names[:repetition % 4]:
                stream.synchronize()
                start = time.perf_counter()
                for _ in range(1000):
                    plans[name]()
                samples[name].append((time.perf_counter()-start)*1e3)
                stream.synchronize()
        medians = {name: statistics.median(values) for name, values in samples.items()}
    result = {'torch': torch.__version__, 'device': torch.cuda.get_device_name(),
              'artifact_sha256': hashlib.sha256(opts.artifact.read_bytes()).hexdigest(),
              'timing': 'median 12 batches of 1000 host calls; allocation included in allocating calls; completion excluded; rotated provider order; no Dynamo wrapper',
              'median_us': medians, 'samples_us': samples,
              'allocating_speedup': medians['python_allocating']/medians['cpp_allocating'],
              'prepared_speedup': medians['python_prepared']/medians['cpp_prepared']}
    opts.out.parent.mkdir(parents=True, exist_ok=True)
    opts.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
