"""Same-GPU Tensor WebGPU versus CLBlast OpenCL GEMM, with explicit precision checks.

Build artifacts in a producer; run measurements in a compiler-free wheel consumer.
Tuning is a bounded search over complete stock database configurations, timed and
validated on the actual requested shapes. It is not an exhaustive CLBlast tuner run.
"""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from contextlib import closing
import ctypes as c
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distributions, version
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

import numpy as np
from scripts.validation.webgpu_validation import compiler_guard, source_hashes, specialize

ROOT = Path(__file__).resolve().parents[2]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def cases():
    shapes = [(s, s, s) for s in (512, 1024, 2048, 4096)]
    shapes += [(m, n, k) for m in (1, 32) for n, k in ((2560, 1024), (1024, 2560))]
    return [dict(name=f'{dtype}-{m}-{n}-{k}-{mode}', m=m, n=n, k=k, dtype=dtype,
                 mode=mode) for dtype in ('float32', 'float16')
            for m, n, k in shapes for mode in ('gemm', 'linear')]


def build(directory):
    import tensor
    from tensor.artifacts.format import read_artifact
    directory.mkdir(parents=True, exist_ok=False)
    suite = {'schema': 'tensor.clblast-comparison-suite.v1', 'cases': [],
             'consumer_source_sha256': source_hashes(ROOT / 'src/tensor'),
             'compiler_source_sha256': {name: sha((ROOT / 'src/tensor/compiler' / name).read_bytes())
                                        for name in ('webgpu.py', 'webgpu_lowering.py')},
             'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    for row in cases():
        constants = dict(M=row['m'], N=row['n'], K=row['k'], DTYPE=row['dtype'],
                         OUTPUT_DTYPE=row['dtype'], TRANSPOSE_B=True,
                         USE_BIAS=row['mode'] == 'linear', RELU=row['mode'] == 'linear')
        source = directory / (row['name'] + '.py')
        source.write_text(specialize((ROOT / 'examples/webgpu_gemm.py').read_text(), constants))
        artifact = source.with_suffix('.tbin')
        tensor.build(source, artifact, provider='webgpu', cache_dir=directory / 'cache')
        manifest, files = read_artifact(artifact)
        suite['cases'].append({**row, 'artifact': artifact.name,
                               'artifact_sha256': sha(artifact.read_bytes()),
                               'wgsl_sha256': sha(files['kernel.wgsl']),
                               'launch': manifest['launch'],
                               'workgroup_storage_bytes': manifest['webgpu']['workgroup_storage_bytes']})
        print('built', row['name'], flush=True)
    (directory / 'suite.json').write_text(json.dumps(suite, indent=2) + '\n', encoding='utf-8')


def inputs(row, index):
    # Pair gemm/linear and both precisions with the same underlying random values.
    rng = np.random.default_rng(np.random.SeedSequence([42, (index % 16) // 2]))
    a = rng.standard_normal((row['m'], row['k']), dtype=np.float32).astype(row['dtype'])
    b = rng.standard_normal((row['n'], row['k']), dtype=np.float32).astype(row['dtype'])
    bias = rng.standard_normal(row['n'], dtype=np.float32).astype(row['dtype'])
    reference = a.astype(np.float32) @ b.astype(np.float32).T
    if row['mode'] == 'linear':
        reference += bias.astype(np.float32)
        np.maximum(reference, 0, out=reference)
    return (a, b, bias), reference.astype(row['dtype'])


def check(actual, reference):
    delta = actual.astype(np.float64) - reference.astype(np.float64)
    matched = bool(np.isfinite(actual).all() and np.all(
        np.abs(delta) <= .002 + .02 * np.abs(reference.astype(np.float64))))
    return {'passed': matched, 'atol': .002, 'rtol': .02,
            'maximum_absolute_error': float(np.max(np.abs(delta))),
            'relative_rms': float(np.sqrt(np.mean(delta ** 2)) /
                                  max(float(np.sqrt(np.mean(reference.astype(np.float64) ** 2))), 1e-20)),
            'mismatched_elements': int(np.count_nonzero(np.abs(delta) > .002 + .02 * np.abs(reference))),
            'output_sha256': sha(memoryview(actual).cast('B'))}


def summary(observations):
    return {'median_ms': statistics.median(observations), 'samples_ms': observations}


def host_samples(call, sync, release):
    for _ in range(20):
        value = call(); sync(); release(value)
    observations = []
    for _ in range(45):
        start = time.perf_counter()
        value = call(); sync()
        observations.append((time.perf_counter() - start) * 1000)
        release(value)
    return summary(observations)


class OpenCL:
    def __init__(self, library, probe):
        import pyopencl as cl
        self.cl = cl
        devices = [d for p in cl.get_platforms() for d in p.get_devices(device_type=cl.device_type.GPU)]
        self.device = next(d for d in devices if 'gfx1031' in d.name or '6700' in d.name)
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.context, properties=cl.command_queue_properties.PROFILING_ENABLE)
        # The CLBlast DLL imports the system OpenCL.dll, not our build-time ICD copy.
        self.library = c.CDLL(str(library.resolve()))
        self.probe = c.CDLL(str(probe.resolve()))
        self.probe.tensor_clblast_parameters.argtypes = [c.c_void_p, c.c_char_p, c.c_int, c.c_char_p, c.c_size_t]
        self.library.CLBlastOverrideParameters.argtypes = [c.c_void_p, c.c_char_p, c.c_int,
                                                            c.c_size_t, c.POINTER(c.c_char_p), c.POINTER(c.c_size_t)]
        for precision, scalar in (('S', c.c_float), ('H', c.c_uint16)):
            function = getattr(self.library, 'CLBlast' + precision + 'gemm')
            function.argtypes = [c.c_int] * 3 + [c.c_size_t] * 3 + [scalar, c.c_void_p,
                c.c_size_t, c.c_size_t, c.c_void_p, c.c_size_t, c.c_size_t,
                scalar, c.c_void_p, c.c_size_t, c.c_size_t,
                c.POINTER(c.c_void_p), c.POINTER(c.c_void_p)]
            function.restype = c.c_int
        self.programs = {}
        self.info = {'platform': self.device.platform.name, 'platform_version': self.device.platform.version,
                     'name': self.device.name, 'vendor': self.device.vendor,
                     'version': self.device.version, 'driver': self.device.driver_version,
                     'fp16': 'cl_khr_fp16' in self.device.extensions.split()}

    def parameters(self, family, precision=32):
        output = c.create_string_buffer(4096)
        status = self.probe.tensor_clblast_parameters(self.device.int_ptr, family.encode(), precision,
                                                       output, len(output))
        if status: raise RuntimeError(f'CLBlast parameter retrieval: {status}')
        return {k: int(v) for k, v in (line.split('=') for line in output.value.decode().splitlines())}

    def override(self, configurations, precision=32):
        for family, params in configurations.items():
            names = (c.c_char_p * len(params))(*[key.encode() for key in params])
            values = (c.c_size_t * len(params))(*params.values())
            status = self.library.CLBlastOverrideParameters(self.device.int_ptr, family.encode(),
                                                            precision, len(params), names, values)
            if status: raise RuntimeError(f'CLBlast parameter override {family}: {status}')

    def upload(self, values):
        return [self.cl.Buffer(self.context, self.cl.mem_flags.READ_ONLY |
                               self.cl.mem_flags.COPY_HOST_PTR, hostbuf=value) for value in values]

    def output(self, row):
        return self.cl.Buffer(self.context, self.cl.mem_flags.READ_WRITE,
                              size=row['m'] * row['n'] * np.dtype(row['dtype']).itemsize)

    def launch(self, row, buffers, output):
        half = row['dtype'] == 'float16'
        function = self.library.CLBlastHgemm if half else self.library.CLBlastSgemm
        one, zero = (0x3c00, 0) if half else (1.0, 0.0)
        queue = c.c_void_p(self.queue.int_ptr)
        # Row-major A[M,K] times transpose of B[N,K], alpha=1, beta=0.
        status = function(101, 111, 112, row['m'], row['n'], row['k'], one,
                          buffers[0].int_ptr, 0, row['k'], buffers[1].int_ptr, 0, row['k'], zero,
                          output.int_ptr, 0, row['n'], c.byref(queue), None)
        if status: raise RuntimeError(f'CLBlast GEMM status {status}')
        if row['mode'] == 'linear':
            dtype = row['dtype']
            if dtype not in self.programs:
                scalar = 'half' if half else 'float'
                source = ('#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n' if half else '') + f'''
                __kernel void epilogue(__global {scalar}* out, __global const {scalar}* bias,
                                       uint n, uint count) {{
                  uint i=get_global_id(0);
                  if(i<count) out[i]=max(out[i]+bias[i%n], ({scalar})0);
                }}'''
                program = self.cl.Program(self.context, source).build()
                self.programs[dtype] = (program, self.cl.Kernel(program, 'epilogue'))
            kernel = self.programs[dtype][1]
            kernel(self.queue, (row['m'] * row['n'],), None, output, buffers[2],
                   np.uint32(row['n']), np.uint32(row['m'] * row['n']))

    def download(self, row, output):
        actual = np.empty((row['m'], row['n']), dtype=row['dtype'])
        self.cl.enqueue_copy(self.queue, actual, output, is_blocking=True)
        return actual

    def measure(self, row, values, reference):
        uploaded = self.upload(values)
        output = self.output(row)
        try:
            self.launch(row, uploaded, output); self.queue.finish()
            correctness = check(self.download(row, output), reference)
            def allocated():
                value = self.output(row); self.launch(row, uploaded, value); return value
            allocating = host_samples(allocated, self.queue.finish, lambda value: value.release())
            preallocated = host_samples(lambda: self.launch(row, uploaded, output), self.queue.finish, lambda _: None)
            gpu = []
            for index in range(21):
                # AMD may delay an asynchronous marker's profiling timestamp
                # past a subsequent unprofiled kernel. Complete a barrier first
                # so the start timestamp is established before submitting GEMM.
                start = self.cl.enqueue_barrier(self.queue)
                start.wait()
                self.launch(row, uploaded, output)
                stop = self.cl.enqueue_barrier(self.queue)
                stop.wait()
                elapsed = (stop.profile.start - start.profile.end) / 1e6
                if index: gpu.append(elapsed)
            return dict(correctness=correctness, allocating=allocating, preallocated=preallocated,
                        gpu_interval=summary(gpu))
        finally:
            output.release()
            for value in uploaded: value.release()


class TimestampAdapter:
    def __init__(self, adapter): self.adapter = adapter
    def __getattr__(self, name): return getattr(self.adapter, name)
    def request_device_sync(self, **kwargs):
        kwargs['required_features'] += ['timestamp-query', 'timestamp-query-inside-passes']
        return self.adapter.request_device_sync(**kwargs)


def webgpu_measure(device, artifact, row, values, reference, timestamp_period_ns):
    import wgpu
    from wgpu.backends.wgpu_native.extras import write_timestamp
    from tensor.runtime.abi import BoundCall
    uploaded = [device.from_numpy(value) for value in (values if row['mode'] == 'linear' else values[:2])]
    try:
        with device.load(artifact) as kernel:
            ordered, dimensions, outputs = kernel.prepare(*uploaded)
            output = next(iter(outputs.values()))
            try:
                kernel.launch(*ordered, **dimensions); device.synchronize()
                correctness = check(output.to_numpy(), reference)
                if not correctness['passed']: raise AssertionError('Tensor failed reference check')
                allocating = host_samples(lambda: kernel(*uploaded), device.synchronize, lambda value: value.release())
                preallocated = host_samples(lambda: kernel.launch(*ordered, **dimensions), device.synchronize, lambda _: None)
                bound, symbols, launch = kernel._bind(ordered, dimensions, include_outputs=True)
                with closing(device.prepare_plan([(kernel, BoundCall(device, kernel.manifest, bound, symbols,
                                                             launch, validated=True))])) as plan:
                    pipeline, group, grid = plan.nodes[0]
                    query = device._gpu.create_query_set(type='timestamp', count=2)
                    result = device._gpu.create_buffer(size=16, usage=wgpu.BufferUsage.QUERY_RESOLVE | wgpu.BufferUsage.COPY_SRC)
                    gpu = []
                    try:
                        for index in range(21):
                            encoder = device._gpu.create_command_encoder()
                            compute = encoder.begin_compute_pass()
                            write_timestamp(compute, query, 0)
                            compute.set_pipeline(pipeline); compute.set_bind_group(0, group)
                            compute.dispatch_workgroups(*grid)
                            write_timestamp(compute, query, 1); compute.end()
                            encoder.resolve_query_set(query, 0, 2, result, 0)
                            device._gpu.queue.submit([encoder.finish()])
                            ticks = np.frombuffer(device._gpu.queue.read_buffer(result), dtype=np.uint64)
                            elapsed = int(ticks[1] - ticks[0]) * timestamp_period_ns / 1e6
                            if index: gpu.append(elapsed)
                    finally:
                        result.destroy(); query.destroy()
                return dict(correctness=correctness, allocating=allocating, preallocated=preallocated,
                            gpu_interval=summary(gpu))
            finally: output.release()
    finally:
        for value in uploaded: value.release()


@compiler_guard()
def consume(suite_dir, library, probe, out, timestamp_period_ns, tuning=None):
    import tensor
    from tensor.providers.webgpu import Device
    suite_path = suite_dir / 'suite.json'
    suite = json.loads(suite_path.read_text())
    if suite['schema'] != 'tensor.clblast-comparison-suite.v1': raise ValueError('unknown suite schema')
    if suite['consumer_source_sha256'] != source_hashes(Path(tensor.__file__).parent):
        raise ValueError('consumer differs from producer')
    if [{key: row[key] for key in cases()[0]} for row in suite['cases']] != cases():
        raise ValueError('suite workload inventory mismatch')
    if timestamp_period_ns <= 0: raise ValueError('requires Vulkan timestampPeriod in nanoseconds')
    ocl = OpenCL(library, probe)
    families = ('Xgemm', 'XgemmDirect', 'GemmRoutine')
    defaults = {str(precision): {name: ocl.parameters(name, precision) for name in families}
                for precision in (32, 16)}
    selected = json.loads(tuning.read_text()) if tuning else None
    report = dict(schema='tensor.clblast-comparison.v1', status='running',
                  timestamp=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                  python=platform.python_version(), opencl=ocl.info, stock_parameters=defaults,
                  library_sha256=sha(library.read_bytes()), probe_sha256=sha(probe.read_bytes()),
                  suite_sha256=sha(suite_path.read_bytes()), benchmark_source_sha256=sha(Path(__file__).read_bytes()),
                  consumer_source_sha256=suite['consumer_source_sha256'],
                  compiler_import_guard=True, packages=sorted(d.metadata['Name'] for d in distributions()),
                  versions={name: version(name) for name in ('numpy', 'wgpu', 'pyopencl', 'tensor-workspace')},
                  methodology={'warmup_calls': 20, 'completed_call_samples': 45, 'gpu_samples': 20,
                      'input_rng': 'NumPy PCG64 SeedSequence([42, shape_index])',
                      'allocating': 'allocate output + submit + completion; release outside timer',
                      'preallocated': 'submit + completion; output reused',
                      'gpu_webgpu': 'timestamps around one GEMM/linear dispatch inside compute pass',
                      'gpu_opencl': 'completed start barrier to end barrier around complete routine including preprocessing/epilogue; includes device idle during host submission and start-wait return',
                      'excluded': 'uploads/downloads, correctness reference, cold compilation and tuning',
                      'vulkan_timestamp_period_ns': timestamp_period_ns,
                      'precision': 'Tensor uses FP32 accumulators for both; CLBlast HGEMM uses half accumulators',
                      'epilogue': 'Tensor fused; CLBlast extra OpenCL kernel included for linear'},
                  tuning_sha256=sha(tuning.read_bytes()) if tuning else None, cases=[])
    out.parent.mkdir(parents=True, exist_ok=True)
    def save(): out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    device = Device(max_buffer_size=268435456)
    if device.info['adapter']['backend_type'] != 'Vulkan' or device.info['adapter']['adapter_type'] != 'DiscreteGPU':
        raise ValueError('requires physical Vulkan GPU')
    device._adapter = TimestampAdapter(device._adapter)
    with device:
        report['webgpu'] = device.info
        for index, row in enumerate(suite['cases']):
            print('start', row['name'], flush=True)
            artifact = suite_dir / row['artifact']
            if sha(artifact.read_bytes()) != row['artifact_sha256']: raise ValueError('artifact checksum mismatch')
            values, reference = inputs(row, index)
            precision = 16 if row['dtype'] == 'float16' else 32
            config = None
            if selected and precision == 32:
                anchor = 'rectangular' if row['m'] <= 32 else 'small' if row['m'] == 512 else 'large'
                config = selected['selected'][anchor]['configuration']
                ocl.override(config, precision)
            entry = {**row, 'inputs_sha256': [sha(memoryview(v).cast('B')) for v in values],
                     'precision_matched': precision == 32, 'tuning_configuration': config}
            try:
                # Alternate whole-provider order across cases, with no overlapping GPU work.
                order = ('tensor', 'clblast') if index % 2 == 0 else ('clblast', 'tensor')
                for provider in order:
                    if provider == 'tensor':
                        entry[provider] = webgpu_measure(device, artifact, row, values, reference, timestamp_period_ns)
                    else:
                        entry[provider] = ocl.measure(row, values, reference)
                entry['status'] = 'passed' if entry['clblast']['correctness']['passed'] else 'clblast_precision_failed'
                for mode in ('allocating', 'preallocated', 'gpu_interval'):
                    entry['tensor_over_clblast_' + mode] = (entry['tensor'][mode]['median_ms'] /
                                                             entry['clblast'][mode]['median_ms'])
                report['cases'].append(entry); save()
                print(row['name'], entry['status'], 'allocating ms',
                      {p: round(entry[p]['allocating']['median_ms'], 3) for p in ('tensor', 'clblast')}, flush=True)
            except Exception as error:
                report.update(status='failed', failed_case=row['name'], error=f'{type(error).__name__}: {error}')
                save(); raise
    report['status'] = 'completed'
    report['matched_fp32_all_passed'] = all(row['status'] == 'passed' for row in report['cases'] if row['precision_matched'])
    save()


def database_candidates(source, family):
    directory = 'xgemm' if family == 'Xgemm' else 'xgemm_direct'
    text = (source / f'src/database/kernels/{directory}/{directory}_32.hpp').read_text()
    keys = re.search(r'Precision::kSingle, \{([^}]+)\}', text)[1]
    names = re.findall(r'"([^"]+)"', keys)
    amd = text.split('{ // AMD GPUs', 1)[1].split('{ // ', 1)[0]
    candidates = []
    for match in re.finditer(r'Params\{([^}]+)\}', amd):
        values = [int(value.strip()) for value in match[1].split(',')]
        params = dict(zip(names, values))
        if params not in candidates: candidates.append(params)
    return candidates


@compiler_guard()
def tune(source, library, probe, out):
    ocl = OpenCL(library, probe)
    defaults = {name: ocl.parameters(name) for name in ('Xgemm', 'XgemmDirect', 'GemmRoutine')}
    pools = {name: database_candidates(source, name) for name in ('Xgemm', 'XgemmDirect')}
    # Device-selected defaults plus a deterministic bounded spread of stock AMD configurations.
    pools = {name: [defaults[name]] + [pool[i] for i in np.linspace(0, len(pool)-1, min(12, len(pool)), dtype=int)
                                      if pool[i] != defaults[name]] for name, pool in pools.items()}
    report = dict(schema='tensor.clblast-tuning.v1', status='running', opencl=ocl.info,
                  source='bounded stock AMD database configuration search; not exhaustive',
                  library_sha256=sha(library.read_bytes()), selected={}, candidates=[])
    out.parent.mkdir(parents=True, exist_ok=True)
    anchors = {'small': (512, 512, 512), 'large': (4096, 4096, 4096), 'rectangular': (32, 2560, 1024)}
    for anchor, (m, n, k) in anchors.items():
        row = dict(name=anchor, m=m, n=n, k=k, dtype='float32', mode='gemm')
        values, reference = inputs(row, 0)
        uploaded = ocl.upload(values); output = ocl.output(row)
        records = []
        configurations = [('stock', defaults)]
        for family in ('Xgemm', 'XgemmDirect'):
            for params in pools[family]:
                config = {name: dict(value) for name, value in defaults.items()}
                config[family] = params
                config['GemmRoutine'] = {'XGEMM_MIN_INDIRECT_SIZE': 0 if family == 'Xgemm' else 100000}
                configurations.append((family, config))
        try:
            for i, (family, config) in enumerate(configurations):
                record = dict(anchor=anchor, index=i, family=family, configuration=config)
                try:
                    ocl.override(config)
                    ocl.launch(row, uploaded, output); ocl.queue.finish()
                    correctness = check(ocl.download(row, output), reference)
                    if not correctness['passed']: raise AssertionError('candidate failed precision check')
                    for _ in range(3): ocl.launch(row, uploaded, output); ocl.queue.finish()
                    samples = []
                    for _ in range(7):
                        start = time.perf_counter(); ocl.launch(row, uploaded, output); ocl.queue.finish()
                        samples.append((time.perf_counter() - start) * 1000)
                    record.update(status='passed', timing=summary(samples), correctness=correctness)
                except Exception as error:
                    record.update(status='rejected', error=f'{type(error).__name__}: {error}')
                records.append(record); report['candidates'].append(record)
                out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
                print('tune', anchor, i, family, record['status'], record.get('timing', {}).get('median_ms'), flush=True)
            report['selected'][anchor] = min((record for record in records if record['status'] == 'passed'),
                                            key=lambda record: record['timing']['median_ms'])
        finally:
            output.release()
            for value in uploaded: value.release()
    report['status'] = 'completed'
    out.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument('--build', type=Path)
    action.add_argument('--consume', type=Path)
    action.add_argument('--tune', type=Path, help='CLBlast source checkout')
    p.add_argument('--library', type=Path)
    p.add_argument('--probe', type=Path)
    p.add_argument('--out', type=Path)
    p.add_argument('--tuning', type=Path)
    p.add_argument('--timestamp-period-ns', type=float, default=0)
    a = p.parse_args()
    if a.build: build(a.build)
    else:
        if not all((a.library, a.probe, a.out)): p.error('measurement requires --library, --probe and --out')
        if a.tune: tune(a.tune, a.library, a.probe, a.out)
        else: consume(a.consume, a.library, a.probe, a.out, a.timestamp_period_ns, a.tuning)


if __name__ == '__main__': main()
