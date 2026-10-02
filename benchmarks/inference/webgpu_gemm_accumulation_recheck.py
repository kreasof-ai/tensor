"""Alternate old/new GEMM shaders on one device to recheck small-profile regressions."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from benchmarks.inference.clblast_comparison import cases, inputs, webgpu_measure, TimestampAdapter
from scripts.validation.webgpu_validation import compiler_guard, source_hashes


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@compiler_guard()
def run(before, after, names, timestamp_period_ns):
    import tensor
    from tensor.providers.webgpu import Device
    suites = [json.loads((directory/'suite.json').read_text()) for directory in (before, after)]
    for suite in suites:
        if suite['schema'] != 'tensor.clblast-comparison-suite.v1':
            raise ValueError('unexpected suite')
        if suite['consumer_source_sha256'] != source_hashes(Path(tensor.__file__).parent):
            raise ValueError('consumer differs from producer')
        if [{key: row[key] for key in cases()[0]} for row in suite['cases']] != cases():
            raise ValueError('requires the complete 32-profile inventory')
    if timestamp_period_ns <= 0:
        raise ValueError('requires a measured Vulkan timestamp period')
    device = Device(max_buffer_size=268435456)
    device._adapter = TimestampAdapter(device._adapter)
    report = dict(schema='tensor.webgpu-gemm-accumulation-recheck.v1',
                  timestamp=datetime.now(timezone.utc).isoformat(), compiler_import_guard=True,
                  suites=[dict(path=str(path), sha256=sha(path/'suite.json')) for path in (before,after)],
                  benchmark_source_sha256=sha(Path(__file__)),
                  sampling_source_sha256=sha(Path(webgpu_measure.__code__.co_filename)),
                  timestamp_period_ns=timestamp_period_ns, cases=[])
    with device:
        report['adapter'] = device.info
        if device.info['adapter']['backend_type'] != 'Vulkan' or device.info['adapter']['adapter_type'] != 'DiscreteGPU':
            raise ValueError('requires a physical Vulkan GPU')
        for name in names:
            index = next(i for i,row in enumerate(cases()) if row['name']==name)
            row = cases()[index]
            values, reference = inputs(row,index)
            paths = [directory/suite['cases'][index]['artifact'] for directory,suite in zip((before,after),suites)]
            for path,suite in zip(paths,suites):
                if sha(path) != suite['cases'][index]['artifact_sha256']:
                    raise ValueError('artifact checksum mismatch')
            record = dict(name=name, inputs_sha256=[hashlib.sha256(memoryview(v).cast('B')).hexdigest() for v in values],
                          runs=[])
            for repeat in range(3):
                measured = {}
                order = ('before','after') if repeat%2==0 else ('after','before')
                for label in order:
                    measured[label] = webgpu_measure(device, paths[0 if label=='before' else 1], row, values,
                                                     reference, timestamp_period_ns)
                if measured['before']['correctness']['output_sha256'] != measured['after']['correctness']['output_sha256']:
                    raise ValueError('outputs differ')
                record['runs'].append(dict(order=order, **measured))
                print(name, repeat, {label:round(measured[label]['allocating']['median_ms'],3) for label in order},flush=True)
            report['cases'].append(record)
    report['status'] = 'passed'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before',type=Path); parser.add_argument('after',type=Path)
    parser.add_argument('--case',action='append',required=True)
    parser.add_argument('--timestamp-period-ns',type=float,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    report=run(args.before,args.after,args.case,args.timestamp_period_ns)
    args.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
