"""Run with the isolated wheel consumer's python -I, passing the repo root."""
import argparse
import importlib.abc
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

FORBIDDEN={'torch','tilelang','tvm','tvm_ffi','triton','ggml','llama_cpp'}


class BlockProducerImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in FORBIDDEN:
            raise AssertionError('consumer imported producer/reference dependency: '+fullname)


def run(root,out):
    absent={name:importlib.util.find_spec(name) is None for name in FORBIDDEN}
    if not all(absent.values()):raise AssertionError(absent)
    sys.meta_path.insert(0,BlockProducerImports())
    import numpy as np
    import tensor
    import tensor_llm
    from tensor_llm import LFM2
    root=Path(root).resolve();rows=[]
    for kind in ('f16','q4_0'):
        evidence=root/f'build/lfm2-230m-{kind}-webgpu-validation'
        report=json.loads((evidence/'report.json').read_text())
        with tensor.Device(provider='webgpu') as device,LFM2(
            root/f'build/lfm2-230m-models/LFM2.5-230M-{kind.upper()}.gguf',
            root/f'build/lfm2-230m-{kind}-webgpu',device,context=512) as model:
            np.testing.assert_array_equal(model.forward(report['validation'][0]['tokens']),np.load(evidence/'0-tensor-logits.npy'))
            generated=model.generate('What is 2 + 2?',max_tokens=96)
            assert generated==report['generation']
            rows.append({'format':kind,'generation':generated,'allocated_bytes':model.allocated_bytes,'adapter':device.info['adapter'],
                         'prepared_encoding':device.info.get('prepared_encoding','python')})
    assert not any(name.split('.')[0] in FORBIDDEN for name in sys.modules)
    result={'status':'passed','python':sys.version,'compiler_dependencies_absent':absent,
            'packages':{name:importlib.metadata.version(name) for name in ('tensor-workspace','tensor-llm','numpy','regex','wgpu')},
            'installed_sources':{'tensor':tensor.__file__,'tensor_llm':tensor_llm.__file__},
            'tensor_compiler_wrappers':[name for name in sys.modules if name.startswith('tensor.compiler')],
            'results':rows}
    Path(out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path);args=parser.parse_args();run(args.root,args.out)
