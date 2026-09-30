"""Run a real checkpoint in a compiler/framework-free installed environment."""
import argparse
import importlib.abc
import importlib.metadata
import json
from pathlib import Path
import sys


class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'tilelang','tvm','tvm_ffi','torch','triton','wgpu','gguf','llama_cpp'}:
            raise ImportError('producer/framework import prohibited: '+fullname)


def consume(model,bundle,reference,out):
    sys.meta_path.insert(0,Guard())
    distributions=sorted(d.metadata['Name'].lower() for d in importlib.metadata.distributions())
    assert set(distributions)=={'tensor-workspace','tensor-llm','numpy','regex'},distributions
    import numpy as np
    import tensor
    from tensor_llm import LFM2
    validation=json.loads((reference/'validation.json').read_text())
    if validation['status']!='passed':raise ValueError('requires a successful numerical fixture')
    spec=json.loads((reference/'reference-spec.json').read_text())
    observations=[]
    with tensor.Device() as device,LFM2(model,bundle,device) as network:
        # Repeat the complete numerical fixture, including cache and reset boundaries.
        for i,item in enumerate(spec['validation']):
            if item.get('reset'):network.reset()
            logits=network.forward(item['tokens'])
            expected=np.load(reference/f'tensor-{i}-logits.npy')
            np.testing.assert_array_equal(logits,expected)
            observations.append({'step':i,'position':network.position,'top1':int(np.argmax(logits))})
        generation=network.generate('What is the capital of France?',max_tokens=256)
        report={'schema':'tensor.lfm2-consumer.v1','status':'passed','adapter':device.info,
                'distributions':distributions,'steps':observations,'generation':generation,
                'owned_device_bytes':network.allocated_bytes}
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n')
    print(generation['text'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','bundle','reference','out'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();consume(args.model,args.bundle,args.reference,args.out)
