"""Build actual adapter-emitted FX profiles with NVRTC on a GPU-free host."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.fx import Interpreter,symbolic_trace
from torch._subclasses.fake_tensor import FakeTensorMode
import tensor
from tensor_torch.lowering import emit,regions


def affine(a,b):return torch.relu(a*2+b)
def linear(a,w,b):return torch.relu(torch.nn.functional.linear(a,w,b))
def attention(q,k,v):return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True)


class Metadata(Interpreter):
    def run_node(self,node):
        value=super().run_node(node)
        node.meta['val']=value
        return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--target',default='sm_86')
    opts=parser.parse_args()
    opts.out.mkdir(parents=True,exist_ok=True)
    profiles=[('affine',affine,[(129,),(129,)],torch.float32),
              ('linear',linear,[(33,64),(65,64),(65,)],torch.float16),
              ('attention',attention,[(1,2,129,64)]*3,torch.float16)]
    report=[]
    for name,function,shapes,dtype in profiles:
        with FakeTensorMode():
            args=[torch.empty(shape,device='cuda',dtype=dtype) for shape in shapes]
            graph=symbolic_trace(function)
            Metadata(graph).run(*args)
            selected=regions(graph.graph)
            assert len(selected)==1
            root,nodes,external=selected[0]
            inputs=[n.meta['val'] for n in external]
            source,order,kind=emit(nodes,external,inputs)
            assert order==tuple(range(len(inputs)))
        source_path=opts.out/(name+'.py')
        source_path.write_bytes(source.encode())
        artifact=opts.out/(name+'.tbin')
        result=tensor.build(source_path,artifact,target=opts.target,compiler='nvrtc',cache_dir=opts.out/'compiler')
        report.append({'name':name,'kind':kind,'nodes':[n.name for n in nodes],
                       'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                       'artifact_sha256':hashlib.sha256(artifact.read_bytes()).hexdigest(),
                       'build':result})
    (opts.out/'producer.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
